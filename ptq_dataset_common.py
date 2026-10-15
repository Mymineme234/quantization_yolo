#!/usr/bin/env python3
"""PTQ-only dataset/representative utilities. No dependency on QAT code.

Uses Ultralytics YAML `path`, `train`, `val`, `test` conventions; file lists and
image directories are supported. Calibration always uses train by default.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Sequence

import numpy as np

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as fp:
        for block in iter(lambda: fp.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path: Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def read_yaml(path: Path) -> tuple[dict, Path]:
    import yaml
    path = Path(path).expanduser().resolve()
    spec = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(spec, dict):
        raise ValueError(f'Expected YAML mapping: {path}')
    root = Path(spec.get('path') or path.parent).expanduser()
    if not root.is_absolute():
        root = path.parent / root
    root = root.resolve()
    if 'nc' in spec and 'names' in spec and len(spec['names']) != int(spec['nc']):
        raise ValueError('data.yaml names count does not match nc')
    return spec, root


def _all_images(directory: Path) -> list[Path]:
    return sorted(p.resolve() for p in directory.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def _parse_txt(txt: Path, root: Path) -> list[Path]:
    lines = [line.strip() for line in txt.read_text(encoding='utf-8').splitlines()
             if line.strip() and not line.lstrip().startswith('#')]
    paths = []
    for line in lines:
        p = Path(line).expanduser()
        if not p.is_absolute():
            # Ultralytics split files typically use ./images/... relative to the .txt parent.
            # Other common exports use dataset-root relative paths.
            candidates = [txt.parent / p, root / p]
            matches = [c for c in candidates if c.is_file()]
            if not matches:
                raise FileNotFoundError(f'Image {line} listed in {txt} does not exist')
            p = matches[0]
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            raise RuntimeError(f'Invalid image in {txt}: {p}')
        paths.append(p.resolve())
    return paths


def read_split_images(data_yaml: Path, split: str) -> tuple[list[Path], dict, Path]:
    if split not in {'train', 'val', 'test'}:
        raise ValueError('split must be train, val or test')
    spec, root = read_yaml(data_yaml)
    definition = spec.get(split)
    if definition is None:
        raise KeyError(f'data.yaml does not declare {split!r} split')
    entries = definition if isinstance(definition, list) else [definition]
    paths = []
    for entry in entries:
        if not isinstance(entry, str):
            raise TypeError(f'data.yaml {split} must be path or list of paths')
        path = Path(entry).expanduser()
        path = (path if path.is_absolute() else root / path).resolve()
        if path.is_dir():
            paths.extend(_all_images(path))
        elif path.is_file() and path.suffix.lower() == '.txt':
            paths.extend(_parse_txt(path, root))
        elif path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            paths.append(path)
        else:
            raise FileNotFoundError(f'No image directory/list for {split}: {path}')
    if not paths:
        raise RuntimeError(f'data.yaml {split} split contains no images')
    if len(paths) != len(set(paths)):
        raise RuntimeError(f'data.yaml {split} contains duplicate images')
    return paths, spec, root


def select_images(paths: Sequence[Path], count: int | None, *, seed: int = 42,
                  shuffle: bool = False) -> list[Path]:
    result = list(paths)
    if shuffle:
        random.Random(seed).shuffle(result)
    if count is not None:
        if count <= 0:
            raise ValueError('--calib-samples/--num-images must be > 0')
        if count > len(result):
            raise RuntimeError(f'Requested {count} images; only {len(result)} available')
        result = result[:count]
    return result


def images_from_manifest(manifest: Path, root: Path, count: int | None = None) -> list[Path]:
    manifest = Path(manifest).expanduser().resolve()
    root = Path(root).expanduser().resolve()
    paths = _parse_txt(manifest, root)
    if len(paths) != len(set(paths)):
        raise RuntimeError('Duplicate calibration/benchmark images in manifest')
    return select_images(paths, count)


def labels_for_image(image_path: Path) -> Path:
    """Mirror Ultralytics' `/images/` -> `/labels/` label convention."""
    p = Path(image_path).resolve()
    parts = list(p.parts)
    matches = [i for i, s in enumerate(parts[:-1]) if s == 'images']
    if not matches:
        raise RuntimeError(f'YOLO labels require images/ counterpart: {p}')
    parts[matches[-1]] = 'labels'
    return Path(*parts).with_suffix('.txt')


def image_to_gt(image_path: Path, width: int, height: int, nc: int) -> list[dict]:
    labels = labels_for_image(image_path)
    if not labels.exists():
        # Ultralytics treats missing label files as background images.
        return []
    result = []
    for number, line in enumerate(labels.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        cols = line.split()
        if len(cols) != 5:
            raise ValueError(f'Only YOLO detection labels supported: {labels}:{number} ({len(cols)} columns)')
        c = int(cols[0]); xywh = [float(x) for x in cols[1:]]
        if c < 0 or c >= nc or any(not np.isfinite(x) for x in xywh):
            raise ValueError(f'Invalid label {labels}:{number}')
        cx, cy, w, h = xywh
        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
            raise ValueError(f'Out-of-range YOLO label {labels}:{number}')
        x1, y1 = max(0., cx - w / 2), max(0., cy - h / 2)
        x2, y2 = min(1., cx + w / 2), min(1., cy + h / 2)
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f'Empty clipped bbox {labels}:{number}')
        result.append({'class_id': c, 'bbox_xyxy': [x1 * width, y1 * height, x2 * width, y2 * height]})
    return result


def create_representative_npy(paths: Sequence[Path], hw: tuple[int, int], dst: Path,
                              *, progress: int = 100) -> dict:
    """Create big [N,H,W,3] float32 NPY with memmap (no multi-GB Python list)."""
    import cv2
    from ptq_raw_head_postprocess import letterbox
    if not paths:
        raise ValueError('Calibration images cannot be empty')
    height, width = map(int, hw)
    n = len(paths)
    dst = Path(dst).resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)
    arr = np.lib.format.open_memmap(str(dst), mode='w+', dtype='float32',
                                    shape=(n, height, width, 3))
    try:
        for i, path in enumerate(paths):
            bgr = cv2.imread(str(path))
            if bgr is None:
                raise RuntimeError(f'Unreadable calibration image: {path}')
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            padded, _, _, _ = letterbox(rgb, height, width)
            arr[i] = padded.astype(np.float32) / 255.
            if (i + 1) % progress == 0 or i + 1 == n:
                print(f'Calibration images processed: {i + 1}/{n}', flush=True)
        arr.flush()
    except Exception:
        del arr
        dst.unlink(missing_ok=True)
        raise
    del arr
    return dict(path=str(dst), count=n, shape=[n, height, width, 3],
                dtype='float32', sha256=sha256(dst))


def validate_representative_npy(path: Path, hw: tuple[int, int]) -> int:
    arr = np.load(path, mmap_mode='r', allow_pickle=False)
    if arr.ndim != 4 or tuple(arr.shape[1:]) != (hw[0], hw[1], 3) or arr.dtype != np.float32 or len(arr) < 1:
        raise ValueError(f'Expected nonempty NHWC float32 [N,{hw[0]},{hw[1]},3], got {arr.shape} {arr.dtype}')
    for i in range(0, len(arr), 16):
        part = arr[i:i+16]
        if not np.isfinite(part).all() or float(part.min()) < 0 or float(part.max()) > 1.00001:
            raise ValueError(f'Invalid normalized calibration data near sample {i}')
    return len(arr)
