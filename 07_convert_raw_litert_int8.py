#!/usr/bin/env python3
"""Step 07: onnx2tf full-integer quantization of verified raw-head ONNX.

Run this script from a separate onnx2tf environment (e.g. .venv-convert).
Does NOT require torch, Ultralytics or ModelOpt. Uses the proven -oiqt recipe.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from raw_head_postprocess import image_paths, letterbox, read_metadata


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def representative_array(manifest: Path, dataset_root: Path, count: int, hw: tuple[int, int]) -> np.ndarray:
    import cv2
    paths = image_paths(manifest, dataset_root, count)
    if len(paths) != count:
        raise RuntimeError(f'Insufficient calibration images: requested {count}, manifest has {len(paths)}')
    values = []
    for p in paths:
        bgr = cv2.imread(str(p))
        if bgr is None:
            raise RuntimeError(f'Unreadable calibration image: {p}')
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image, _, _, _ = letterbox(rgb, *hw)
        values.append(image.astype(np.float32) / 255.)
    return np.stack(values, axis=0)


def validate_npy(path: Path, hw: tuple[int, int]):
    samples = np.load(path, mmap_mode='r', allow_pickle=False)
    expected = (hw[0], hw[1], 3)
    if samples.ndim != 4 or samples.shape[1:] != expected:
        raise RuntimeError(f'Expected NHWC [N,{hw[0]},{hw[1]},3], got {samples.shape}')
    if samples.shape[0] < 1 or samples.dtype != np.float32:
        raise RuntimeError('Calibration NPY must contain float32 nonempty NHWC images')
    if not np.isfinite(samples).all() or samples.min() < 0 or samples.max() > 1.00001:
        raise RuntimeError('Calibration image values must be finite and normalized to [0,1]')
    return int(samples.shape[0])


def args_parser():
    p = argparse.ArgumentParser(description='Raw-head ONNX -> full-integer LiteRT via onnx2tf')
    p.add_argument('--prepared-dir', required=True, type=Path)
    p.add_argument('--calib-npy', type=Path, help='Existing NHWC float32 [0,1] calibration NPY')
    p.add_argument('--calib-manifest', type=Path, help='Create calibration NPY from image manifest')
    p.add_argument('--dataset-root', type=Path, help='Resolve relative image paths in manifest')
    p.add_argument('--calib-samples', type=int, default=64)
    p.add_argument('--output-dir', type=Path, help='Defaults to <prepared-dir>/07_litert_int8')
    p.add_argument('--force', action='store_true', help='Allow replacement of existing Step 07 conversion directory')
    return p.parse_args()


def main():
    args = args_parser()
    work = args.prepared_dir
    summary = work / '07_convert_raw_litert_int8.json'
    result = dict(stage='07_convert_raw_litert_int8', status='FAIL')
    try:
        if args.calib_samples <= 0:
            raise ValueError('--calib-samples must be positive')
        meta = read_metadata(work)
        source = Path(meta['onnx_path'])
        if not source.is_file() or sha256(source) != meta['onnx_sha256']:
            raise RuntimeError('Step 06 ONNX is missing or its hash changed')
        hw = tuple(int(x) for x in meta['imgsz_hw'])
        if bool(args.calib_npy) == bool(args.calib_manifest):
            raise ValueError('Provide exactly one of --calib-npy or --calib-manifest')
        if args.calib_npy:
            calib_path = args.calib_npy.resolve()
            if not calib_path.is_file():
                raise FileNotFoundError(calib_path)
        else:
            if not args.dataset_root:
                raise ValueError('--dataset-root required with --calib-manifest')
            calib_path = (work / '07_representative_nhwc_float32.npy').resolve()
            samples = representative_array(args.calib_manifest, args.dataset_root, args.calib_samples, hw)
            calib_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(calib_path, samples)
            print('Prepared calibration array:', calib_path, samples.shape, flush=True)
        num_samples = validate_npy(calib_path, hw)
        out = (args.output_dir or work / '07_litert_int8').resolve()
        if out.exists() and any(out.iterdir()):
            if not args.force:
                raise RuntimeError(f'Nonempty conversion directory {out}; pass --force to replace')
            shutil.rmtree(out)
        out.mkdir(parents=True, exist_ok=True)
        command = [
            'onnx2tf', '-i', str(source), '-o', str(out), '-tb', 'tf_converter',
            '-oiqt', '-iqd', 'int8', '-oqd', 'int8', '-cind', 'images',
            str(calib_path), '[[[[0.0,0.0,0.0]]]]', '[[[[1.0,1.0,1.0]]]]',
        ]
        log_file = work / '07_onnx2tf_convert.log'
        print('Converting:', ' '.join(command), flush=True)
        with log_file.open('w', encoding='utf-8') as f:
            proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, bufsize=1)
            for line in proc.stdout:
                sys.stdout.write(line)
                f.write(line)
            exit_code = proc.wait()
        result.update(source_onnx_sha256=meta['onnx_sha256'],
                      source_checkpoint_sha256=meta['checkpoint_sha256'],
                      conversion_command=command, converter_exit_code=exit_code,
                      representative_file=str(calib_path), representative_images=num_samples,
                      representative_sha256=sha256(calib_path), output_directory=str(out))
        if exit_code != 0:
            raise RuntimeError(f'onnx2tf conversion failed with exit code {exit_code}, see {log_file}')
        candidates = list(out.rglob('*full_integer_quant.tflite'))
        if len(candidates) != 1:
            raise RuntimeError(f'Expected exactly one full_integer_quant.tflite, found {candidates}')
        tflite = candidates[0].resolve()
        # Only the *audit* Step 08 can certify full INT8 runtime.
        result.update(status='CONVERSION_PASS', model_path=str(tflite),
                      model_sha256=sha256(tflite), size_bytes=tflite.stat().st_size,
                      note='Conversion PASS is not STRUCTURAL_PASS. Run Step 08.')
        print(f'CONVERSION_PASS | {tflite} | {tflite.stat().st_size / 1048576:.2f} MiB')
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
        print(f'CONVERSION_FAIL: {result["error"]}', file=sys.stderr)
    finally:
        write_json(summary, result)
    return 0 if result['status'] == 'CONVERSION_PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
