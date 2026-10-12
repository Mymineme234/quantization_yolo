#!/usr/bin/env python3
"""Shared, architecture-aware host postprocessing for raw-head full-INT8 LiteRT.

Graph: raw reg (4*reg_max) and raw cls logits (nc); host: DFL/anchor decode,
sigmoid and either class-aware NMS or end-to-end top-k. No float ops in LiteRT.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np


def read_metadata(prepared_dir: Path) -> dict:
    p = Path(prepared_dir) / '06_raw_head_metadata.json'
    if not p.is_file():
        raise FileNotFoundError(f'Missing Step 06 metadata: {p}')
    meta = json.loads(p.read_text(encoding='utf-8'))
    if meta.get('status') != 'EXPORT_PASS' or meta.get('output_contract') != 'raw_reg_cls_v1':
        raise RuntimeError(f'Step 06 export is not verified: {p}')
    return meta


def anchors_and_strides(meta: dict) -> tuple[np.ndarray, np.ndarray]:
    shapes = meta['feature_shapes']
    strides = meta['strides']
    if len(shapes) != len(strides) or not shapes:
        raise ValueError('Invalid feature map metadata')
    a, s = [], []
    for (h, w), stride in zip(shapes, strides):
        h, w, stride = int(h), int(w), float(stride)
        if min(h, w, stride) <= 0:
            raise ValueError('Invalid feature shape / stride')
        yy, xx = np.meshgrid(np.arange(h, dtype=np.float32) + .5,
                             np.arange(w, dtype=np.float32) + .5, indexing='ij')
        a.append(np.stack([xx, yy], -1).reshape(-1, 2))
        s.append(np.full((h * w,), stride, dtype=np.float32))
    return np.concatenate(a, axis=0), np.concatenate(s, axis=0)


def raw_to_bcn(array: np.ndarray, channels: int, positions: int) -> np.ndarray:
    a = np.asarray(array)
    if a.ndim != 3 or a.shape[0] != 1:
        raise RuntimeError(f'Expected batch=1 output, got {a.shape}')
    if a.shape[1:] == (channels, positions):
        return a.astype(np.float32)
    if a.shape[1:] == (positions, channels):
        return a.transpose(0, 2, 1).astype(np.float32)
    raise RuntimeError(f'Expected [1,{channels},{positions}] (or transposed), got {a.shape}')


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x.astype(np.float32), -50, 50)
    return 1. / (1. + np.exp(-x))


def raw_decode(reg: np.ndarray, cls: np.ndarray, meta: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return boxes [N,4] xywh input pixel coordinates, probabilities [N,nc]."""
    reg_max, nc = int(meta['reg_max']), int(meta['nc'])
    anchors, strides = anchors_and_strides(meta)
    n = len(strides)
    r = raw_to_bcn(reg, 4 * reg_max, n)
    c = raw_to_bcn(cls, nc, n)
    if not np.isfinite(r).all() or not np.isfinite(c).all():
        raise RuntimeError('NaN/Inf in raw model outputs')
    if reg_max > 1:
        r = r.reshape(1, 4, reg_max, n)
        r = r - r.max(axis=2, keepdims=True)
        exp = np.exp(r)
        p = exp / exp.sum(axis=2, keepdims=True)
        bins = np.arange(reg_max, dtype=np.float32).reshape(1, 1, reg_max, 1)
        dist = (p * bins).sum(axis=2)[0].T
    else:
        dist = r[0].T
    left_top = anchors - dist[:, :2]
    right_bottom = anchors + dist[:, 2:]
    xywh = np.concatenate([(left_top + right_bottom) / 2,
                           right_bottom - left_top], axis=1) * strides[:, None]
    return xywh.astype(np.float32), sigmoid(c[0].T)


def letterbox(rgb: np.ndarray, height: int, width: int):
    import cv2
    h, w = rgb.shape[:2]
    ratio = min(height / h, width / w)
    nw, nh = round(w * ratio), round(h * ratio)
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dw, dh = (width - nw) / 2, (height - nh) / 2
    left, right = round(dw - 0.1), round(dw + 0.1)
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
    if padded.shape[:2] != (height, width):
        raise RuntimeError(f'Invalid letterbox shape {padded.shape}')
    return padded, float(ratio), int(left), int(top)


def iou_one(b: np.ndarray, other: np.ndarray) -> np.ndarray:
    lt = np.maximum(b[:2], other[:, :2])
    rb = np.minimum(b[2:], other[:, 2:])
    inter = np.maximum(rb - lt, 0).prod(axis=1)
    area_b = np.maximum(b[2:] - b[:2], 0).prod()
    area_o = np.maximum(other[:, 2:] - other[:, :2], 0).prod(axis=1)
    return inter / (area_b + area_o - inter + 1e-9)


def nms_for_class(boxes: np.ndarray, scores: np.ndarray, iou: float) -> np.ndarray:
    indices = np.argsort(-scores, kind='stable')
    out = []
    while indices.size:
        i = int(indices[0])
        out.append(i)
        indices = indices[1:]
        if indices.size:
            indices = indices[iou_one(boxes[i], boxes[indices]) <= iou]
    return np.asarray(out, dtype=np.int64)


def postprocess(boxes_xywh: np.ndarray, scores: np.ndarray, *,
                conf: float, iou: float, max_det: int, method: str,
                ratio: float, pad_left: int, pad_top: int,
                image_width: int, image_height: int) -> list[dict]:
    """YOLOv8/11 class-aware NMS; YOLO26 end2end top-k without NMS."""
    if max_det <= 0 or not (0 <= conf <= 1) or not (0 <= iou <= 1):
        raise ValueError('Invalid postprocess thresholds')
    n, nc = scores.shape
    if boxes_xywh.shape != (n, 4):
        raise ValueError('Shape mismatch in postprocessing')
    if method == 'topk':
        # Ultralytics end2end: first keep at most max_det best anchors,
        # then choose max_det class scores from this subset (may repeat anchors).
        top_anchor = np.argsort(-scores.max(axis=1), kind='stable')[:max_det]
        sub_scores = scores[top_anchor]
        top_flat = np.argsort(-sub_scores.ravel(), kind='stable')[:max_det]
        anchor_ids, classes = top_anchor[top_flat // nc], top_flat % nc
        selected_scores = scores[anchor_ids, classes]
        selected = [(int(i), int(c), float(s)) for i, c, s in
                    zip(anchor_ids, classes, selected_scores) if s >= conf]
    elif method == 'nms':
        classes = scores.argmax(axis=1)
        values = scores.max(axis=1)
        candidates = np.flatnonzero(values >= conf)
        boxes = xywh_to_xyxy(boxes_xywh)
        selected = []
        for c in np.unique(classes[candidates]):
            ids = candidates[classes[candidates] == c]
            keep = nms_for_class(boxes[ids], values[ids], iou)
            selected.extend((int(ids[k]), int(c), float(values[ids[k]])) for k in keep)
        selected.sort(key=lambda item: item[2], reverse=True)
        selected = selected[:max_det]
    else:
        raise ValueError(f'Unknown postprocess mode: {method}')
    boxes = xywh_to_xyxy(boxes_xywh)
    detections = []
    for i, c, score in selected:
        x1, y1, x2, y2 = boxes[i]
        x1, x2 = np.clip([(x1 - pad_left) / ratio, (x2 - pad_left) / ratio], 0, image_width)
        y1, y2 = np.clip([(y1 - pad_top) / ratio, (y2 - pad_top) / ratio], 0, image_height)
        if x2 <= x1 or y2 <= y1:
            continue
        detections.append(dict(class_id=c, score=float(score), bbox_xyxy=[float(x1), float(y1), float(x2), float(y2)]))
    return detections


def xywh_to_xyxy(xywh: np.ndarray) -> np.ndarray:
    out = np.empty_like(xywh, dtype=np.float32)
    out[:, :2] = xywh[:, :2] - xywh[:, 2:] / 2
    out[:, 2:] = xywh[:, :2] + xywh[:, 2:] / 2
    return out


def image_paths(manifest: Path, dataset_root: Path, limit: int | None = None) -> list[Path]:
    manifest, dataset_root = Path(manifest), Path(dataset_root)
    lines = [s.strip() for s in manifest.read_text(encoding='utf-8').splitlines() if s.strip() and not s.lstrip().startswith('#')]
    if limit is not None:
        lines = lines[:limit]
    if not lines:
        raise RuntimeError(f'Empty image manifest: {manifest}')
    paths = [(Path(s) if Path(s).is_absolute() else dataset_root / s).resolve() for s in lines]
    absent = [str(p) for p in paths if not p.is_file()]
    if absent:
        raise FileNotFoundError(f'Manifest includes {len(absent)} missing images, e.g. {absent[:3]}')
    return paths


def quant_params(t: dict) -> tuple[float, int]:
    q = t.get('quantization_parameters', {})
    scales = np.asarray(q.get('scales', []))
    zeros = np.asarray(q.get('zero_points', []))
    if scales.size != 1 or zeros.size != 1 or not np.isfinite(scales[0]) or scales[0] <= 0:
        raise RuntimeError(f'Output/input tensor missing per-tensor quantization: {t.get("name")}')
    return float(scales[0]), int(zeros[0])


def make_interpreter(path: Path, *, threads: int = 1):
    from ai_edge_litert.interpreter import Interpreter
    try:
        from ai_edge_litert.interpreter import OpResolverType
        itp = Interpreter(model_path=str(path), num_threads=threads,
                          experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    except (ImportError, AttributeError):
        itp = Interpreter(model_path=str(path), num_threads=threads)
    itp.allocate_tensors()
    return itp


def infer_int8(itp, rgb_letterboxed: np.ndarray, meta: dict):
    inputs = itp.get_input_details()
    outputs = itp.get_output_details()
    if len(inputs) != 1 or len(outputs) != 2:
        raise RuntimeError('Expected exactly one input and two outputs')
    inp = inputs[0]
    if np.dtype(inp['dtype']) != np.dtype('int8'):
        raise RuntimeError(f'Input must be int8, got {inp["dtype"]}')
    h, w = [int(v) for v in meta['imgsz_hw']]
    data = rgb_letterboxed.astype(np.float32) / 255.
    shape = tuple(map(int, inp['shape']))
    if shape == (1, h, w, 3):
        data = data[None]
    elif shape == (1, 3, h, w):
        data = np.transpose(data, (2, 0, 1))[None]
    else:
        raise RuntimeError(f'Unexpected LiteRT input shape: {shape}')
    scale, zp = quant_params(inp)
    data = np.clip(np.rint(data / scale + zp), -128, 127).astype(np.int8)
    itp.set_tensor(inp['index'], data)
    start = time.perf_counter()
    itp.invoke()
    latency_ms = (time.perf_counter() - start) * 1000.0
    n = sum(int(a) * int(b) for a, b in meta['feature_shapes'])
    reg_ch, cls_ch = 4 * int(meta['reg_max']), int(meta['nc'])
    matched = {}
    for detail in outputs:
        if np.dtype(detail['dtype']) != np.dtype('int8'):
            raise RuntimeError('Non-INT8 LiteRT output')
        shape = tuple(map(int, detail['shape']))
        matches = [kind for kind, channels in [('reg', reg_ch), ('cls', cls_ch)]
                   if shape in ((1, channels, n), (1, n, channels))]
        if len(matches) != 1 or matches[0] in matched:
            # If class count == regression channels, the shape cannot distinguish
            # semantic outputs. Require names that retain export labels.
            name = detail['name'].lower()
            matches = [kind for kind in matches if kind in name or ('class' in name and kind == 'cls')]
        if len(matches) != 1 or matches[0] in matched:
            raise RuntimeError(f'Ambiguous LiteRT output mapping {detail["name"]}: {shape}')
        scale, zp = quant_params(detail)
        raw = itp.get_tensor(detail['index']).astype(np.float32)
        matched[matches[0]] = (raw - zp) * scale
    if set(matched) != {'reg', 'cls'}:
        raise RuntimeError('Unable to map both raw outputs')
    return matched['reg'], matched['cls'], latency_ms
