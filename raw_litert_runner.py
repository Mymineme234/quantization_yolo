#!/usr/bin/env python3
"""Safe shared loader for Step 09 sanity and benchmark scripts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from raw_head_postprocess import (infer_int8, letterbox, make_interpreter,
                                  postprocess, raw_decode, read_metadata)


def hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_verified_model(prepared_dir: Path, *, threads: int = 1):
    meta = read_metadata(prepared_dir)
    base = Path(prepared_dir)
    convert = json.loads((base / '07_convert_raw_litert_int8.json').read_text(encoding='utf-8'))
    audit = json.loads((base / '08_audit_raw_litert.json').read_text(encoding='utf-8'))
    if convert.get('status') != 'CONVERSION_PASS' or audit.get('status') != 'STRUCTURAL_PASS':
        raise RuntimeError('Must pass Step 07 conversion and Step 08 full INT8 structural audit first')
    if convert.get('source_onnx_sha256') != meta['onnx_sha256']:
        raise RuntimeError('Step 07 is stale: re-convert after Step 06 ONNX changed')
    path = Path(convert['model_path'])
    digest = hash_file(path)
    if digest != convert['model_sha256'] or digest != audit['model_sha256']:
        raise RuntimeError('LiteRT model changed after structural audit')
    if int(meta['nc']) <= 0:
        raise RuntimeError('Invalid class count')
    return meta, path, make_interpreter(path, threads=threads)


def detect_image(itp, rgb, meta, *, conf, iou, max_det, method=None):
    h, w = map(int, meta['imgsz_hw'])
    padded, ratio, left, top = letterbox(rgb, h, w)
    reg, cls, latency_ms = infer_int8(itp, padded, meta)
    box, probs = raw_decode(reg, cls, meta)
    if method is None or method == 'auto':
        method = 'topk' if meta['family'] == 'yolo26' and meta['branch'] == 'one2one' else 'nms'
    dets = postprocess(box, probs, conf=conf, iou=iou, max_det=max_det, method=method,
                       ratio=ratio, pad_left=left, pad_top=top,
                       image_height=rgb.shape[0], image_width=rgb.shape[1])
    return dets, latency_ms


def rgb_image(path: Path):
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise RuntimeError(f'Cannot open image: {path}')
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def draw_detections(rgb, dets):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    for d in dets:
        x1, y1, x2, y2 = (int(round(v)) for v in d['bbox_xyxy'])
        cv2.rectangle(bgr, (x1, y1), (x2, y2), (50, 210, 30), 2)
        cv2.putText(bgr, f'{d["class_id"]} {d["score"]:.2f}',
                    (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, .5, (50, 210, 30), 1)
    return bgr
