#!/usr/bin/env python3
"""Step 09A: visually inspect INT8 raw-head inference and host decoding."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from raw_head_postprocess import image_paths
from raw_litert_runner import detect_image, draw_detections, read_verified_model, rgb_image


def main():
    p = argparse.ArgumentParser(description='Raw-head INT8 LiteRT sanity visualization')
    p.add_argument('--prepared-dir', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--dataset-root', type=Path, required=True)
    p.add_argument('--num-images', type=int, default=5)
    p.add_argument('--conf', type=float, default=.25)
    p.add_argument('--iou', type=float, default=.45)
    p.add_argument('--max-det', type=int, default=300)
    p.add_argument('--method', choices=('auto', 'nms', 'topk'), default='auto')
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--output-dir', type=Path)
    args = p.parse_args()
    out = args.output_dir or args.prepared_dir / '09a_sanity_images'
    out.mkdir(parents=True, exist_ok=True)
    report = {'stage': '09a_sanity_raw_litert', 'status': 'FUNCTIONAL_FAIL'}
    try:
        if args.num_images <= 0:
            raise ValueError('--num-images must be positive')
        meta, model, interpreter = read_verified_model(args.prepared_dir, threads=args.threads)
        paths = image_paths(args.manifest, args.dataset_root, args.num_images)
        if len(paths) != args.num_images:
            raise RuntimeError(f'Expected {args.num_images} images; manifest has {len(paths)}')
        rows = []
        for i, path in enumerate(paths):
            image = rgb_image(path)
            detections, latency_ms = detect_image(interpreter, image, meta,
                                                   conf=args.conf, iou=args.iou,
                                                   max_det=args.max_det, method=args.method)
            dst = out / f'{i:04d}_{path.stem}.jpg'
            if not cv2.imwrite(str(dst), draw_detections(image, detections)):
                raise RuntimeError(f'Cannot write image {dst}')
            rows.append({'image': str(path), 'output_image': str(dst),
                         'detections': len(detections), 'invoke_ms': latency_ms})
            print(f'[{i+1}/{len(paths)}] detections={len(detections)} invoke={latency_ms:.2f} ms {path}')
        total = sum(r['detections'] for r in rows)
        report.update(status='FUNCTIONAL_PASS' if total else 'FUNCTIONAL_FAIL',
                      model_path=str(model), family=meta['family'], branch=meta['branch'],
                      processed_images=len(rows), total_detections=total,
                      mean_invoke_ms=float(np.mean([r['invoke_ms'] for r in rows])),
                      confidence=args.conf, postprocess=args.method, results=rows)
        if not total:
            report['error'] = 'No detections at this threshold; lower --conf and inspect output'
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        print(report['error'])
    finally:
        (out / 'sanity_report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        print('VERDICT:', report['status'])
    return 0 if report['status'] == 'FUNCTIONAL_PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
