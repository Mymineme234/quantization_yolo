#!/usr/bin/env python3
"""Step 09B: controlled COCO-format benchmark for INT8 raw-head LiteRT.

Same host DFL/decode + NMS/top-k implementation as Step 09A.
For nonstandard COCO datasets provide explicit class-index -> category-id JSON.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from raw_head_postprocess import image_paths
from raw_litert_runner import detect_image, read_verified_model, rgb_image

# Standard Ultralytics COCO80 class order -> official 91-category IDs.
COCO80_TO_91 = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
    35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
    46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
    56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
    67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
    80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
]


def category_map(meta: dict, coco, override: Path | None):
    nc = int(meta['nc'])
    available = set(coco.getCatIds())
    if override:
        spec = json.loads(override.read_text(encoding='utf-8'))
        if isinstance(spec, dict):
            mapping = [int(spec[str(i)]) for i in range(nc)]
        elif isinstance(spec, list):
            mapping = [int(i) for i in spec]
        else:
            raise TypeError('Class map must be list or {"0": category_id, ...}')
    elif nc == 80 and set(COCO80_TO_91).issubset(available):
        mapping = COCO80_TO_91
    else:
        raise RuntimeError('Nonstandard dataset requires --class-map JSON explicitly; '
                           'never guess the mapping from class indices to COCO category IDs')
    if len(mapping) != nc or len(set(mapping)) != nc or not set(mapping).issubset(available):
        raise ValueError(f'Invalid class map (nc={nc}, available categories={len(available)})')
    return mapping


def manifest_to_coco_ids(paths, coco):
    # image filename equality validated against official COCO annotations.
    exact = {}
    ambiguities = set()
    for img in coco.dataset['images']:
        base = Path(img['file_name']).name
        if base in exact:
            ambiguities.add(base)
        exact[base] = int(img['id'])
    ids = []
    for path in paths:
        base = path.name
        if base not in exact or base in ambiguities:
            raise RuntimeError(f'Manifest image filename not uniquely found in annotation JSON: {base}')
        ids.append(exact[base])
    if len(set(ids)) != len(ids):
        raise RuntimeError('Duplicate COCO images in benchmark manifest')
    return ids


def parse_args():
    p = argparse.ArgumentParser(description='COCOeval for raw-head LiteRT full INT8 detection')
    p.add_argument('--prepared-dir', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--dataset-root', type=Path, required=True)
    p.add_argument('--annotations', type=Path, required=True, help='COCO-format instances_*.json')
    p.add_argument('--class-map', type=Path, help='Custom class index -> COCO category_id JSON (list or object)')
    p.add_argument('--num-images', type=int, default=1000)
    p.add_argument('--conf', type=float, default=.001)
    p.add_argument('--iou', type=float, default=.7)
    p.add_argument('--max-det', type=int, default=300)
    p.add_argument('--method', choices=('auto', 'nms', 'topk'), default='auto')
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--min-map50', type=float, help='Acceptance gate against chosen baseline')
    p.add_argument('--min-map50-95', type=float, help='Acceptance gate against chosen baseline')
    p.add_argument('--output-json', type=Path)
    p.add_argument('--predictions-json', type=Path)
    return p.parse_args()


def main():
    args = parse_args()
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    report_path = args.output_json or args.prepared_dir / '09b_coco_metrics.json'
    predictions_path = args.predictions_json or args.prepared_dir / '09b_coco_predictions.json'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    result = dict(stage='09b_benchmark_raw_litert_coco', status='ACCURACY_FAIL')
    try:
        if args.num_images <= 0:
            raise ValueError('--num-images must be positive')
        meta, model, interpreter = read_verified_model(args.prepared_dir, threads=args.threads)
        paths = image_paths(args.manifest, args.dataset_root, args.num_images)
        if len(paths) != args.num_images:
            raise RuntimeError(f'Asked for exactly {args.num_images} images; found {len(paths)}')
        coco_gt = COCO(str(args.annotations))
        mapping = category_map(meta, coco_gt, args.class_map)
        ids = manifest_to_coco_ids(paths, coco_gt)
        predictions, latencies = [], []
        for i, (path, image_id) in enumerate(zip(paths, ids)):
            image = rgb_image(path)
            detections, invoke_ms = detect_image(interpreter, image, meta,
                                                  conf=args.conf, iou=args.iou,
                                                  max_det=args.max_det, method=args.method)
            latencies.append(invoke_ms)
            for det in detections:
                x1, y1, x2, y2 = det['bbox_xyxy']
                predictions.append(dict(image_id=image_id, category_id=mapping[det['class_id']],
                                        bbox=[float(x1), float(y1), float(x2-x1), float(y2-y1)],
                                        score=float(det['score'])))
            if (i+1) % 50 == 0 or i+1 == len(paths):
                print(f'Progress {i+1}/{len(paths)} | detections={len(predictions)} | invoke={invoke_ms:.2f}ms', flush=True)
        predictions_path.write_text(json.dumps(predictions, indent=2) + '\n', encoding='utf-8')
        if not predictions:
            raise RuntimeError('Zero detections: cannot compute a meaningful COCOeval result')
        coco_det = coco_gt.loadRes(str(predictions_path))
        evaluator = COCOeval(coco_gt, coco_det, iouType='bbox')
        evaluator.params.imgIds = ids
        evaluator.params.maxDets = [1, 10, args.max_det]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        stats = evaluator.stats.tolist()

        # pycocotools COCOeval._summarizeDets() uses a hardcoded
        # maxDets=100 for stats[0], even when params.maxDets=[1, 10, 300].
        # In that configuration stats[0] becomes -1 (undefined), although
        # AP50/AP75 and the other metrics are computed at maxDets=300.
        # Recompute overall AP over IoU 0.50:0.95 directly from the
        # accumulated precision tensor at the *requested* maxDets.
        precision = evaluator.eval.get('precision')
        if precision is None or precision.ndim != 5:
            raise RuntimeError('COCOeval did not produce valid precision tensor')
        maxdet_index = evaluator.params.maxDets.index(args.max_det)
        area_index = evaluator.params.areaRngLbl.index('all')
        selected = precision[:, :, :, area_index, maxdet_index]
        valid_precision = selected[selected > -1]
        if valid_precision.size == 0:
            raise RuntimeError('COCOeval overall AP is undefined at requested max_det')
        stats[0] = float(valid_precision.mean())
        print(f'CORRECTED COCO AP50-95 (maxDets={args.max_det}): {stats[0]:.6f}', flush=True)

        metrics = {
            'map50_95': float(stats[0]), 'map50': float(stats[1]), 'map75': float(stats[2]),
            'map_small': float(stats[3]), 'map_medium': float(stats[4]), 'map_large': float(stats[5]),
            'mar_1': float(stats[6]), 'mar_10': float(stats[7]),
            'mar_max_det': float(stats[8]),
        }
        accepted = args.min_map50 is not None and args.min_map50_95 is not None
        gate_pass = (accepted and metrics['map50'] >= args.min_map50
                     and metrics['map50_95'] >= args.min_map50_95)
        verdict = ('ACCURACY_PASS' if gate_pass else
                   ('ACCURACY_FAIL' if accepted else 'EVALUATION_PASS'))
        result.update(status=verdict,
                      min_map50=args.min_map50, min_map50_95=args.min_map50_95,
                      model_path=str(model), family=meta['family'], branch=meta['branch'],
                      num_images=len(ids), detections=len(predictions),
                      confidence=args.conf, iou=args.iou, max_det=args.max_det,
                      method=args.method, class_map=mapping, metrics=metrics,
                      invoke_ms_mean=float(np.mean(latencies)),
                      invoke_ms_p50=float(np.percentile(latencies, 50)),
                      invoke_ms_p95=float(np.percentile(latencies, 95)),
                      predictions_json=str(predictions_path), model_size_bytes=model.stat().st_size)
        print(f'{verdict}:', json.dumps(metrics, indent=2))
        if not accepted:
            print('NOTE: Provide both --min-map50 and --min-map50-95 for ACCURACY_PASS/FAIL acceptance gate.')
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
        print('EVALUATION_FAIL:', result['error'], file=sys.stderr)
    finally:
        report_path.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    return 0 if result['status'] in ('EVALUATION_PASS', 'ACCURACY_PASS') else 1


if __name__ == '__main__':
    raise SystemExit(main())
