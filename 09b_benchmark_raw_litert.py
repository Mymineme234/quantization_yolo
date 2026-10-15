#!/usr/bin/env python3
"""Step 09B: COCO-native OR Ultralytics YOLO data.yaml evaluation.

Backward compatible with the old --manifest --dataset-root --annotations CLI.
Supports both existing, untouched QAT runtime and new independent PTQ runtime.
Reports fixed-confidence P/R/F1 and correct AP50/AP50:95 at requested max-det.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from raw_head_postprocess import image_paths
from raw_litert_runner import (detect_image as qat_detect, read_verified_model as qat_read,
                               rgb_image as qat_rgb)
from ptq_dataset_common import read_split_images, select_images, save_json
from evaluation_common import (COCO80_TO_91, classes_from_yaml, coco_evaluate,
                                   coco_gt_per_image, fixed_precision_recall,
                                   make_yolo_coco, xyxy_to_xywh)
from raw_head_postprocess import read_metadata
from ptq_raw_head_postprocess import read_verified_ptq_model as ptq_read


def category_map(meta: dict, coco, override: Path | None):
    nc = int(meta['nc'])
    available = set(coco.getCatIds())
    if override:
        source = json.loads(override.read_text(encoding='utf-8'))
        if isinstance(source, list):
            mapping = [int(x) for x in source]
        elif isinstance(source, dict):
            mapping = [int(source[str(i)]) for i in range(nc)]
        else:
            raise TypeError('--class-map must be JSON array or mapping')
    elif nc == 80 and set(COCO80_TO_91).issubset(available):
        mapping = COCO80_TO_91
    else:
        raise RuntimeError('Custom COCO annotations require explicit --class-map JSON; '
                           'for YOLO .txt labels use --data instead')
    if len(mapping) != nc or len(set(mapping)) != nc or not set(mapping).issubset(available):
        raise ValueError(f'Invalid class-to-category mapping for nc={nc}')
    return mapping


def manifest_to_coco_ids(paths, coco):
    lookup, collision = {}, set()
    for im in coco.dataset['images']:
        key = Path(im['file_name']).name
        if key in lookup:
            collision.add(key)
        lookup[key] = int(im['id'])
    result = []
    for p in paths:
        if p.name not in lookup or p.name in collision:
            raise RuntimeError(f'Image {p.name} missing or ambiguous in COCO annotations')
        result.append(lookup[p.name])
    if len(set(result)) != len(result):
        raise RuntimeError('Duplicate image ID in benchmark subset')
    return result


def parse_args():
    p = argparse.ArgumentParser(description='Raw-head LiteRT evaluation: COCO JSON or YOLO data.yaml')
    p.add_argument('--prepared-dir', type=Path, required=True)
    p.add_argument('--manifest', type=Path, help='Legacy image manifest; use with --annotations')
    p.add_argument('--dataset-root', type=Path, help='Root for manifest image paths')
    p.add_argument('--annotations', type=Path, help='Native COCO instances_*.json')
    p.add_argument('--class-map', type=Path, help='COCO category ID mapping JSON for custom COCO')
    p.add_argument('--data', type=Path, help='Ultralytics data.yaml, reads YOLO .txt labels by default')
    p.add_argument('--split', choices=('train', 'val', 'test'), default='test')
    p.add_argument('--num-images', type=int, help='Default: all images with --data, 1000 with legacy manifest')
    p.add_argument('--conf', type=float, default=.001, help='Low detection threshold used for AP curve')
    p.add_argument('--iou', type=float, default=.7, help='NMS IoU threshold')
    p.add_argument('--max-det', type=int, default=300)
    p.add_argument('--method', choices=('auto', 'nms', 'topk'), default='auto')
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--pr-conf', type=float, default=.25, help='Fixed score threshold for Precision/Recall/F1')
    p.add_argument('--match-iou', type=float, default=.50, help='GT matching IoU threshold for Precision/Recall/F1')
    p.add_argument('--min-map50', type=float, help='Gate: minimum mAP50')
    p.add_argument('--min-map50-95', type=float, help='Gate: minimum mAP50-95')
    p.add_argument('--output-json', type=Path)
    p.add_argument('--predictions-json', type=Path)
    return p.parse_args()


def main():
    args = parse_args()
    native = bool(args.annotations)
    legacy = bool(args.manifest)
    prefix = '09b_coco' if native or legacy else '09b'
    report_path = args.output_json or args.prepared_dir / (prefix + '_metrics.json')
    pred_path = args.predictions_json or args.prepared_dir / (prefix + '_predictions.json')
    result = dict(stage='09b_benchmark_raw_litert', status='EVALUATION_FAIL')
    try:
        from pycocotools.coco import COCO
        if not (0 <= args.conf <= args.pr_conf <= 1):
            raise ValueError('Must have 0 <= --conf <= --pr-conf <= 1 so P/R predictions are complete')
        if not (0 < args.match_iou <= 1 and 0 < args.iou <= 1 and args.max_det > 10):
            raise ValueError('Invalid IoU/max-det thresholds; max-det must exceed 10')
        if args.num_images is not None and args.num_images <= 0:
            raise ValueError('--num-images must be positive')
        if bool(args.data) == bool(args.manifest):
            raise ValueError('Choose exactly one of --data or --manifest')
        if args.manifest and not args.dataset_root:
            raise ValueError('--dataset-root required with --manifest')
        if args.annotations and not (args.data or args.manifest):
            raise ValueError('--annotations requires --data or --manifest')
        method = read_metadata(args.prepared_dir).get('quantization_method', 'qat')
        if method not in ('qat', 'ptq'):
            raise RuntimeError(f'Unknown quantization method: {method}')
        read_model = ptq_read if method == 'ptq' else qat_read
        meta, model, interpreter = read_model(args.prepared_dir, threads=args.threads)
        detect_image, rgb_image = qat_detect, qat_rgb
        nc = int(meta['nc'])
        if args.data:
            available, spec, _ = read_split_images(args.data, args.split)
            paths = select_images(available, args.num_images)
            names = classes_from_yaml(spec, nc)
            if 'nc' in spec and int(spec['nc']) != nc:
                raise ValueError(f'data.yaml nc={spec["nc"]} != model nc={nc}')
        else:
            count = args.num_images if args.num_images is not None else 1000
            paths = image_paths(args.manifest, args.dataset_root, count)
            if len(paths) != count:
                raise RuntimeError(f'Expected {count} manifest images, only found {len(paths)}')
            names = [str(i) for i in range(nc)]
        if args.annotations:
            gt = COCO(str(args.annotations))
            category_ids = category_map(meta, gt, args.class_map)
            ids = manifest_to_coco_ids(paths, gt)
            truths = coco_gt_per_image(gt, ids, category_ids)
            mode = 'coco'
        else:
            if not args.data:
                raise ValueError('A YOLO-label benchmark requires --data and --split')
            ids = list(range(1, len(paths) + 1))
            category_ids = list(range(1, nc+1))
            mode = 'yolo'
            truths = None
            images_for_gt = []
            dimensions = []
        raw_detections, predictions, latencies = {}, [], []
        for count, (path, image_id) in enumerate(zip(paths, ids), 1):
            rgb = rgb_image(path)
            if mode == 'yolo':
                dimensions.append((int(rgb.shape[1]), int(rgb.shape[0])))
            detections, latency = detect_image(interpreter, rgb, meta,
                                                conf=args.conf, iou=args.iou,
                                                max_det=args.max_det, method=args.method)
            latencies.append(latency)
            raw_detections[image_id] = detections
            for det in detections:
                predictions.append(dict(image_id=image_id,
                                        category_id=category_ids[int(det['class_id'])],
                                        bbox=xyxy_to_xywh(det['bbox_xyxy']), score=float(det['score'])))
            if count % 50 == 0 or count == len(paths):
                print(f'Progress {count}/{len(paths)} | detections={len(predictions)} | invoke={latency:.2f}ms', flush=True)
        if mode == 'yolo':
            ground_truth, truths = make_yolo_coco(paths, dimensions, nc, names)
            gt = COCO()
            gt.dataset = ground_truth
            gt.createIndex()
        save_json(pred_path, predictions)
        metrics = coco_evaluate(gt, predictions, ids, category_ids, args.max_det)
        matching = fixed_precision_recall(truths, raw_detections, nc, conf=args.pr_conf,
                                          match_iou=args.match_iou, names=names)
        for c in matching['per_class']:
            ap = metrics['ap_per_class'][str(category_ids[c['class_id']])]
            c.update(ap50=ap['ap50'], ap50_95=ap['ap50_95'])
        matching.update(map50=metrics['map50'], map50_95=metrics['map50_95'])
        gate_used = args.min_map50 is not None and args.min_map50_95 is not None
        gate = (gate_used and metrics['map50'] >= args.min_map50
                and metrics['map50_95'] >= args.min_map50_95)
        status = 'ACCURACY_PASS' if gate else ('ACCURACY_FAIL' if gate_used else 'EVALUATION_PASS')
        result.update(status=status, dataset_mode=mode, split=args.split if args.data else None,
                      quantization_method=meta.get('quantization_method', 'qat'),
                      num_images=len(paths), detections=len(predictions), model_path=str(model),
                      family=meta['family'], branch=meta['branch'], class_map=category_ids,
                      confidence=args.conf, pr_conf=args.pr_conf, match_iou=args.match_iou,
                      iou=args.iou, max_det=args.max_det, metrics=metrics,
                      precision=matching['precision'], recall=matching['recall'], f1=matching['f1'],
                      classification_metrics=matching, per_class=matching['per_class'],
                      invoke_ms_mean=float(np.mean(latencies)),
                      invoke_ms_p50=float(np.percentile(latencies, 50)),
                      invoke_ms_p95=float(np.percentile(latencies, 95)),
                      model_size_bytes=model.stat().st_size,
                      predictions_json=str(pred_path))
        print(f'\n{status}: P={matching["precision"]:.5f} R={matching["recall"]:.5f} '
              f'F1={matching["f1"]:.5f} mAP50={metrics["map50"]:.5f} '
              f'mAP50-95={metrics["map50_95"]:.5f}')
        if not gate_used:
            print('NOTE: No --min-map50/--min-map50-95 gate: EVALUATION_PASS means benchmark completed.')
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
        print('EVALUATION_FAIL:', result['error'], file=sys.stderr)
    finally:
        save_json(report_path, result)
    return 0 if result['status'] in ('EVALUATION_PASS', 'ACCURACY_PASS') else 1


if __name__ == '__main__':
    raise SystemExit(main())
