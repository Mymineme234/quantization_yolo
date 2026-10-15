#!/usr/bin/env python3
"""Shared metrics for QAT/PTQ Step 09: COCO AP + fixed-threshold P/R/F1.

No direct QAT import. COCOeval metric tensors are averaged at actual max_det,
never from COCOeval's fixed maxDets=100 summarize() shortcut.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from ptq_dataset_common import image_to_gt

COCO80_TO_91 = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
    46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
    67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
]


def classes_from_yaml(spec: dict, nc: int) -> list[str]:
    names = spec.get('names')
    if isinstance(names, dict):
        return [str(names[i] if i in names else names[str(i)]) for i in range(nc)]
    if isinstance(names, list) and len(names) == nc:
        return [str(x) for x in names]
    return [str(i) for i in range(nc)]


def xyxy_to_xywh(box):
    x1, y1, x2, y2 = map(float, box)
    return [x1, y1, x2 - x1, y2 - y1]


def make_yolo_coco(paths: list[Path], dims: list[tuple[int, int]], nc: int, names: list[str]):
    """Create COCO-like ground truth, preserving empty/negative images."""
    images, annotations, gts = [], [], {}
    ann_id = 1
    for i, (path, (w, h)) in enumerate(zip(paths, dims), start=1):
        images.append(dict(id=i, file_name=str(path), width=w, height=h))
        truth = image_to_gt(path, w, h, nc)
        gts[i] = truth
        for gt in truth:
            xywh = xyxy_to_xywh(gt['bbox_xyxy'])
            annotations.append(dict(id=ann_id, image_id=i, category_id=gt['class_id'] + 1,
                                    bbox=xywh, area=xywh[2] * xywh[3], iscrowd=0))
            ann_id += 1
    if not annotations:
        raise RuntimeError('Selected split contains no YOLO detection annotations; cannot evaluate AP')
    dataset = dict(info={'description': 'Synthetic COCO ground truth built from YOLO labels'},
                   licenses=[], images=images, annotations=annotations,
                   categories=[dict(id=i+1, name=names[i]) for i in range(nc)])
    return dataset, gts


def coco_gt_per_image(coco, image_ids: list[int], category_ids: list[int]):
    id_to_index = {int(c): i for i, c in enumerate(category_ids)}
    result = {}
    for i in image_ids:
        ann_ids = coco.getAnnIds(imgIds=[int(i)], iscrowd=False)
        anns = coco.loadAnns(ann_ids)
        gts = []
        for ann in anns:
            cat = int(ann['category_id'])
            if cat not in id_to_index or bool(ann.get('ignore', 0)):
                continue
            x, y, w, h = ann['bbox']
            gts.append(dict(class_id=id_to_index[cat], bbox_xyxy=[x, y, x+w, y+h]))
        result[i] = gts
    return result


def bbox_iou(box1, box2) -> float:
    a, b = np.asarray(box1, dtype=np.float64), np.asarray(box2, dtype=np.float64)
    overlap = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]))
    intersection = overlap.prod()
    area_a = np.maximum(0, a[2:] - a[:2]).prod()
    area_b = np.maximum(0, b[2:] - b[:2]).prod()
    return float(intersection / (area_a + area_b - intersection + 1e-9))


def fixed_precision_recall(
    truths: dict[int, list[dict]], detections: dict[int, list[dict]],
    nc: int, *, conf: float = 0.25, match_iou: float = 0.50,
    names: list[str] | None = None,
):
    """Greedy one-to-one matching, per image and class, at declared thresholds.

    Ground-truth crowd/ignore COCO semantics are handled by COCOeval for AP;
    this fixed-threshold P/R/F1 calculation ignores those cases by convention.
    """
    totals = np.zeros((nc, 3), dtype=np.int64)  # tp fp fn
    for image_id, gt in truths.items():
        preds = sorted([x for x in detections.get(image_id, []) if x['score'] >= conf],
                       key=lambda d: d['score'], reverse=True)
        for c in range(nc):
            this_gt = [x for x in gt if x['class_id'] == c]
            this_pred = [x for x in preds if x['class_id'] == c]
            used = set()
            for pred in this_pred:
                candidate, best_iou = None, match_iou - 1e-12
                for j, actual in enumerate(this_gt):
                    if j in used:
                        continue
                    overlap = bbox_iou(pred['bbox_xyxy'], actual['bbox_xyxy'])
                    if overlap >= match_iou and overlap > best_iou:
                        candidate, best_iou = j, overlap
                if candidate is not None:
                    used.add(candidate)
                    totals[c, 0] += 1
                else:
                    totals[c, 1] += 1
            totals[c, 2] += len(this_gt) - len(used)
    by_class = []
    for c, (tp, fp, fn) in enumerate(totals):
        p = float(tp / (tp+fp)) if tp+fp else 0.
        r = float(tp / (tp+fn)) if tp+fn else 0.
        f = 2*p*r/(p+r) if p+r else 0.
        by_class.append(dict(class_id=c, class_name=names[c] if names else str(c),
                             tp=int(tp), fp=int(fp), fn=int(fn),
                             precision=p, recall=r, f1=f, support=int(tp+fn)))
    tp, fp, fn = [int(x) for x in totals.sum(axis=0)]
    p = tp/(tp+fp) if tp+fp else 0.
    r = tp/(tp+fn) if tp+fn else 0.
    mask = (totals[:, 0] + totals[:, 2]) > 0
    return dict(confidence=float(conf), match_iou=float(match_iou),
                precision=float(p), recall=float(r), f1=float(2*p*r/(p+r) if p+r else 0),
                average='micro', tp=tp, fp=fp, fn=fn,
                macro_precision=float(np.mean([c['precision'] for c, flag in zip(by_class, mask) if flag])) if mask.any() else 0.,
                macro_recall=float(np.mean([c['recall'] for c, flag in zip(by_class, mask) if flag])) if mask.any() else 0.,
                macro_f1=float(np.mean([c['f1'] for c, flag in zip(by_class, mask) if flag])) if mask.any() else 0.,
                per_class=by_class)


def coco_evaluate(coco_gt, predictions: list[dict], image_ids: list[int],
                  class_ids: list[int], max_det: int):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    if not coco_gt.dataset.get('annotations'):
        raise RuntimeError('No ground truth annotations in chosen evaluation images')
    if predictions:
        detection_gt = coco_gt.loadRes(predictions)
    else:
        detection_gt = COCO()
        detection_gt.dataset = dict(info={}, images=coco_gt.dataset['images'],
                                    categories=coco_gt.dataset['categories'], annotations=[])
        detection_gt.createIndex()
    evaluator = COCOeval(coco_gt, detection_gt, 'bbox')
    evaluator.params.imgIds = sorted(image_ids)
    evaluator.params.catIds = list(class_ids)
    evaluator.params.maxDets = [1, 10, max_det]
    evaluator.evaluate(); evaluator.accumulate()
    p = evaluator.eval['precision']  # [IoU, recall, category, area, maxDet]
    r = evaluator.eval['recall']
    max_index = evaluator.params.maxDets.index(max_det)
    areas = evaluator.params.areaRngLbl
    all_area = areas.index('all')
    ious = np.asarray(evaluator.params.iouThrs)

    def average(data):
        valid = data[data > -1]
        return float(valid.mean()) if valid.size else None

    # COCOeval sorts catIds internally; preserve correct per-class indexing even
    # if a custom --class-map lists categories in a different order.
    actual_cat_ids = list(map(int, evaluator.params.catIds))

    def ap(iou=None, area='all', c_index=None):
        t = p if iou is None else p[np.argmin(abs(ious - iou)):np.argmin(abs(ious - iou))+1]
        if iou is None:
            t = p
        t = t[:, :, :, areas.index(area), max_index]
        if c_index is not None:
            t = t[:, :, c_index]
        return average(t)

    metrics = dict(map50_95=ap(), map50=ap(iou=0.5), map75=ap(iou=0.75),
                   map_small=ap(area='small'), map_medium=ap(area='medium'),
                   map_large=ap(area='large'),
                   mar_1=average(r[:, :, all_area, evaluator.params.maxDets.index(1)]),
                   mar_10=average(r[:, :, all_area, evaluator.params.maxDets.index(10)]),
                   mar_max_det=average(r[:, :, all_area, max_index]),
                   ap_per_class={str(c): {'ap50': ap(0.5, c_index=actual_cat_ids.index(int(c))),
                                         'ap50_95': ap(c_index=actual_cat_ids.index(int(c)))}
                                 for c in class_ids})
    if metrics['map50'] is None or metrics['map50_95'] is None:
        raise RuntimeError('COCOeval AP is undefined (check labels and category IDs)')
    return metrics
