#!/usr/bin/env python3
"""Step 09C — Benchmark original Ultralytics .pt against Step 09B metrics.

Uses the EXACT same shared evaluation_common.py functions as Step 09B:
  - COCOeval AP50 / AP50:95 at requested max_det
  - class-aware, fixed-confidence P/R/F1 at requested matching IoU

Only the model inference backend differs (.pt Ultralytics instead of LiteRT).
Works with YOLO detection datasets in data.yaml and optionally COCO JSON.
This script does not modify or require any QAT/PTQ checkpoint.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from ptq_dataset_common import images_from_manifest, read_split_images, save_json, select_images
from evaluation_common import (
    COCO80_TO_91, classes_from_yaml, coco_evaluate, coco_gt_per_image,
    fixed_precision_recall, make_yolo_coco, xyxy_to_xywh,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Benchmark original .pt with Step 09B metrics")
    p.add_argument("--model", "--model-pt", dest="model", type=Path, required=True,
                   help="Original Ultralytics .pt model (not a QAT state_dict)")
    p.add_argument("--data", type=Path, help="data.yaml (YOLO detection labels)")
    p.add_argument("--split", choices=("train", "val", "test"), default="test")
    p.add_argument("--manifest", type=Path, help="Optional COCO image subset manifest")
    p.add_argument("--dataset-root", type=Path, help="Image root for --manifest")
    p.add_argument("--annotations", type=Path, help="Optional COCO instances_*.json")
    p.add_argument("--class-map", type=Path, help="Custom COCO class ID mapping JSON")
    p.add_argument("--num-images", type=int, help="Limit evaluation to the first N images")
    p.add_argument("--imgsz", type=int, nargs="+", default=[640], metavar="SIZE",
                   help="Square: 640; rectangular: 352 640")
    p.add_argument("--device", default="cpu", help="Ultralytics device: cpu, 0, cuda:0, ...")
    p.add_argument("--conf", type=float, default=0.001,
                   help="Minimum prediction confidence retained for mAP")
    p.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    p.add_argument("--max-det", type=int, default=300)
    p.add_argument("--pr-conf", type=float, default=0.25,
                   help="Fixed prediction confidence used for Precision/Recall/F1")
    p.add_argument("--match-iou", type=float, default=0.5,
                   help="IoU for matching predictions to GT for P/R/F1")
    p.add_argument("--output-json", type=Path)
    p.add_argument("--predictions-json", type=Path)
    p.add_argument("--min-map50", type=float)
    p.add_argument("--min-map50-95", type=float)
    return p.parse_args(argv)


def _coco_class_map(nc, coco, override):
    valid = set(map(int, coco.getCatIds()))
    if override:
        raw = json.loads(override.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            ids = list(map(int, raw))
        elif isinstance(raw, dict):
            ids = [int(raw[str(i)]) for i in range(nc)]
        else:
            raise ValueError("--class-map must be JSON list or object")
    elif nc == 80 and set(COCO80_TO_91).issubset(valid):
        ids = COCO80_TO_91
    else:
        raise ValueError("COCO annotations for a custom model require --class-map JSON")
    if len(ids) != nc or len(set(ids)) != nc or not set(ids).issubset(valid):
        raise ValueError(f"Invalid class-map for {nc} classes")
    return ids


def _coco_img_ids(paths, coco):
    lookup, dup = {}, set()
    for im in coco.dataset.get("images", []):
        name = Path(im["file_name"]).name
        if name in lookup:
            dup.add(name)
        lookup[name] = int(im["id"])
    result = []
    for path in paths:
        if path.name not in lookup or path.name in dup:
            raise RuntimeError(f"No unique COCO image ID for {path.name}")
        result.append(lookup[path.name])
    if len(set(result)) != len(result):
        raise RuntimeError("Duplicate COCO image IDs")
    return result


def _read_predictions(result, nc):
    """Convert Ultralytics Results.boxes to Step 09B's detection dicts."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.detach().cpu().numpy()
    scores = boxes.conf.detach().cpu().numpy()
    cls = boxes.cls.detach().cpu().numpy()
    out = []
    for box, score, class_id in zip(xyxy, scores, cls):
        c = int(class_id)
        if c < 0 or c >= nc:
            raise ValueError(f"Model predicted class {c} outside [0,{nc})")
        if not np.isfinite(box).all() or not np.isfinite(score):
            raise RuntimeError("Non-finite prediction")
        out.append(dict(class_id=c, bbox_xyxy=[float(v) for v in box], score=float(score)))
    return out


def main(argv=None):
    args = parse_args(argv)
    report = args.output_json or Path("runs/fp32") / args.model.stem / "09c_fp32_metrics.json"
    predictions_path = args.predictions_json or report.with_name("09c_fp32_predictions.json")
    payload = dict(stage="09c_benchmark_fp32", status="EVALUATION_FAIL")
    try:
        if bool(args.data) == bool(args.manifest):
            raise ValueError("Choose exactly one: --data or --manifest")
        if args.manifest and not args.dataset_root:
            raise ValueError("--manifest requires --dataset-root")
        if args.manifest and not args.annotations:
            raise ValueError("--manifest requires --annotations (COCO JSON)")
        if len(args.imgsz) not in (1, 2) or min(args.imgsz) <= 0:
            raise ValueError("--imgsz must be one or two positive integers")
        if not (0 <= args.conf <= args.pr_conf <= 1):
            raise ValueError("Require 0 <= --conf <= --pr-conf <= 1")
        if not (0 < args.iou <= 1 and 0 < args.match_iou <= 1 and args.max_det > 10):
            raise ValueError("Invalid IoU/max-det arguments; max-det must exceed 10")
        if args.num_images is not None and args.num_images <= 0:
            raise ValueError("--num-images must be positive")
        if (args.min_map50 is None) != (args.min_map50_95 is None):
            raise ValueError("Provide both --min-map50 and --min-map50-95")

        from ultralytics import YOLO
        from pycocotools.coco import COCO

        original = args.model.expanduser().resolve()
        if not original.is_file() or original.suffix.lower() != ".pt":
            raise FileNotFoundError(f"Expected existing .pt model: {original}")
        model = YOLO(str(original))
        if getattr(model, "task", None) != "detect":
            raise ValueError(f"Expected detection model, got task={model.task!r}")
        nc = len(model.names)
        if nc <= 0:
            raise ValueError("Model has no class names")

        if args.data:
            paths, spec, _ = read_split_images(args.data, args.split)
            if "nc" in spec and int(spec["nc"]) != nc:
                raise ValueError(f"Dataset nc={spec['nc']} != model nc={nc}")
            paths = select_images(paths, args.num_images)
            names = classes_from_yaml(spec, nc)
        else:
            paths = images_from_manifest(args.manifest, args.dataset_root, args.num_images)
            names = [str(i) for i in range(nc)]

        if args.annotations:
            gt = COCO(str(args.annotations))
            category_ids = _coco_class_map(nc, gt, args.class_map)
            ids = _coco_img_ids(paths, gt)
            truths = coco_gt_per_image(gt, ids, category_ids)
            mode = "coco"
        else:
            if not args.data:
                raise ValueError("YOLO-label benchmark requires --data")
            ids = list(range(1, len(paths) + 1))
            category_ids = list(range(1, nc + 1))
            mode = "yolo"
            dimensions = []
            truths = {}

        raw_detections, predictions, latencies = {}, [], []
        image_size = args.imgsz[0] if len(args.imgsz) == 1 else tuple(args.imgsz)
        for idx, (path, image_id) in enumerate(zip(paths, ids), start=1):
            # Passing a file path avoids ambiguous RGB/BGR NumPy-array conventions.
            # rect=False ensures rectangular source images are padded to declared
            # model input shape, like LiteRT's fixed-size letterbox preprocessing.
            start = time.perf_counter()
            results = model.predict(source=str(path), imgsz=image_size, device=args.device,
                                    conf=args.conf, iou=args.iou, max_det=args.max_det,
                                    rect=False, half=False, augment=False,
                                    verbose=False, save=False, stream=False)
            elapsed_ms = (time.perf_counter() - start) * 1000
            if len(results) != 1:
                raise RuntimeError(f"Expected one result for {path}, got {len(results)}")
            pred = results[0]
            dets = _read_predictions(pred, nc)
            raw_detections[image_id] = dets
            if mode == "yolo":
                h, w = pred.orig_shape
                dimensions.append((int(w), int(h)))
            for det in dets:
                predictions.append(dict(image_id=int(image_id),
                                        category_id=int(category_ids[det["class_id"]]),
                                        bbox=xyxy_to_xywh(det["bbox_xyxy"]),
                                        score=float(det["score"])))
            latencies.append(elapsed_ms)
            if idx % 50 == 0 or idx == len(paths):
                print(f"Progress {idx}/{len(paths)} | detections={len(predictions)} "
                      f"| total_predict={elapsed_ms:.2f}ms", flush=True)

        if mode == "yolo":
            dataset, truths = make_yolo_coco(paths, dimensions, nc, names)
            gt = COCO()
            gt.dataset = dataset
            gt.createIndex()

        save_json(predictions_path, predictions)
        metrics = coco_evaluate(gt, predictions, ids, category_ids, args.max_det)
        fixed = fixed_precision_recall(truths, raw_detections, nc, conf=args.pr_conf,
                                       match_iou=args.match_iou, names=names)
        for cls in fixed["per_class"]:
            ap = metrics["ap_per_class"][str(category_ids[cls["class_id"]])]
            cls.update(ap50=ap["ap50"], ap50_95=ap["ap50_95"])

        gate_used = args.min_map50 is not None
        accepted = (metrics["map50"] >= args.min_map50 and
                    metrics["map50_95"] >= args.min_map50_95) if gate_used else False
        status = "ACCURACY_PASS" if accepted else ("ACCURACY_FAIL" if gate_used else "EVALUATION_PASS")
        payload.update(status=status, model=str(original), model_size_bytes=original.stat().st_size,
                       dataset_mode=mode, split=args.split if args.data else None,
                       images=len(paths), detections=len(predictions), classes=nc,
                       image_size=args.imgsz, device=args.device, conf=args.conf,
                       nms_iou=args.iou, max_det=args.max_det, pr_conf=args.pr_conf,
                       match_iou=args.match_iou, precision=fixed["precision"],
                       recall=fixed["recall"], f1=fixed["f1"], map50=metrics["map50"],
                       map50_95=metrics["map50_95"], metrics=metrics,
                       per_class=fixed["per_class"], classification_metrics=fixed,
                       total_predict_ms_mean=float(np.mean(latencies)),
                       total_predict_ms_p50=float(np.percentile(latencies, 50)),
                       total_predict_ms_p95=float(np.percentile(latencies, 95)),
                       predictions_json=str(predictions_path))
        print(f"\n{status}: P={fixed['precision']:.5f} R={fixed['recall']:.5f} "
              f"F1={fixed['f1']:.5f} mAP50={metrics['map50']:.5f} "
              f"mAP50-95={metrics['map50_95']:.5f}")
        print("NOTE: PT total_predict latency includes Ultralytics overhead; it is not LiteRT invoke latency.")
    except Exception as exc:
        payload["error"] = f"{type(exc).__name__}: {exc}"
        print("EVALUATION_FAIL:", payload["error"], file=sys.stderr)
    finally:
        save_json(report, payload)
        print(f"Metrics JSON: {report}")
    return 0 if payload["status"] in ("EVALUATION_PASS", "ACCURACY_PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
