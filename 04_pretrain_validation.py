#!/usr/bin/env python3
"""Step 04: fair FP32 vs calibrated fake-Q validation on the same dataset.

Uses Ultralytics detection evaluator, NOT COCOeval. Match postprocess settings
before comparing; a fixed 1000-image subset should be represented by a YAML
with an appropriately restricted val list.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from qat_stage_common import (
    activation_quantizers, check_calibration_meta, extract_metrics, fail_closed,
    load_prepared_spec, metric_delta, parse_hw, restore_qat_from_artifact,
    torch_device, validate_amax, validate_dataset_classes,
)


def parse_args():
    p = argparse.ArgumentParser(description="Step 04: validate FP32 and calibrated fake-Q models")
    p.add_argument("--prepared-dir", type=Path, required=True)
    p.add_argument("--calibrated-checkpoint", type=Path, help="Default: prepared-dir/03_calibrated_state_dict.pt")
    p.add_argument("--data", type=Path, required=True, help="Ultralytics dataset YAML, with val split defined")
    p.add_argument("--output", type=Path, help="Defaults to prepared-dir")
    p.add_argument("--imgsz", nargs="+", type=int, help="Consistency check with Step 02")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="auto")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--conf", type=float, default=0.001)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--max-det", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output or args.prepared_dir

    def run(logger, result):
        from ultralytics import YOLO
        if not args.data.is_file():
            raise FileNotFoundError(args.data)
        if args.batch <= 0:
            raise ValueError("--batch must be positive")
        spec = load_prepared_spec(args.prepared_dir, imgsz_override=parse_hw(args.imgsz) if args.imgsz else None)
        hw = spec["imgsz_hw"]
        data_info = validate_dataset_classes(args.data, int(spec["meta"]["head"]["nc"]))
        # Ultralytics validation may dynamically resize images for non-square shapes;
        # reject this rather than publishing scores for a different input contract.
        if hw[0] != hw[1]:
            raise RuntimeError("Current Ultralytics validator requires square --imgsz for strict reproducibility; use a fixed-shape evaluator for rectangular inputs")
        device = torch_device(args.device)
        ckpt = args.calibrated_checkpoint or (args.prepared_dir / "03_calibrated_state_dict.pt")
        wrapper, qat_model, state, _ = restore_qat_from_artifact(spec, device=device, checkpoint=ckpt, seed=args.seed)
        check_calibration_meta(state, spec, ckpt)
        quantizers = activation_quantizers(qat_model)
        validate_amax(quantizers)
        result.update({"family": spec["family"], "data": str(args.data.resolve()),
                       "calibrated_checkpoint": str(ckpt), "imgsz_hw": list(hw),
                       **data_info,
                       "input_contract": "square_ultralytics_val", "device": str(device),
                       "calib_sha256": state["calib_sha256"],
                       "settings": {"conf": args.conf, "iou": args.iou, "max_det": args.max_det,
                                    "batch": args.batch, "workers": args.workers}})
        val_kwargs = dict(data=str(args.data.resolve()), imgsz=hw[0], batch=args.batch,
                          device=str(device), workers=args.workers, conf=args.conf,
                          iou=args.iou, max_det=args.max_det, plots=False,
                          save=False, verbose=False)
        logger.info("Validating FP32 base model on %s", args.data)
        fp32 = YOLO(str(spec["model"]), task="detect")
        fp32_results = extract_metrics(fp32.val(**val_kwargs))
        logger.info("Validating calibrated fake-Q model on same dataset/split/settings")
        wrapper.model = qat_model.eval()
        qat_results = extract_metrics(wrapper.val(**val_kwargs))
        deltas = metric_delta(fp32_results, qat_results)
        result.update({"status": "VALIDATION_PASS", "fp32": fp32_results,
                       "fake_q": qat_results, **deltas,
                       "note": "Ultralytics metrics; do not compare as directly identical to COCOeval with different postprocessing"})
        logger.info("FP32: %s", fp32_results)
        logger.info("Fake-Q: %s", qat_results)
        logger.info("Delta: %s", deltas)

    return fail_closed("04_pretrain_validation", output, run)


if __name__ == "__main__":
    raise SystemExit(main())
