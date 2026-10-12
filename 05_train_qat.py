#!/usr/bin/env python3
"""Step 05: train an already-calibrated QAT graph (YOLOv8/11/26 detection).

Uses a pinned compatible Ultralytics DetectionTrainer. Fresh optimizer each run;
--init-checkpoint continues weights, not optimizer/scheduler state. Export is a
separate stage and can choose architecture-specific raw-head contracts.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from qat_stage_common import (
    activation_quantizers, check_calibration_meta, extract_metrics, fail_closed,
    file_sha256, load_prepared_spec, parse_hw, restore_qat_from_artifact,
    seed_all, torch_device, validate_amax, validate_dataset_classes,
    metric_delta, read_json, write_state,
)


def parse_args():
    p = argparse.ArgumentParser(description="Step 05: QAT fine-tuning from Step 03 calibrated checkpoint")
    p.add_argument("--prepared-dir", type=Path, required=True)
    p.add_argument("--calibrated-checkpoint", type=Path, help="Default prepared-dir/03_calibrated_state_dict.pt")
    p.add_argument("--init-checkpoint", type=Path, help="Optional prior Step 05 weights; optimizer/scheduler restart")
    p.add_argument("--data", type=Path, required=True, help="Ultralytics dataset.yaml with train/val")
    p.add_argument("--output", type=Path, help="Defaults to prepared-dir")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", "--lr0", dest="lr", type=float, default=1e-5)
    p.add_argument("--lrf", type=float, default=1.0, help="Final LR multiplier in Ultralytics scheduler")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--optimizer", choices=("AdamW", "Adam", "SGD"), default="AdamW")
    p.add_argument("--warmup-epochs", type=float, default=0.0)
    p.add_argument("--imgsz", nargs="+", type=int, help="Consistency check against prepared image size")
    p.add_argument("--device", default="auto")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    bn = p.add_mutually_exclusive_group()
    bn.add_argument("--freeze-bn", action="store_true", default=True, help="Keep pretrained BN running statistics (default)")
    bn.add_argument("--update-bn", action="store_false", dest="freeze_bn")
    p.add_argument("--augment", action="store_true", help="Enable standard Ultralytics augmentations; off by default")
    p.add_argument("--validate-after", action="store_true", help="Evaluate trained fake-Q on data.yaml val split")
    p.add_argument("--conf", type=float, default=0.001)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--max-det", type=int, default=300)
    return p.parse_args()


def make_trainer(qat_model: nn.Module, freeze_bn: bool):
    from ultralytics.models.yolo.detect import DetectionTrainer

    class QATDetectionTrainer(DetectionTrainer):
        def __init__(self, *args, **kwargs):
            self.prepared_qat_model = qat_model
            super().__init__(*args, **kwargs)

        def get_model(self, cfg=None, weights=None, verbose=True):
            # Do not rebuild QAT model from YAML, which loses injected quantizers.
            return self.prepared_qat_model

        def preprocess_batch(self, batch):
            if freeze_bn:
                # Ultralytics calls model.train() per epoch; restore eval mode for
                # BN running stats before each forward. Affine grads remain enabled.
                for module in self.model.modules():
                    if isinstance(module, nn.modules.batchnorm._BatchNorm):
                        module.eval()
            return super().preprocess_batch(batch)

        def final_eval(self):
            # Do not reload best.pt; this can re-insert ModelOpt on an already
            # quantized object. Step 04/--validate-after is the independent gate.
            return None

    return QATDetectionTrainer


def no_aug_overrides() -> dict:
    return {"mosaic": 0.0, "mixup": 0.0, "copy_paste": 0.0,
            "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0,
            "degrees": 0.0, "translate": 0.0, "scale": 0.0,
            "shear": 0.0, "perspective": 0.0, "flipud": 0.0,
            "fliplr": 0.0}


def main() -> int:
    args = parse_args()
    output = args.output or args.prepared_dir

    def run(logger, result):
        if not args.data.is_file():
            raise FileNotFoundError(args.data)
        if args.epochs <= 0 or args.batch <= 0 or args.lr <= 0 or args.weight_decay < 0:
            raise ValueError("Invalid epochs/batch/lr/weight-decay")
        if not (0 < args.lrf <= 1):
            raise ValueError("--lrf must be in (0,1]")
        if args.warmup_epochs < 0:
            raise ValueError("--warmup-epochs cannot be negative")
        spec = load_prepared_spec(args.prepared_dir, imgsz_override=parse_hw(args.imgsz) if args.imgsz else None)
        hw = spec["imgsz_hw"]
        data_info = validate_dataset_classes(args.data, int(spec["meta"]["head"]["nc"]))
        if hw[0] != hw[1]:
            raise RuntimeError("Ultralytics DetectionTrainer presently uses square imgsz; rectangular QAT training requires a separately verified trainer/preprocessor")
        device = torch_device(args.device)
        if device.type == "cpu":
            logger.warning("CPU QAT training can be very slow")
        seed_all(args.seed)
        calibrated_ckpt = args.calibrated_checkpoint or (args.prepared_dir / "03_calibrated_state_dict.pt")
        # Verify calibration artifact first even if loading previously trained weights.
        _, _, calibrated, _ = restore_qat_from_artifact(
            spec, device=device, checkpoint=calibrated_ckpt, seed=args.seed)
        check_calibration_meta(calibrated, spec, calibrated_ckpt)
        init_ckpt = args.init_checkpoint or calibrated_ckpt
        yolo, qat_model, init_meta, head = restore_qat_from_artifact(
            spec, device=device, checkpoint=init_ckpt, seed=args.seed)
        check_calibration_meta(init_meta, spec, init_ckpt)
        if init_meta["calib_sha256"] != calibrated["calib_sha256"]:
            raise RuntimeError("Training init checkpoint derives from different calibration data")
        validate_amax(activation_quantizers(qat_model))
        del yolo
        # ModelOpt's state and loss must survive trainer initialization intact.
        trainer_type = make_trainer(qat_model, args.freeze_bn)
        overrides = {
            "model": str(spec["model"]), "data": str(args.data.resolve()),
            "epochs": args.epochs, "imgsz": hw[0], "batch": args.batch,
            "device": str(device), "workers": args.workers,
            "optimizer": args.optimizer, "lr0": args.lr, "lrf": args.lrf,
            "weight_decay": args.weight_decay,
            "warmup_epochs": args.warmup_epochs,
            "warmup_bias_lr": 0.0 if args.warmup_epochs == 0 else args.lr,
            "warmup_momentum": 0.0 if args.warmup_epochs == 0 else 0.8,
            "amp": False, "val": True, "save": False,
            "plots": False, "verbose": True, "seed": args.seed,
            "project": str(output.resolve()), "name": "ultralytics_qat", "exist_ok": True,
        }
        if not args.augment:
            overrides.update(no_aug_overrides())
        result.update({"family": spec["family"], "data": str(args.data.resolve()),
                       "imgsz_hw": list(hw), "device": str(device),
                       "calibrated_checkpoint": str(calibrated_ckpt), "init_checkpoint": str(init_ckpt),
                       **data_info,
                       "calib_sha256": calibrated["calib_sha256"],
                       "training_overrides": overrides, "freeze_bn": args.freeze_bn,
                       "warning": "Fresh optimizer/scheduler on every run, including --init-checkpoint"})
        logger.info("QAT training | family=%s | epochs=%d | batch=%d | lr=%g",
                    spec["family"], args.epochs, args.batch, args.lr)
        trainer = trainer_type(overrides=overrides)
        trainer.train()
        trained = trainer.model
        if not isinstance(trained, nn.Module):
            raise RuntimeError("Ultralytics trainer did not retain QAT model")
        before = set(name for name, _ in activation_quantizers(qat_model))
        after = set(name for name, _ in activation_quantizers(trained))
        if before != after:
            raise RuntimeError("Quantizer topology changed during training")
        validate_amax(activation_quantizers(trained))
        trained.eval()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        checkpoint_path = output / "05_qat_trained_state_dict.pt"
        write_state(checkpoint_path, trained, family=spec["family"],
                    base_sha256=file_sha256(spec["model"]), calib_sha256=calibrated["calib_sha256"],
                    calibrated=True, imgsz_hw=list(hw),
                    epochs_this_run=args.epochs,
                    epoch_completed=args.epochs + int(init_meta.get("epoch_completed", 0)),
                    optimizer_state_saved=False, head=head, seed=args.seed,
                    training_overrides=overrides)
        result["trained_checkpoint"] = str(checkpoint_path)
        result["ultralytics_run_dir"] = str(getattr(trainer, "save_dir", output / "ultralytics_qat"))
        # Check trainable values before advertising TRAINING_PASS.
        for name, parameter in trained.named_parameters():
            if not bool(torch.isfinite(parameter).all()):
                raise RuntimeError(f"Non-finite trained weight: {name}")
        result["epoch_completed"] = args.epochs + int(init_meta.get("epoch_completed", 0))
        result["quantizer_count"] = len(after)
        if args.validate_after:
            from ultralytics import YOLO
            wrapper = YOLO(str(spec["model"]), task="detect")
            wrapper.model = trained.to(device).eval()
            val = wrapper.val(data=str(args.data.resolve()), imgsz=hw[0], batch=args.batch,
                              device=str(device), workers=args.workers,
                              conf=args.conf, iou=args.iou, max_det=args.max_det,
                              plots=False, save=False, verbose=False)
            result["trained_fake_q_metrics"] = extract_metrics(val)
            prior_path = args.prepared_dir / "04_pretrain_validation.json"
            if prior_path.is_file():
                prior = read_json(prior_path)
                if (prior.get("status") == "VALIDATION_PASS" and
                    prior.get("dataset_yaml_sha256") == data_info["dataset_yaml_sha256"] and
                    prior.get("calib_sha256") == calibrated["calib_sha256"]):
                    result["vs_pretrain_fake_q"] = metric_delta(prior["fake_q"], result["trained_fake_q_metrics"])
                    result["vs_fp32"] = metric_delta(prior["fp32"], result["trained_fake_q_metrics"])
                else:
                    logger.warning("Step 04 baseline dataset/calibration mismatch; no misleading metric comparison")
        result["status"] = "TRAINING_PASS"
        logger.info("Saved QAT trained state: %s", checkpoint_path)

    return fail_closed("05_train_qat", output, run)


if __name__ == "__main__":
    raise SystemExit(main())
