#!/usr/bin/env python3
"""Inspect an Ultralytics detection checkpoint and emit a JSON model inventory.

A filename-based family hint is *not* an architectural proof. For renamed
custom models specify --family explicitly at the QAT preparation stage.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--family", choices=("auto", "yolov8", "yolo11", "yolov10", "yolo26", "custom"), default="auto")
    p.add_argument("--output", type=Path, default=Path("runs/qat/inspect"))
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def family_hint(path: Path) -> str:
    name = path.name.lower()
    if re.search(r"yolov10", name):
        return "yolov10"
    if re.search(r"yolo11", name):
        return "yolo11"
    if re.search(r"yolov8", name):
        return "yolov8"
    if re.search(r"yolo26", name):
        return "yolo26"
    return "unknown"


def scalar_or_list(value):
    try:
        import torch
        if torch.is_tensor(value):
            return value.detach().cpu().flatten().tolist()
    except ImportError:
        pass
    if isinstance(value, (tuple, list)):
        return list(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def inspect(model, checkpoint: Path, family: str) -> dict:
    import torch
    import torch.nn as nn
    raw = model.model
    if getattr(raw, "task", "detect") != "detect":
        raise ValueError("Only detection models are in scope for this YOLO QAT pipeline")
    module_rows = []
    head_rows = []
    for name, module in raw.named_modules():
        if isinstance(module, nn.Conv2d):
            module_rows.append({
                "name": name, "class": type(module).__name__,
                "in_channels": module.in_channels, "out_channels": module.out_channels,
                "kernel_size": list(module.kernel_size), "groups": module.groups,
            })
        if hasattr(module, "nc") and hasattr(module, "reg_max"):
            head_rows.append({
                "name": name, "class": type(module).__name__,
                "nc": scalar_or_list(module.nc),
                "reg_max": scalar_or_list(module.reg_max),
                "stride": scalar_or_list(getattr(module, "stride", None)),
                "end2end": bool(getattr(module, "end2end", False)),
                "one2one": hasattr(module, "one2one_cv2") or hasattr(module, "one2one_cv3"),
            })
    yaml_meta = getattr(raw, "yaml", None)
    if not isinstance(yaml_meta, dict):
        yaml_meta = {}
    return {
        "checkpoint": str(checkpoint),
        "family": family if family != "auto" else family_hint(checkpoint),
        "family_provenance": "user_declared" if family != "auto" else "filename_hint_only",
        "model_class": type(raw).__name__,
        "task": getattr(raw, "task", "unknown"),
        "nc": scalar_or_list(getattr(raw, "nc", None)),
        "names": {str(k): v for k, v in getattr(raw, "names", {}).items()},
        "parameters": sum(p.numel() for p in raw.parameters()),
        "yaml_keys": sorted(str(k) for k in yaml_meta),
        "conv2d_count": len(module_rows),
        "conv2d_modules": module_rows,
        "head_candidates": head_rows,
        "note": "Family from filename is not proof; custom graphs require separate structural compatibility checks.",
    }


def main(args: argparse.Namespace) -> int:
    args.output.mkdir(parents=True, exist_ok=True)
    log_file = args.output / "01_inspect_model.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, encoding="utf-8")],
        force=True,
    )
    if not args.model.is_file():
        logging.error("Model missing: %s", args.model)
        return 1
    try:
        from ultralytics import YOLO
        wrapper = YOLO(str(args.model), task="detect")
        wrapper.model.to(args.device).eval()
        info = inspect(wrapper, args.model, args.family)
        path = args.output / "01_model_info.json"
        path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
        logging.info("Model family: %s (%s)", info["family"], info["family_provenance"])
        logging.info("Conv2d: %s; heads: %s; parameters: %s", info["conv2d_count"], len(info["head_candidates"]), info["parameters"])
        logging.info("Saved: %s", path)
        return 0
    except Exception:
        logging.exception("Model inspection failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
