#!/usr/bin/env python3
"""Step 02a: prepare post-activation QAT for YOLOv8 detection (DFL head).

Input: Ultralytics YOLOv8-family .pt. Output: STRUCTURAL audit and prepared
state_dict, NOT calibration, trained QAT, export, or deployable INT8 model.
"""
from __future__ import annotations

# Allow standalone execution `python pipeline/02a_prepare_qat_yolov8.py` as well
# as importlib-based test loading without installing the package.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from qat_prepare_common import (  # noqa: E402
    PostActQuant, QuantAudit, apply_post_activation_policy, build_modelopt_int8,
    build_parser, main_for_family, prepare_qat, q_enabled, set_quant_enabled,
)


def prepare_qat_yolov8(model, *, imgsz, device, allow_custom_wrappers=False, sanity_forward=True):
    patched, audit, _ = prepare_qat(
        model, family="yolov8", imgsz=imgsz, device=device,
        allow_custom_wrappers=allow_custom_wrappers, sanity_forward=sanity_forward,
    )
    return patched, audit


if __name__ == "__main__":
    raise SystemExit(main_for_family(build_parser("yolov8").parse_args(), "yolov8", "02a_prepare_qat_yolov8"))
