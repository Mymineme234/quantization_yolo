#!/usr/bin/env python3
"""Step 02b: prepare post-activation QAT for YOLO11 detection.

Supports DFL (reg_max>1), non-end2end Detect and YOLO11 Conv/DWConv
blocks including the depthwise classification branch. C2PSA attention
requires separate non-Conv INT8 export validation in a later stage.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from qat_prepare_common import build_parser, main_for_family, prepare_qat  # noqa: E402


def prepare_qat_yolo11(model, *, imgsz, device, allow_custom_wrappers=False, sanity_forward=True):
    patched, audit, _ = prepare_qat(
        model, family="yolo11", imgsz=imgsz, device=device,
        allow_custom_wrappers=allow_custom_wrappers, sanity_forward=sanity_forward,
    )
    return patched, audit


if __name__ == "__main__":
    raise SystemExit(main_for_family(build_parser("yolo11").parse_args(), "yolo11", "02b_prepare_qat_yolo11"))
