#!/usr/bin/env python3
"""Step 02c: prepare structural QAT for Ultralytics YOLO26 detection.

Unlike YOLOv8/11, baseline YOLO26 uses reg_max=1 (no DFL),
end2end=True, and both one2many and one2one training heads. The shared
placement handles quantized Conv2d in BOTH heads; this is NOT an export
adapter and DOES NOT make raw-DFL assumptions about regression outputs.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from qat_prepare_common import build_parser, main_for_family, prepare_qat  # noqa: E402


def prepare_qat_yolo26(model, *, imgsz, device, allow_custom_wrappers=False, sanity_forward=True):
    patched, audit, _ = prepare_qat(
        model, family="yolo26", imgsz=imgsz, device=device,
        allow_custom_wrappers=allow_custom_wrappers, sanity_forward=sanity_forward,
    )
    return patched, audit


if __name__ == "__main__":
    raise SystemExit(main_for_family(build_parser("yolo26").parse_args(), "yolo26", "02c_prepare_qat_yolo26"))
