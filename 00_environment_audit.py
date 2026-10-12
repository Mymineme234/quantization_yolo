#!/usr/bin/env python3
"""Shared PyTorch/CUDA/Ultralytics/ModelOpt preflight audit.

This is a *preflight* only. CUDA execution and imports do not prove that a
particular YOLO model is supported or can be converted to LiteRT INT8.
"""
from __future__ import annotations

import argparse
import importlib
import json
import logging
import platform
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("qat.preflight")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, encoding="utf-8")):
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


def version_of(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("runs/qat/preflight"), help="Directory for log + JSON")
    p.add_argument("--device", default="auto", help="auto, cpu, cuda or cuda:N")
    p.add_argument("--require-cuda", action="store_true", help="FAIL if selected device is not CUDA")
    p.add_argument("--skip-compute", action="store_true", help="Skip the tiny device compute test")
    return p.parse_args()


def run(args: argparse.Namespace) -> int:
    args.output.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(args.output / "00_environment_audit.log")
    result: dict = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "versions": {p: version_of(p) for p in ("torch", "ultralytics", "nvidia-modelopt", "onnx")},
        "checks": {},
        "errors": [],
        "status": "FAIL",
    }
    try:
        import torch
        result["torch_cuda_build"] = torch.version.cuda
        result["cudnn_version"] = torch.backends.cudnn.version()
        result["cuda_available"] = bool(torch.cuda.is_available())
        result["cuda_device_count"] = int(torch.cuda.device_count())
        result["gpu_devices"] = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                result["gpu_devices"].append({
                    "index": i,
                    "name": props.name,
                    "compute_capability": list(torch.cuda.get_device_capability(i)),
                    "vram_mib": round(props.total_memory / (1024 ** 2), 1),
                })

        choice = args.device
        if choice == "auto":
            choice = "cuda:0" if torch.cuda.is_available() else "cpu"
        if choice == "cuda":
            choice = "cuda:0"
        device = torch.device(choice)
        result["device"] = str(device)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA selected but torch.cuda.is_available() is False")
            if device.index is not None and device.index >= torch.cuda.device_count():
                raise RuntimeError(f"Device {device} not present")
        if args.require_cuda and device.type != "cuda":
            raise RuntimeError("--require-cuda requested, but no CUDA device selected")

        from ultralytics import YOLO  # noqa: F401
        result["checks"]["ultralytics_import"] = "PASS"
        mtq = importlib.import_module("modelopt.torch.quantization")
        required = ["quantize", "INT8_DEFAULT_CFG"]
        missing = [name for name in required if not hasattr(mtq, name)]
        if missing:
            raise RuntimeError(f"ModelOpt API missing: {missing}")
        result["checks"]["modelopt_import_and_api"] = "PASS"

        if not args.skip_compute:
            with torch.no_grad():
                x = torch.randn(1, 3, 32, 32, device=device)
                layer = torch.nn.Conv2d(3, 4, 3, padding=1).to(device)
                out = layer(x)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                if out.device.type != device.type or not torch.isfinite(out).all().item():
                    raise RuntimeError("Compute test returned wrong device or NaN/Inf")
            result["checks"]["device_compute"] = "PASS"
        else:
            result["checks"]["device_compute"] = "SKIPPED"
        result["status"] = "PASS"
    except Exception as exc:
        result["errors"].append(f"{type(exc).__name__}: {exc}")
        logger.exception("Preflight failed")
    finally:
        path = args.output / "00_environment_audit.json"
        path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Audit JSON: %s", path)
        logger.info("PREFLIGHT: %s", result["status"])
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
