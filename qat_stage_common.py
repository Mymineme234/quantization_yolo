#!/usr/bin/env python3
"""Shared artifact I/O for numbered QAT stages 03-05.

DO NOT import Ultralytics/ModelOpt eagerly: smoke unit tests run without them.
Loading is restricted to artifacts produced by this pipeline and trusted .pt models.
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from qat_prepare_common import prepare_qat, q_enabled, validate_detection_head

FAMILY_STEP = {"yolov8": "02a_prepare_qat_yolov8", "yolo11": "02b_prepare_qat_yolo11", "yolo26": "02c_prepare_qat_yolo26"}
STAGE_SCHEMA = 1


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    temp.replace(path)


def logger_for(output: Path, stage: str) -> logging.Logger:
    output.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger(f"qat.{stage}")
    log.setLevel(logging.INFO)
    log.propagate = False
    for old in log.handlers:
        old.close()
    log.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(output / f"{stage}.log", encoding="utf-8")):
        handler.setFormatter(formatter)
        log.addHandler(handler)
    return log


def parse_hw(values: list[int]) -> tuple[int, int]:
    if len(values) == 1:
        hw = (values[0], values[0])
    elif len(values) == 2:
        hw = tuple(values)
    else:
        raise ValueError("--imgsz accepts one integer or H W")
    if any(n <= 0 or n % 32 != 0 for n in hw):
        raise ValueError(f"Image dimensions must be positive multiples of 32: {hw}")
    return hw


def torch_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda:0" if torch.cuda.is_available() else "cpu"
    elif value.isdecimal():
        value = f"cuda:{value}"
    dev = torch.device(value)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA unavailable: {dev}")
    return dev


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def starting_payload(stage: str) -> dict[str, Any]:
    return {"stage": stage, "schema_version": STAGE_SCHEMA,
            "started_utc": datetime.now(timezone.utc).isoformat(), "status": "FAIL"}


def load_prepared_spec(prepared_dir: Path, *, model_override: Path | None = None,
                       imgsz_override: tuple[int, int] | None = None) -> dict[str, Any]:
    """Read Step 02 machine-readable artifact and reject incompatible overrides."""
    audits = [prepared_dir / f"{step}_quant_audit.json" for step in FAMILY_STEP.values()]
    valid = [p for p in audits if p.is_file()]
    if len(valid) != 1:
        raise RuntimeError(f"Expected one Step 02 audit JSON in {prepared_dir}, found {len(valid)}")
    meta = read_json(valid[0])
    if meta.get("status") != "STRUCTURAL_PASS":
        raise RuntimeError("Step 02 structural audit has not passed")
    family = meta.get("family")
    if family not in FAMILY_STEP:
        raise RuntimeError(f"Unsupported family in Step 02 artifact: {family}")
    step = FAMILY_STEP[family]
    state_path = prepared_dir / f"{step}_prepared_state_dict.pt"
    if not state_path.is_file():
        raise FileNotFoundError(f"Step 02 prepared state missing: {state_path}. Re-run without --no-save-state")
    model_path = Path(meta["model"]).expanduser()
    if not model_path.is_absolute():
        # Step 02 records --model, often relative to repo root.
        model_path = (Path.cwd() / model_path).resolve()
    if model_override is not None:
        override = model_override.expanduser().resolve()
        if override != model_path.resolve():
            raise RuntimeError("--model does not match Step 02 base model (refuse cross-model restore)")
    if not model_path.is_file():
        raise FileNotFoundError(f"Base model referenced by Step 02 is unavailable: {model_path}")
    hw = tuple(meta["imgsz_hw"])
    if imgsz_override is not None and tuple(imgsz_override) != hw:
        raise RuntimeError(f"--imgsz {imgsz_override} differs from Step 02 {hw}; rerun Step 02")
    return {"meta": meta, "family": family, "model": model_path, "imgsz_hw": hw,
            "state": state_path, "step": step, "audit": valid[0]}


def tensor_state(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict) and isinstance(payload.get("state_dict"), dict):
        return payload["state_dict"]
    # ModelOpt may register _extra_state with non-tensor contents. A raw
    # Step 02 state_dict can contain these; reject unrelated checkpoint maps.
    if (isinstance(payload, dict) and payload and
            all(isinstance(k, str) for k in payload) and
            any(torch.is_tensor(v) for v in payload.values())):
        return payload
    raise TypeError("Expected a state_dict or checkpoint with 'state_dict'")


def materialize_amax_buffers(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    """ModelOpt may declare _amax as a None buffer until calibration/first forward.

    An incoming `_amax` tensor requires registering the buffer before strict loading.
    """
    named = dict(model.named_modules())
    for key, value in state.items():
        if not key.endswith("._amax"):
            continue
        parent, _ = key.rsplit(".", 1)
        module = named.get(parent)
        if module is None or "tensorquantizer" not in type(module).__name__.lower():
            raise RuntimeError(f"Unexpected quantizer amax buffer: {key}")
        if not torch.is_tensor(value):
            raise RuntimeError(f"Invalid amax tensor type: {key}")
        present = getattr(module, "_amax", None)
        if present is None:
            module.register_buffer("_amax", torch.empty_like(value, device=next(model.parameters()).device))
        elif tuple(present.shape) != tuple(value.shape):
            module._amax = torch.empty_like(value, device=present.device)


def strict_load(model: nn.Module, path: Path, *, expected_family: str | None = None,
                expected_base_sha256: str | None = None) -> dict[str, Any]:
    # Input is a trusted locally produced pipeline artifact; never load untrusted .pt files.
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(payload, dict) and "state_dict" in payload:
        if payload.get("schema_version") != STAGE_SCHEMA:
            raise RuntimeError("Unsupported stage checkpoint schema")
        if expected_family is not None and payload.get("family") != expected_family:
            raise RuntimeError("Checkpoint family mismatch")
        if expected_base_sha256 is not None and payload.get("base_sha256") != expected_base_sha256:
            raise RuntimeError("Checkpoint base checkpoint SHA256 mismatch")
    state = tensor_state(payload)
    materialize_amax_buffers(model, state)
    model.load_state_dict(state, strict=True)
    return payload if isinstance(payload, dict) else {}


def cpu_state(model: nn.Module) -> dict[str, Any]:
    # Preserve optional nn.Module extra_state, which is not always a Tensor.
    return {key: (v.detach().cpu().clone() if torch.is_tensor(v) else copy.deepcopy(v))
            for key, v in model.state_dict().items()}


def write_state(path: Path, model: nn.Module, *, family: str, base_sha256: str,
                calib_sha256: str | None, calibrated: bool, **extras: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    info = {"schema_version": STAGE_SCHEMA, "family": family,
            "base_sha256": base_sha256, "calib_sha256": calib_sha256,
            "calibrated": calibrated, "state_dict": cpu_state(model), **extras}
    temp = path.with_name(path.name + ".tmp")
    torch.save(info, temp)
    temp.replace(path)


def restore_qat_from_artifact(spec: dict[str, Any], *, device: torch.device,
                              checkpoint: Path, seed: int = 42) -> tuple[Any, nn.Module, dict[str, Any], dict[str, Any]]:
    """Always reconstruct QAT graph identically, then strict-load the serialized state."""
    from ultralytics import YOLO
    seed_all(seed)
    wrapper = YOLO(str(spec["model"]), task="detect")
    model, audit, head = prepare_qat(
        wrapper.model, family=spec["family"], imgsz=spec["imgsz_hw"], device=device,
        allow_custom_wrappers=bool(spec["meta"].get("custom_wrapper_opt_in", False)),
        sanity_forward=False,
    )
    if head != spec["meta"]["head"]:
        raise RuntimeError("Detection head differs from Step 02 metadata")
    if audit.stats != spec["meta"]["counts"]:
        raise RuntimeError("QAT quantizer topology/count differs from Step 02 metadata")
    meta = strict_load(model, checkpoint, expected_family=spec["family"] if checkpoint != spec["state"] else None,
                       expected_base_sha256=file_sha256(spec["model"]) if checkpoint != spec["state"] else None)
    model.to(device).eval()
    wrapper.model = model
    return wrapper, model, meta, head


def activation_quantizers(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Select *enabled* input/output TensorQuantizers; never weight or disabled BN/Conv outputs."""
    found = []
    for name, module in model.named_modules():
        if "tensorquantizer" not in type(module).__name__.lower():
            continue
        if name.endswith("weight_quantizer") or not name.endswith(("input_quantizer", "output_quantizer")):
            continue
        if q_enabled(module) is True:
            found.append((name, module))
    if not found:
        raise RuntimeError("No enabled activation quantizers found")
    return found


def validate_amax(quantizers: list[tuple[str, nn.Module]]) -> list[dict[str, Any]]:
    results = []
    for name, q in quantizers:
        amax = getattr(q, "_amax", None)
        if not torch.is_tensor(amax) or not amax.numel() or not bool(torch.isfinite(amax).all()) or not bool((amax > 0).all()):
            raise RuntimeError(f"Activation amax missing/non-finite/non-positive: {name}")
        results.append({"name": name, "amax_shape": list(amax.shape),
                        "amax_min": float(amax.min().item()), "amax_max": float(amax.max().item())})
    return results


def check_calibration_meta(payload: dict[str, Any], spec: dict[str, Any], checkpoint: Path) -> None:
    if payload.get("calibrated") is not True or payload.get("family") != spec["family"]:
        raise RuntimeError("Expected calibrated checkpoint for the selected model")
    if tuple(payload.get("imgsz_hw", [])) != spec["imgsz_hw"]:
        raise RuntimeError("Checkpoint H/W differs from preparation")
    if payload.get("base_sha256") != file_sha256(spec["model"]):
        raise RuntimeError("Base model changed since calibration")
    if not payload.get("calib_sha256"):
        raise RuntimeError("Calibration artifact digest missing")


def validate_dataset_classes(path: Path, expected_nc: int) -> dict[str, Any]:
    """Reject class mismatches; a changed head needs training/re-preparation first."""
    import yaml
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid dataset YAML: {path}")
    nc = data.get("nc")
    names = data.get("names")
    if nc is not None:
        nc = int(nc)
    if names is not None:
        if not isinstance(names, (dict, list)):
            raise ValueError("dataset.yaml 'names' must be a list or a mapping")
        named_nc = len(names)
        if nc is not None and nc != named_nc:
            raise ValueError(f"Dataset 'nc'={nc} differs from len(names)={named_nc}")
        nc = named_nc
    if nc is None:
        raise ValueError("dataset.yaml must declare nc or names for safe class-head validation")
    if nc != expected_nc:
        raise RuntimeError(f"Dataset contains {nc} classes, but pretrained QAT head has {expected_nc}. "
                           "Adapt the head and prepare a new QAT graph before training/evaluation")
    return {"dataset_num_classes": nc, "dataset_yaml_sha256": file_sha256(path)}


def extract_metrics(result: Any) -> dict[str, float]:
    box = result.box
    return {"map50": float(box.map50), "map50_95": float(box.map),
            "precision": float(box.mp), "recall": float(box.mr)}


def metric_delta(fp32: dict[str, float], fake_q: dict[str, float]) -> dict[str, Any]:
    delta = {k: fake_q[k] - value for k, value in fp32.items()}
    return {"delta": delta,
            "relative_percent": {k: (v / fp32[k] * 100 if abs(fp32[k]) > 1e-12 else None)
                                 for k, v in delta.items()}}


def fail_closed(stage: str, output: Path, run: Any) -> int:
    logger = logger_for(output, stage)
    payload = starting_payload(stage)
    try:
        run(logger, payload)
    except Exception as exc:
        logger.exception("%s failed", stage)
        payload["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        write_json(output / f"{stage}.json", payload)
        logger.info("%s: %s", stage, payload["status"])
    return 0 if payload["status"].endswith("_PASS") else 1
