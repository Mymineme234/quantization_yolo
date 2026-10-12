#!/usr/bin/env python3
"""Shared quantizer placement + reproducible QAT preparation, not a standalone stage.

Import this helper from the numbered 02a/02b/02c entrypoints. Structural
preparation only; real-image calibration, QAT training, ONNX/TFLite conversion,
and downstream accuracy validation remain separate steps.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class PostActQuant(nn.Module):
    """Match activation placement: Conv -> BN -> activation -> fake quant."""

    def __init__(self, activation: nn.Module, quantizer: nn.Module):
        super().__init__()
        self.act = activation
        self.output_quantizer = quantizer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_quantizer(self.act(x))


def q_enabled(q: nn.Module | None) -> bool | None:
    if q is None:
        return None
    if hasattr(q, "is_enabled"):
        value = getattr(q, "is_enabled")
        return bool(value() if callable(value) else value)
    if hasattr(q, "_disabled"):
        return not bool(q._disabled)
    return None


def set_quant_enabled(q: nn.Module | None, enabled: bool, description: str) -> None:
    if q is None:
        raise RuntimeError(f"{description}: missing quantizer")
    fn = getattr(q, "enable" if enabled else "disable", None)
    if callable(fn):
        fn()
    elif hasattr(q, "_disabled"):
        q._disabled = not enabled
    else:
        raise RuntimeError(f"{description}: unsupported quantizer API: {type(q)}")
    if q_enabled(q) is not enabled:
        raise RuntimeError(f"{description}: failed to set enabled={enabled}")


def is_conv2d(module: nn.Module) -> bool:
    return isinstance(module, nn.Conv2d) or (
        "conv2d" in type(module).__name__.lower()
        and isinstance(getattr(module, "weight", None), torch.Tensor)
        and hasattr(module, "input_quantizer")
    )


def has_conv_act(module: nn.Module) -> bool:
    return (
        isinstance(getattr(module, "conv", None), nn.Module)
        and is_conv2d(module.conv)
        and isinstance(getattr(module, "act", None), nn.Module)
        and not isinstance(module.act, PostActQuant)
    )


def trusted_wrapper(module: nn.Module) -> bool:
    # Source-verified Ultralytics Conv/DWConv forward is conv->bn->act.
    # Other wrappers, even with these attributes, need explicit inspection.
    return (
        type(module).__name__ in {"Conv", "DWConv"}
        and type(module).__module__.startswith("ultralytics.nn.modules")
    )


@dataclass
class QuantAudit:
    stats: dict[str, int]
    wrapped: list[dict[str, Any]]
    bare: list[dict[str, Any]]
    unhandled: list[str]


def apply_post_activation_policy(
    model: nn.Module, *, allow_custom_wrappers: bool = False
) -> QuantAudit:
    """Patch ModelOpt Conv2d quantizers; reject unknown conv-act wrapper layouts.

    All counted Conv2d have input/weight quantizers from ModelOpt. Plain
    (unwrapped) Conv2d outputs are quantized directly; wrapped Conv+BN+Act
    quantization is moved to after Act. Conv2 / RepConv are deliberately NOT
    auto-classified as a simple wrapper (they contain parallel branches).
    """
    all_convs = [(n, m) for n, m in model.named_modules() if is_conv2d(m)]
    if not all_convs:
        raise RuntimeError("No Conv2d modules found after ModelOpt insertion")
    for name, conv in all_convs:
        for attr in ("input_quantizer", "weight_quantizer", "output_quantizer"):
            if getattr(conv, attr, None) is None:
                raise RuntimeError(f"{name}: missing {attr}; unsupported ModelOpt conversion")
        if q_enabled(conv.input_quantizer) is not True or q_enabled(conv.weight_quantizer) is not True:
            raise RuntimeError(f"{name}: input and weight quantizers must be enabled")

    wrappers: list[tuple[str, nn.Module]] = []
    unhandled: list[str] = []
    for name, module in model.named_modules():
        # Explicitly reject a known parallel-branch wrapper: treating it as
        # Conv->BN->Act would place the quantizer at an incorrect boundary.
        cls = type(module).__name__
        if cls in {"Conv2", "RepConv"} and type(module).__module__.startswith("ultralytics.nn.modules"):
            unhandled.append(f"{name}: {cls} uses a multi-branch forward; dedicated quantizer policy required")
            continue
        if not has_conv_act(module):
            continue
        if not trusted_wrapper(module) and not allow_custom_wrappers:
            unhandled.append(f"{name}: unverified {type(module).__module__}.{cls}")
            continue
        bn = getattr(module, "bn", None)
        if bn is not None and not isinstance(bn, nn.Module):
            unhandled.append(f"{name}: bn is not nn.Module")
            continue
        if bn is not None and getattr(bn, "output_quantizer", None) is None and not isinstance(bn, nn.Identity):
            unhandled.append(f"{name}: BN output quantizer unavailable")
            continue
        wrappers.append((name, module))
    if unhandled:
        raise RuntimeError(
            "Unsupported conv/activation wrappers (no patch applied). "
            "Review forward semantics and implement a handler for multi-branch blocks; "
            "--allow-custom-wrappers is only for verified conv->bn->act.\n"
            + "\n".join(unhandled[:40])
        )

    owned: set[int] = set()
    for name, module in wrappers:
        if id(module.conv) in owned:
            raise RuntimeError(f"Conv owned by more than one wrapper: {name}")
        owned.add(id(module.conv))
    bare = [(n, m) for n, m in all_convs if id(m) not in owned]
    wrapped_report: list[dict[str, Any]] = []
    for name, module in wrappers:
        original_act = module.act
        output_q = module.conv.output_quantizer
        post_q = copy.deepcopy(output_q)
        set_quant_enabled(output_q, False, f"{name}.conv.output_quantizer")
        bn = getattr(module, "bn", None)
        bn_q = getattr(bn, "output_quantizer", None)
        if bn_q is not None:
            set_quant_enabled(bn_q, False, f"{name}.bn.output_quantizer")
        set_quant_enabled(post_q, True, f"{name}.post_activation")
        module.act = PostActQuant(original_act, post_q)
        wrapped_report.append({
            "name": name, "class": type(module).__name__,
            "activation": type(original_act).__name__,
            "conv_output_quant_enabled": q_enabled(output_q),
            "bn_output_quant_enabled": q_enabled(bn_q),
            "post_activation_quant_enabled": q_enabled(module.act.output_quantizer),
        })

    bare_report: list[dict[str, Any]] = []
    for name, module in bare:
        set_quant_enabled(module.output_quantizer, True, f"{name}.output_quantizer")
        bare_report.append({
            "name": name, "class": type(module).__name__,
            "output_quant_enabled": q_enabled(module.output_quantizer),
        })

    if len(wrapped_report) + len(bare_report) != len(all_convs):
        raise RuntimeError("Conv coverage mismatch")
    if any(row["post_activation_quant_enabled"] is not True or row["conv_output_quant_enabled"] is not False
           or row["bn_output_quant_enabled"] is True for row in wrapped_report):
        raise RuntimeError("Incorrect quantizer state in wrapped Conv")
    if any(row["output_quant_enabled"] is not True for row in bare_report):
        raise RuntimeError("Bare Conv output quantizer disabled")

    stats = {
        "all_convs": len(all_convs),
        "post_activation_wrappers": len(wrapped_report),
        "bare_convs": len(bare_report),
        "enabled_post_activation_outputs": sum(r["post_activation_quant_enabled"] is True for r in wrapped_report),
        "enabled_bare_conv_outputs": sum(r["output_quant_enabled"] is True for r in bare_report),
        "one2one_branch_convs": sum("one2one_cv" in name for name, _ in all_convs),
        "one2many_branch_convs": sum(
            (".cv2." in name or ".cv3." in name) and "one2one" not in name for name, _ in all_convs
        ),
        "linear_modules": sum(isinstance(m, nn.Linear) for m in model.modules()),
        "conv_transpose_modules": sum(isinstance(m, nn.ConvTranspose2d) for m in model.modules()),
    }
    return QuantAudit(stats, wrapped_report, bare_report, [])


def find_detect_head(model: nn.Module) -> tuple[str, nn.Module]:
    candidates = [(n, m) for n, m in model.named_modules()
                  if hasattr(m, "nc") and hasattr(m, "reg_max")
                  and hasattr(m, "cv2") and hasattr(m, "cv3")]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected exactly one detection head with nc/reg_max/cv2/cv3; found {len(candidates)}")
    return candidates[0]


def ensure_yolo26_dual_head(model: nn.Module) -> dict[str, Any]:
    """Select dual-head *training* mode without inventing missing head weights.

    Ultralytics may load a YOLO26 checkpoint with end2end=False to select
    one-to-many/NMS inference while retaining both trained head branches.
    QAT training must use the actual one-to-one and one-to-many branches.
    This normalization is deterministic and repeated on every stage restore.
    """
    name, head = find_detect_head(model)
    one2one_box = isinstance(getattr(head, "one2one_cv2", None), nn.Module)
    one2one_cls = isinstance(getattr(head, "one2one_cv3", None), nn.Module)
    one2many_box = isinstance(getattr(head, "cv2", None), nn.Module)
    one2many_cls = isinstance(getattr(head, "cv3", None), nn.Module)
    details = {
        "head": name,
        "head_class": type(head).__name__,
        "reg_max": int(head.reg_max),
        "end2end_before": bool(getattr(head, "end2end", False)),
        "one2one_cv2": one2one_box,
        "one2one_cv3": one2one_cls,
        "cv2": one2many_box,
        "cv3": one2many_cls,
    }
    if details["reg_max"] != 1:
        raise RuntimeError(f"YOLO26 needs reg_max=1, observed head: {details}")
    if not all((one2one_box, one2one_cls, one2many_box, one2many_cls)):
        raise RuntimeError(
            "YOLO26 checkpoint lacks a complete dual detection head; cannot safely "
            "synthesize one-to-one weights. Inspect installed Ultralytics version "
            f"and checkpoint origin. Observed: {details}"
        )
    if not details["end2end_before"]:
        try:
            head.end2end = True
        except Exception as exc:
            raise RuntimeError(
                f"YOLO26 head has both branches but cannot enable dual-head mode: {details}"
            ) from exc
    if not bool(getattr(head, "end2end", False)):
        raise RuntimeError(f"YOLO26 dual-head mode is still inactive: {details}")
    details["end2end_after"] = True
    details["mode_changed"] = not details["end2end_before"]
    return details


def validate_detection_head(model: nn.Module, family: str) -> dict[str, Any]:
    """Inspect actual head rather than assuming weights filename == family."""
    name, head = find_detect_head(model)
    reg_max = int(head.reg_max)
    end2end = bool(getattr(head, "end2end", False))
    one2one_box = getattr(head, "one2one_cv2", None) is not None
    one2one_cls = getattr(head, "one2one_cv3", None) is not None
    if family in ("yolov8", "yolo11"):
        if reg_max <= 1:
            raise RuntimeError(f"{family}: expected DFL regression reg_max>1, observed {reg_max}")
        if end2end or one2one_box or one2one_cls:
            raise RuntimeError(f"{family}: this entrypoint expects non-end2end Detect; got dual/end2end head")
    elif family == "yolo26":
        if reg_max != 1:
            raise RuntimeError(f"YOLO26 baseline reg_max=1 (DFL-free); got {reg_max}. A custom head needs a separate policy")
        if not (end2end and one2one_box and one2one_cls):
            raise RuntimeError("YOLO26 requires end2end=True and both one2one_cv2/one2one_cv3 branches")
    else:
        raise ValueError(f"Unrecognized family: {family}")
    if int(head.nc) <= 0:
        raise RuntimeError(f"Invalid nc={head.nc}")
    info = {
        "name": name, "class": type(head).__name__, "nc": int(head.nc),
        "reg_max": reg_max, "dfl": reg_max > 1, "end2end": end2end,
        "one2one_box_present": one2one_box, "one2one_class_present": one2one_cls,
        "one2many_box_present": getattr(head, "cv2", None) is not None,
        "one2many_class_present": getattr(head, "cv3", None) is not None,
        "output_contract_hint": (
            "raw_dfl_reg_logits_and_class_logits; downstream export must verify numerical equivalence"
            if reg_max > 1 else
            "raw_direct_reg_and_class_logits; choose one-to-one vs one-to-many during export and verify decoding"
        ),
    }
    return info


def build_modelopt_int8(model: nn.Module, dummy: torch.Tensor) -> nn.Module:
    import modelopt.torch.quantization as mtq
    cfg = copy.deepcopy(mtq.INT8_DEFAULT_CFG)
    if not isinstance(cfg.get("quant_cfg"), list):
        raise RuntimeError("ModelOpt INT8_DEFAULT_CFG.quant_cfg must be a list; pin compatible ModelOpt version")

    def forward_loop(quant_model: nn.Module) -> None:
        quant_model.eval()
        with torch.no_grad():
            _ = quant_model(dummy)
    return mtq.quantize(model, cfg, forward_loop)


def collect_tensors(value: Any) -> list[torch.Tensor]:
    if torch.is_tensor(value):
        return [value]
    if isinstance(value, dict):
        return [t for v in value.values() for t in collect_tensors(v)]
    if isinstance(value, (tuple, list)):
        return [t for v in value for t in collect_tensors(v)]
    return []


def prepare_qat(
    model: nn.Module, *, family: str, imgsz: tuple[int, int], device: torch.device,
    allow_custom_wrappers: bool = False, sanity_forward: bool = True,
) -> tuple[nn.Module, QuantAudit, dict[str, Any]]:
    """Insert + patch quantizers, check head before and after conversion."""
    if family == "yolo26":
        ensure_yolo26_dual_head(model)
    head_before = validate_detection_head(model, family)
    model.to(device).eval()
    h, w = imgsz
    dummy = torch.zeros((1, 3, h, w), dtype=torch.float32, device=device)
    model = build_modelopt_int8(model, dummy)
    audit = apply_post_activation_policy(model, allow_custom_wrappers=allow_custom_wrappers)
    head_after = validate_detection_head(model, family)
    for attr in ("nc", "reg_max", "end2end"):
        if head_after[attr] != head_before[attr]:
            raise RuntimeError(f"Detect head changed after Q insertion: {attr}")
    if family == "yolo26" and audit.stats["one2one_branch_convs"] == 0:
        raise RuntimeError("YOLO26 one-to-one branch has no quantized Conv2d; structural gate failed")
    if sanity_forward:
        with torch.no_grad():
            out = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        tensors = collect_tensors(out)
        if not tensors or not all(bool(torch.isfinite(t).all()) for t in tensors):
            raise RuntimeError("No tensor outputs or NaN/Inf during dummy forward")
    return model, audit, head_after


def build_parser(family: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=f"Structural post-activation QAT preparation for {family} detection; no dataset calibration."
    )
    p.add_argument("--model", type=Path, required=True, help="Ultralytics .pt checkpoint")
    p.add_argument("--output", type=Path, required=True, help="Run directory for log/JSON/prepared state")
    p.add_argument("--imgsz", nargs="+", type=int, default=[640], metavar="SIZE", help="One integer or H W")
    p.add_argument("--device", default="auto", help="auto, cpu, cuda:0, ...")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--allow-custom-wrappers", action="store_true",
                   help="Only when custom block is verified conv->bn->act; parallel conv blocks unsupported")
    p.add_argument("--sanity-forward", action="store_true", help="Run post-patch forward and check finite results")
    p.add_argument("--no-save-state", action="store_true", help="Audit only")
    return p


def setup_logger(path: Path, family: str) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"qat.prepare.{family}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(path, encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def main_for_family(args: argparse.Namespace, family: str, step_name: str) -> int:
    args.output.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(args.output / f"{step_name}.log", family)
    payload: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": str(args.model), "family": family, "status": "FAIL",
        "seed": args.seed, "calibration": "NOT DONE: dummy structural forward only",
        "training": "NOT DONE", "conversion": "NOT DONE", "errors": [],
        "notes": ["INT8 deployment and accuracy are NOT certified by this step"],
    }
    try:
        if len(args.imgsz) not in (1, 2):
            raise ValueError("--imgsz takes 1 integer or 2 integers H W")
        hw = (args.imgsz[0], args.imgsz[0]) if len(args.imgsz) == 1 else tuple(args.imgsz)
        if any(x <= 0 or x % 32 for x in hw):
            raise ValueError(f"Image dimensions must be positive multiples of 32, got {hw}")
        payload["imgsz_hw"] = list(hw)
        if not args.model.is_file():
            raise FileNotFoundError(args.model)

        from ultralytics import YOLO
        import modelopt.torch.quantization as _mtq  # noqa: F401
        device_text = ("cuda:0" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
        device = torch.device(device_text)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA unavailable: {device}")
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        payload["device"] = str(device)
        payload["torch_version"] = torch.__version__
        payload["ultralytics_version"] = metadata.version("ultralytics")
        payload["modelopt_version"] = metadata.version("nvidia-modelopt")
        wrapper = YOLO(str(args.model), task="detect")
        model = wrapper.model
        if getattr(model, "task", "detect") != "detect":
            raise RuntimeError("Only detection task is supported")
        logger.info("Prepare %s | model=%s | imgsz=%s | device=%s", family, args.model, hw, device)
        if family == "yolo26":
            mode_info = ensure_yolo26_dual_head(model)
            payload["yolo26_dual_head_mode"] = mode_info
            logger.info("YOLO26 head mode: %s", mode_info)
            if mode_info["mode_changed"]:
                logger.warning("YOLO26 checkpoint was in one-to-many inference mode; restored dual-head mode for QAT")
        head = validate_detection_head(model, family)
        logger.info("Head: %s", head)
        model, audit, head = prepare_qat(
            model, family=family, imgsz=hw, device=device,
            allow_custom_wrappers=args.allow_custom_wrappers,
            sanity_forward=args.sanity_forward,
        )
        payload["head"] = head
        payload["counts"] = audit.stats
        payload["wrapped_convs"] = audit.wrapped
        payload["bare_convs"] = audit.bare
        payload["custom_wrapper_opt_in"] = args.allow_custom_wrappers
        if not args.no_save_state:
            path = args.output / f"{step_name}_prepared_state_dict.pt"
            torch.save(model.state_dict(), path)
            payload["prepared_state_dict"] = str(path)
            logger.info("Saved prepared state: %s", path)
        payload["status"] = "STRUCTURAL_PASS"
        logger.info("Quantizer placement coverage: %s", audit.stats)
        if family in ("yolo11", "yolo26"):
            logger.warning("Attention/non-Conv operators may need special QAT/export treatment; Conv audit alone does NOT certify full-INT8 LiteRT")
        logger.warning("Structural PASS only: real-image calibration, QAT, ONNX and LiteRT remain required")
    except Exception as exc:
        logger.exception("QAT structural preparation failed")
        payload["errors"].append(f"{type(exc).__name__}: {exc}")
    finally:
        path = args.output / f"{step_name}_quant_audit.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Audit %s | %s", path, payload["status"])
    return 0 if payload["status"] == "STRUCTURAL_PASS" else 1
