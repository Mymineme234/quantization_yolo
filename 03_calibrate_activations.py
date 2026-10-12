#!/usr/bin/env python3
"""Step 03: reload the SAME Step-02 QAT graph and calibrate activation quantizers.

Calibration uses real training images only. It never exports ONNX or starts QAT.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import cv2
import numpy as np
import torch

from qat_stage_common import (
    activation_quantizers, fail_closed, file_sha256,
    load_prepared_spec, parse_hw, restore_qat_from_artifact, seed_all,
    torch_device, validate_amax, write_state,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    p = argparse.ArgumentParser(description="Step 03: calibrate all active QAT activation quantizers")
    p.add_argument("--prepared-dir", type=Path, required=True, help="Run directory created by Step 02a/b/c")
    p.add_argument("--calib-images", type=Path, help="Directory of TRAIN images, recursively scanned")
    p.add_argument("--calib-manifest", type=Path, help="Text file with TRAIN image paths")
    p.add_argument("--dataset-root", type=Path, default=Path("."), help="Base for relative manifest image paths")
    p.add_argument("--calib-samples", type=int, default=64)
    p.add_argument("--imgsz", nargs="+", type=int, help="Optional H W consistency check against Step 02")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, help="Output dir; defaults to --prepared-dir")
    p.add_argument("--no-sanity-forward", action="store_true")
    return p.parse_args()


def collect_images(root: Path | None, manifest: Path | None, dataset_root: Path,
                   samples: int, seed: int) -> list[Path]:
    if (root is None) == (manifest is None):
        raise ValueError("Specify exactly one of --calib-images or --calib-manifest")
    if samples <= 0:
        raise ValueError("--calib-samples must be > 0")
    if root:
        if not root.is_dir():
            raise NotADirectoryError(root)
        paths = sorted(p.resolve() for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS and p.is_file())
    else:
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
        paths = []
        for line in manifest.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            candidate = Path(line)
            if not candidate.is_absolute():
                candidate = dataset_root / candidate
            paths.append(candidate.resolve())
    if len(paths) != len(set(paths)):
        raise RuntimeError("Duplicate calibration image paths found (reject ambiguous sample count)")
    if len(paths) < samples:
        raise RuntimeError(f"Not enough calibration images: {len(paths)} < {samples}")
    if not all(p.is_file() for p in paths):
        bad = [str(p) for p in paths if not p.is_file()]
        raise FileNotFoundError(f"Missing calibration images: {bad[:5]}")
    rng = np.random.default_rng(seed)
    selected = [paths[int(i)] for i in rng.choice(len(paths), size=samples, replace=False)]
    return selected


def letterbox(image: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    H, W = hw
    h, w = image.shape[:2]
    r = min(H / h, W / w)
    nh, nw = round(h * r), round(w * r)
    if (nw, nh) != (w, h):
        image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dh, dw = H - nh, W - nw
    t, b = round(dh / 2 - 0.1), round(dh / 2 + 0.1)
    l, rr = round(dw / 2 - 0.1), round(dw / 2 + 0.1)
    return cv2.copyMakeBorder(image, t, b, l, rr, cv2.BORDER_CONSTANT, value=(114, 114, 114))


def to_tensor(image: Path, hw: tuple[int, int], device: torch.device) -> torch.Tensor:
    frame = cv2.imread(str(image))
    if frame is None:
        raise RuntimeError(f"Cannot decode calibration image: {image}")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = letterbox(frame, hw)
    frame = np.ascontiguousarray(frame.transpose(2, 0, 1)[None].astype(np.float32) / 255.0)
    return torch.from_numpy(frame).to(device)


def _drop_stale_amax(q: torch.nn.Module, name: str) -> bool:
    """Remove an existing ModelOpt amax buffer, including a registered None.

    ModelOpt's `amax` setter checks ``hasattr(q, "_amax")`` and then
    immediately reads ``q._amax.shape``. A buffer registered with value
    None therefore raises AttributeError when `load_calib_amax()` runs.
    Deleting the attribute (not assigning None) forces ModelOpt to create a
    new buffer with the calibrated shape. This is safe here ONLY for
    activation quantizers being recalibrated, never weight quantizers.
    """
    had_amax = hasattr(q, "_amax") or "_amax" in getattr(q, "_buffers", {})
    if had_amax:
        # nn.Module.__delattr__ removes registered buffers, including None.
        delattr(q, "_amax")
    if hasattr(q, "_amax") or "_amax" in getattr(q, "_buffers", {}):
        raise RuntimeError(f"{name}: stale _amax buffer could not be removed")
    return had_amax


def calibrate(model: torch.nn.Module, quantizers: list, paths: list[Path],
              hw: tuple[int, int], device: torch.device, logger) -> list[dict]:
    """Collect activation statistics and install fresh ModelOpt amax buffers.

    Re-check `_amax` immediately before `load_calib_amax`, not just before
    calibration: ModelOpt versions may register an `_amax=None` buffer during
    bootstrap/forward, or receive one from a restored prepared checkpoint.
    """
    for name, q in quantizers:
        for api in ("disable_quant", "enable_calib", "disable_calib", "enable_quant", "load_calib_amax"):
            if not callable(getattr(q, api, None)):
                raise RuntimeError(f"{name}: ModelOpt API missing {api}")
        calibrator = getattr(q, "_calibrator", None)
        if calibrator is None or not callable(getattr(calibrator, "reset", None)):
            raise RuntimeError(f"{name}: missing resettable ModelOpt calibrator")

    model.eval()
    for name, q in quantizers:
        q._calibrator.reset()
        q.disable_quant()
        q.enable_calib()

    try:
        with torch.inference_mode():
            for index, path in enumerate(paths, 1):
                _ = model(to_tensor(path, hw, device))
                if index == 1 or index % 10 == 0 or index == len(paths):
                    logger.info("Calibrated %d/%d | %s", index, len(paths), path.name)
    finally:
        for _, q in quantizers:
            q.disable_calib()

    cleared = 0
    for name, q in quantizers:
        # This MUST happen after collecting stats and immediately before
        # ModelOpt's load_calib_amax(). Do not reset the calibrator here.
        cleared += int(_drop_stale_amax(q, name))
        try:
            q.load_calib_amax()
        except Exception as exc:
            raise RuntimeError(f"{name}: load_calib_amax failed: {exc}") from exc
        amax = getattr(q, "_amax", None)
        if not torch.is_tensor(amax):
            raise RuntimeError(f"{name}: load_calib_amax did not register a tensor _amax")
        q.enable_quant()

    logger.info("Installed fresh calibration amax for %d quantizers; cleared %d stale buffers",
                len(quantizers), cleared)
    return validate_amax(quantizers)


def main() -> int:
    args = parse_args()
    output = args.output or args.prepared_dir

    def run(logger, result):
        spec = load_prepared_spec(args.prepared_dir, imgsz_override=parse_hw(args.imgsz) if args.imgsz else None)
        seed_all(args.seed)
        device = torch_device(args.device)
        paths = collect_images(args.calib_images, args.calib_manifest, args.dataset_root,
                               args.calib_samples, args.seed)
        result.update({"family": spec["family"], "imgsz_hw": list(spec["imgsz_hw"]),
                       "device": str(device), "seed": args.seed,
                       "calibration_images": len(paths), "prepared_state": str(spec["state"])})
        logger.info("Recreating Step 02 QAT model from %s", spec["model"])
        _, model, _, head = restore_qat_from_artifact(spec, device=device, checkpoint=spec["state"], seed=args.seed)
        quantizers = activation_quantizers(model)
        logger.info("Detected %d enabled activation quantizers", len(quantizers))
        statistics = calibrate(model, quantizers, paths, spec["imgsz_hw"], device, logger)
        if not args.no_sanity_forward:
            with torch.no_grad():
                outs = model(to_tensor(paths[0], spec["imgsz_hw"], device))
            from qat_prepare_common import collect_tensors
            tensors = collect_tensors(outs)
            if not tensors or any(not bool(torch.isfinite(t).all()) for t in tensors):
                raise RuntimeError("Post-calibration forward not finite")
        base_digest = file_sha256(spec["model"])
        # Selection + content of sample manifests is included in the fingerprint.
        calib_payload = '\n'.join(f"{p}|{file_sha256(p)}" for p in paths).encode()
        calib_digest = hashlib.sha256(calib_payload).hexdigest()
        path = output / "03_calibrated_state_dict.pt"
        write_state(path, model, family=spec["family"], base_sha256=base_digest,
                    calib_sha256=calib_digest, calibrated=True,
                    imgsz_hw=list(spec["imgsz_hw"]), seed=args.seed,
                    calib_samples=len(paths), head=head)
        (output / "03_calibration_images.txt").write_text('\n'.join(map(str, paths)) + '\n', encoding='utf-8')
        result.update({"status": "CALIBRATION_PASS", "calibrated_checkpoint": str(path),
                       "base_sha256": base_digest, "calib_sha256": calib_digest,
                       "activation_quantizers": len(quantizers), "amax": statistics,
                       "warning": "Fake-Q calibration PASS does not prove Q/DQ export or LiteRT native INT8"})
        logger.info("Saved calibrated checkpoint: %s", path)

    return fail_closed("03_calibrate_activations", output, run)


if __name__ == "__main__":
    raise SystemExit(main())
