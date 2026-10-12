#!/usr/bin/env python3
"""Step 06 shared exporter: trained QAT weights -> raw regression/class ONNX.

Export is a *QAT-trained-weights float ONNX* for onnx2tf re-calibration.
ModelOpt fake quantizers are disabled during export intentionally. This is NOT
scale-preserving Q/DQ lowering. Only locally trusted Step 05 artifacts are loaded.
"""
from __future__ import annotations

import argparse
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from qat_prepare_common import find_detect_head
from qat_stage_common import (check_calibration_meta, fail_closed, file_sha256,
                              load_prepared_spec, restore_qat_from_artifact, torch_device)
from raw_head_postprocess import raw_decode


def patch_raw_head(head: nn.Module, family: str, head_branch: str):
    """Call trained detection head's existing output branches directly.

    Nothing is guessed from ONNX node names. Both outputs are pre-sigmoid,
    pre-DFL expectation and pre-anchor/stride decoding.
    """
    branch = head_branch
    box_layers = getattr(head, 'one2one_cv2' if branch == 'one2one' else 'cv2')
    cls_layers = getattr(head, 'one2one_cv3' if branch == 'one2one' else 'cv3')
    reg_ch = 4 * int(head.reg_max)
    nc = int(head.nc)
    if len(box_layers) != len(cls_layers) or not box_layers:
        raise RuntimeError('Incomplete detection head branch')

    def raw_forward(self, features):
        if not isinstance(features, (list, tuple)) or len(features) != len(box_layers):
            raise RuntimeError('Expected detection features list matching the head levels')
        reg_parts, cls_parts = [], []
        for i, f in enumerate(features):
            b = f.shape[0]
            reg_parts.append(box_layers[i](f).reshape(b, reg_ch, -1))
            cls_parts.append(cls_layers[i](f).reshape(b, nc, -1))
        return torch.cat(reg_parts, dim=2), torch.cat(cls_parts, dim=2)

    old_forward = head.forward
    head.forward = types.MethodType(raw_forward, head)
    return old_forward, branch


def inspect_raw_reference(model: nn.Module, head: nn.Module, family: str, head_branch: str, sample: torch.Tensor):
    """Capture *actually executed* original branch outputs before patching."""
    box_layers = getattr(head, 'one2one_cv2' if head_branch == 'one2one' else 'cv2')
    cls_layers = getattr(head, 'one2one_cv3' if head_branch == 'one2one' else 'cv3')
    b = [None] * len(box_layers)
    c = [None] * len(cls_layers)
    shapes = []
    handles = []

    def pre_hook(_module, inputs):
        features = inputs[0]
        shapes[:] = [[int(f.shape[-2]), int(f.shape[-1])] for f in features]

    def capture(target, i):
        def hook(_module, _inputs, output):
            if not torch.is_tensor(output):
                raise RuntimeError('Non-tensor regression/class branch output')
            target[i] = output.detach().clone()
        return hook

    handles.append(head.register_forward_pre_hook(pre_hook))
    for i, layer in enumerate(box_layers):
        handles.append(layer.register_forward_hook(capture(b, i)))
    for i, layer in enumerate(cls_layers):
        handles.append(layer.register_forward_hook(capture(c, i)))
    try:
        with torch.no_grad():
            original = model(sample)
    finally:
        for h in handles:
            h.remove()
    if any(t is None for t in b + c) or not shapes:
        raise RuntimeError('Baseline Detect forward did not execute all selected head branches. '
                           'Check YOLO26 end2end/one-to-one policy')
    reg = torch.cat([x.flatten(2) for x in b], dim=2).cpu().numpy()
    cls = torch.cat([x.flatten(2) for x in c], dim=2).cpu().numpy()
    if isinstance(original, (tuple, list)):
        original = original[0]
    decoded_reference = original.detach().cpu().numpy() if torch.is_tensor(original) else None
    return reg, cls, shapes, decoded_reference


def max_error(a, b):
    if a.shape != b.shape:
        raise RuntimeError(f'Numerical comparison shape mismatch {a.shape} != {b.shape}')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise RuntimeError('Non-finite numerical comparison')
    difference = np.abs(a.astype(np.float32) - b.astype(np.float32))
    return dict(mae=float(difference.mean()), max_abs=float(difference.max()))


def parse_args(family: str):
    p = argparse.ArgumentParser(description=f'Export verified {family} raw-head ONNX from Step 05')
    p.add_argument('--prepared-dir', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, help='Default: <prepared-dir>/05_qat_trained_state_dict.pt')
    p.add_argument('--output', type=Path, help='Default: <prepared-dir>/06_raw_head.onnx')
    p.add_argument('--device', default='cpu', help='CPU recommended to avoid extra GPU memory')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--opset', type=int, default=13)
    if family == 'yolo26':
        p.add_argument('--head-branch', choices=('one2many', 'one2one'), default='one2many',
                       help='Default NMS-compatible; one2one enables NMS-free top-k')
    p.add_argument('--max-abs-tol', type=float, default=3e-3)
    return p.parse_args()


def main_for_family(family: str) -> int:
    args = parse_args(family)
    output = args.output or args.prepared_dir / '06_raw_head.onnx'

    def run(log, result):
        if args.max_abs_tol <= 0:
            raise ValueError('--max-abs-tol must be positive')
        spec = load_prepared_spec(args.prepared_dir)
        if spec['family'] != family:
            raise RuntimeError(f'Entrypoint {family} cannot export checkpoint family {spec["family"]}')
        checkpoint = args.checkpoint or args.prepared_dir / '05_qat_trained_state_dict.pt'
        if not checkpoint.is_file():
            raise FileNotFoundError(f'QAT trained state unavailable: {checkpoint}')
        device = torch_device(args.device)
        wrapper, model, ckpt, head_meta = restore_qat_from_artifact(
            spec, device=device, checkpoint=checkpoint, seed=args.seed)
        check_calibration_meta(ckpt, spec, checkpoint)
        if 'epoch_completed' not in ckpt or int(ckpt['epoch_completed']) <= 0:
            raise RuntimeError('Checkpoint has no completed QAT training metadata')
        # Float ONNX export; fake quantizers were only a training-time mechanism.
        # Converter will quantize from representative images in Step 07.
        quant_count = 0
        for mod in model.modules():
            if 'tensorquantizer' in type(mod).__name__.lower():
                if not hasattr(mod, 'disable'):
                    raise RuntimeError('ModelOpt quantizer lacks disable()')
                mod.disable()
                quant_count += 1
        if quant_count == 0:
            raise RuntimeError('QAT graph contains no quantizers')
        model.eval().to(device)
        name, head = find_detect_head(model)
        h, w = spec['imgsz_hw']
        torch.manual_seed(args.seed)
        sample = torch.rand((1, 3, h, w), dtype=torch.float32, device=device)
        selected_branch = args.head_branch if family == 'yolo26' else 'one2many'
        reg_ref, cls_ref, shapes, decoded_ref = inspect_raw_reference(model, head, family, selected_branch, sample)
        stride_values = getattr(head, 'stride', None)
        if stride_values is None:
            raise RuntimeError('Detect stride metadata missing')
        strides = [float(s) for s in stride_values]
        if len(strides) != len(shapes) or any(s <= 0 for s in strides):
            raise RuntimeError(f'Invalid head stride metadata: {strides} / {shapes}')
        reg_ch, cls_ch = 4 * int(head.reg_max), int(head.nc)
        n = sum(y * x for y, x in shapes)
        if reg_ref.shape != (1, reg_ch, n) or cls_ref.shape != (1, cls_ch, n):
            raise RuntimeError(f'Raw reference shapes invalid: {reg_ref.shape}, {cls_ref.shape}')
        old_forward, branch = patch_raw_head(head, family, selected_branch)
        try:
            with torch.no_grad():
                raw_out = model(sample)
            if not isinstance(raw_out, (tuple, list)) or len(raw_out) != 2:
                raise RuntimeError('Patched head must output (reg_logits, cls_logits)')
            reg_pt, cls_pt = [t.detach().cpu().numpy() for t in raw_out]
            errors_pt = {'reg': max_error(reg_ref, reg_pt), 'cls': max_error(cls_ref, cls_pt)}
            if max(v['max_abs'] for v in errors_pt.values()) > 1e-5:
                raise RuntimeError(f'Patched head differs from original branch tensors: {errors_pt}')
            host_ref = dict(family=family, reg_max=int(head.reg_max), nc=int(head.nc),
                            feature_shapes=shapes, strides=strides)
            decoded_error = None
            if family in ('yolov8', 'yolo11'):
                if decoded_ref is None or decoded_ref.shape != (1, 4 + cls_ch, n):
                    raise RuntimeError(f'Original decoded output contract mismatch: '
                                       f'{None if decoded_ref is None else decoded_ref.shape}')
                boxes, class_scores = raw_decode(reg_pt, cls_pt, host_ref)
                rebuilt = np.concatenate((boxes.T[None], class_scores.T[None]), axis=1)
                decoded_error = max_error(decoded_ref, rebuilt)
                if decoded_error['max_abs'] > args.max_abs_tol:
                    raise RuntimeError(f'Host DFL/anchor decoding differs from PyTorch: {decoded_error}')
            # YOLO26 canonical forward may choose the other branch and/or top-k.
            # Raw branch tensors are independently checked above. The host
            # boxes will be compared with the canonical Ultralytics dist2bbox.
            if family == 'yolo26':
                from ultralytics.utils.tal import dist2bbox
                from raw_head_postprocess import anchors_and_strides
                anch, stride = anchors_and_strides(host_ref)
                with torch.no_grad():
                    distances = torch.from_numpy(reg_pt)
                    anchors_t = torch.from_numpy(anch.T).unsqueeze(0)
                    boxes_t = dist2bbox(distances, anchors_t, xywh=True, dim=1)
                    boxes_t = boxes_t * torch.from_numpy(stride).reshape(1, 1, -1)
                host_xywh, _ = raw_decode(reg_pt, cls_pt, host_ref)
                decoded_error = max_error(boxes_t.numpy(), host_xywh.T[None])
                if decoded_error['max_abs'] > args.max_abs_tol:
                    raise RuntimeError(f'YOLO26 dist2bbox differs from host decode: {decoded_error}')
            output.parent.mkdir(parents=True, exist_ok=True)
            log.info('Export float raw-head ONNX | family=%s branch=%s | %s', family, branch, output)
            torch.onnx.export(model, sample, str(output), opset_version=args.opset,
                              input_names=['images'], output_names=['reg_logits', 'cls_logits'],
                              dynamic_axes=None, do_constant_folding=True, dynamo=False)
        finally:
            head.forward = old_forward
        import onnx
        import onnxruntime as ort
        graph = onnx.load(str(output))
        onnx.checker.check_model(graph)
        session = ort.InferenceSession(str(output), providers=['CPUExecutionProvider'])
        if len(session.get_inputs()) != 1 or len(session.get_outputs()) != 2:
            raise RuntimeError('Exported ONNX must have 1 input and 2 outputs')
        if session.get_inputs()[0].name != 'images':
            raise RuntimeError('Unexpected ONNX input name')
        reg_ort, cls_ort = session.run(None, {'images': sample.cpu().numpy()})
        errors_onnx = {'reg': max_error(reg_ref, reg_ort), 'cls': max_error(cls_ref, cls_ort)}
        if max(v['max_abs'] for v in errors_onnx.values()) > args.max_abs_tol:
            raise RuntimeError(f'ONNX runtime differs from PyTorch raw head: {errors_onnx}')
        result.update({'status': 'EXPORT_PASS', 'output_contract': 'raw_reg_cls_v1',
                       'family': family, 'branch': branch, 'nc': int(head.nc),
                       'reg_max': int(head.reg_max), 'imgsz_hw': [h, w],
                       'feature_shapes': shapes, 'strides': strides,
                       'quantizers_disabled_during_float_export': quant_count,
                       'output_shapes': [list(reg_ort.shape), list(cls_ort.shape)],
                       'checkpoint': str(checkpoint.resolve()), 'checkpoint_sha256': file_sha256(checkpoint),
                       'base_sha256': file_sha256(spec['model']),
                       'onnx_path': str(output.resolve()), 'onnx_sha256': file_sha256(output),
                       'pt_raw_error': errors_pt, 'onnx_raw_error': errors_onnx,
                       'host_decode_vs_pytorch_error': decoded_error,
                       'host_decode_equivalence_verified': decoded_error is not None,
                       'quantization_provenance': 'QAT trained weights, re-quantized in Step 07 by onnx2tf PTQ'})
        log.info('Verified ONNX output %s | max abs reg=%g cls=%g',
                 result['output_shapes'], errors_onnx['reg']['max_abs'], errors_onnx['cls']['max_abs'])

    # Step 06 metadata at fixed location for downstream stages.
    # fail_closed always names its report stage.json; write the interface alias
    # only on PASS to avoid downstream consumption of stale exports.
    rc = fail_closed('06_raw_head_export', args.prepared_dir, run)
    import json
    report_path = args.prepared_dir / '06_raw_head_export.json'
    meta_path = args.prepared_dir / '06_raw_head_metadata.json'
    if rc == 0:
        meta_path.write_text(report_path.read_text(encoding='utf-8'), encoding='utf-8')
    else:
        meta_path.unlink(missing_ok=True)
    return rc
