#!/usr/bin/env python3
"""Independent PTQ FP32 -> two-output raw-head ONNX exporter.

Never imports/changes QAT code, never inserts fake-Q/DQ or updates weights.
YOLOv8 and YOLO11 use DFL reg logits; YOLO26 default reg_max=1 and
one-to-many branch. Export verified against the live detection head.
"""
from __future__ import annotations

import json
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ptq_dataset_common import save_json, sha256
from ptq_raw_head_postprocess import raw_decode, anchors_and_strides


def find_head(model: nn.Module) -> nn.Module:
    candidates = [m for m in model.modules()
                  if hasattr(m, 'cv2') and hasattr(m, 'cv3') and
                  hasattr(m, 'nc') and hasattr(m, 'reg_max') and hasattr(m, 'stride')]
    if len(candidates) != 1:
        raise RuntimeError(f'Expected exactly one Ultralytics Detect head, got {len(candidates)}')
    return candidates[0]


def detect_family(model: nn.Module, head: nn.Module, model_path: Path, requested: str) -> str:
    if requested != 'auto':
        return requested
    label = (model_path.stem + ' ' + str(getattr(model, 'yaml', {}).get('yaml_file', ''))).lower()
    if 'yolo26' in label:
        return 'yolo26'
    if 'yolov8' in label or 'yolo8' in label:
        return 'yolov8'
    if 'yolo11' in label:
        return 'yolo11'
    if int(head.reg_max) == 1 and hasattr(head, 'one2one_cv2'):
        return 'yolo26'
    mods = {type(m).__name__ for m in model.modules()}
    if 'C3k2' in mods:
        return 'yolo11'
    if 'C2f' in mods and 'C3k2' not in mods:
        return 'yolov8'
    raise RuntimeError('Cannot establish YOLO model family from architecture; use --family explicitly')


def select_layers(head, branch):
    prefix = 'one2one_' if branch == 'one2one' else ''
    box = getattr(head, prefix + 'cv2', None)
    cls = getattr(head, prefix + 'cv3', None)
    if box is None or cls is None or len(box) == 0 or len(box) != len(cls):
        raise RuntimeError(f'Head missing complete {branch} regression/classification layers')
    return box, cls


def capture_reference(model, head, branch, sample):
    box, cls = select_layers(head, branch)
    reg, classes = [None] * len(box), [None] * len(cls)
    shapes = []
    hooks = []

    def pre_hook(_m, inputs):
        feats = inputs[0]
        if not isinstance(feats, (tuple, list)):
            raise RuntimeError('Detection head input is not feature-list')
        shapes[:] = [[int(f.shape[-2]), int(f.shape[-1])] for f in feats]

    def hook_fn(dst, index):
        def capture(_m, _in, output):
            if not torch.is_tensor(output):
                raise RuntimeError('Non-tensor detection output')
            dst[index] = output.detach().clone()
        return capture

    hooks.append(head.register_forward_pre_hook(pre_hook))
    for i, layer in enumerate(box):
        hooks.append(layer.register_forward_hook(hook_fn(reg, i)))
    for i, layer in enumerate(cls):
        hooks.append(layer.register_forward_hook(hook_fn(classes, i)))
    try:
        with torch.no_grad():
            original = model(sample)
    finally:
        for handle in hooks:
            handle.remove()
    if not shapes or any(t is None for t in reg + classes):
        raise RuntimeError(f'Original Detect.forward did not execute {branch} branch')
    reg_ref = torch.cat([t.flatten(2) for t in reg], dim=2).cpu().numpy()
    cls_ref = torch.cat([t.flatten(2) for t in classes], dim=2).cpu().numpy()
    if isinstance(original, (tuple, list)):
        original = original[0]
    original_np = original.detach().cpu().numpy() if torch.is_tensor(original) else None
    return reg_ref, cls_ref, shapes, original_np


def patch_head(head, branch):
    box_layers, cls_layers = select_layers(head, branch)
    reg_ch, nc = 4 * int(head.reg_max), int(head.nc)

    def raw_forward(self, features):
        if not isinstance(features, (tuple, list)) or len(features) != len(box_layers):
            raise RuntimeError('Unexpected raw-head feature inputs')
        regs, clss = [], []
        for i, x in enumerate(features):
            b = x.shape[0]
            regs.append(box_layers[i](x).reshape(b, reg_ch, -1))
            clss.append(cls_layers[i](x).reshape(b, nc, -1))
        return torch.cat(regs, dim=2), torch.cat(clss, dim=2)

    old = head.forward
    head.forward = types.MethodType(raw_forward, head)
    return old


def compare(a, b):
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise RuntimeError(f'Invalid numerical output comparison: {a.shape} vs {b.shape}')
    d = np.abs(a.astype(np.float32) - b.astype(np.float32))
    return dict(mae=float(d.mean()), max_abs=float(d.max()))


def export_ptq(*, model_path: Path, output_dir: Path, family: str = 'auto',
               head_branch: str = 'one2many', imgsz: tuple[int, int] = (640, 640),
               device: str = 'cpu', seed: int = 42, opset: int = 13,
               max_abs_tol: float = 3e-3):
    from ultralytics import YOLO
    import onnx
    import onnxruntime as ort

    torch.manual_seed(seed)
    model_path = Path(model_path).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    output_dir = Path(output_dir).expanduser().resolve()
    qat_markers = ['05_qat_trained_state_dict.pt', '07_convert_raw_litert_int8.json',
                   '08_audit_raw_litert.json', '02a_prepare_qat_yolov8_quant_audit.json',
                   '02b_prepare_qat_yolo11_quant_audit.json',
                   '02c_prepare_qat_yolo26_quant_audit.json']
    existing_meta = output_dir / '06_raw_head_metadata.json'
    if ((any((output_dir / m).exists() for m in qat_markers) or 'qat' in output_dir.parts)
            or (existing_meta.exists() and json.loads(existing_meta.read_text()).get('quantization_method') != 'ptq')):
        raise RuntimeError('Refusing to export PTQ into a QAT run; use a new runs/ptq/... directory')
    if min(imgsz) <= 0 or any(x % 32 for x in imgsz) or max_abs_tol <= 0:
        raise ValueError('imgsz must be positive multiples of 32; max_abs_tol must be positive')
    loaded = YOLO(str(model_path))
    model = loaded.model
    model.float().eval().to(device)
    head = find_head(model)
    family = detect_family(model, head, model_path, family)
    if family not in {'yolov8', 'yolo11', 'yolo26'}:
        raise RuntimeError(f'Unsupported family {family}')
    if family in ('yolov8', 'yolo11') and head_branch != 'one2many':
        raise RuntimeError(f'{family} supports one2many only')
    if family == 'yolo26' and int(head.reg_max) != 1:
        raise RuntimeError('YOLO26 reg_max differs from expected 1; explicit adapter review required')
    if family in ('yolov8', 'yolo11') and int(head.reg_max) <= 1:
        raise RuntimeError('Expected DFL with reg_max>1 for YOLOv8/YOLO11; inspect custom head')
    # An end2end flag can be disabled in a checkpoint even though both heads exist.
    # Restore the execution mode only for capturing one-to-one if requested.
    old_end2end = getattr(head, 'end2end', False)
    if family == 'yolo26' and head_branch == 'one2one' and not old_end2end:
        select_layers(head, 'one2one')
        head.end2end = True
    h, w = map(int, imgsz)
    sample = torch.rand((1, 3, h, w), device=device, dtype=torch.float32)
    try:
        reg_ref, cls_ref, shapes, decoded_ref = capture_reference(model, head, head_branch, sample)
        strides = [float(x) for x in head.stride]
        if len(strides) != len(shapes) or any(x <= 0 for x in strides):
            raise RuntimeError(f'Invalid head feature strides: {strides}/{shapes}')
        positions = sum(y * x for y, x in shapes)
        if reg_ref.shape != (1, 4 * int(head.reg_max), positions) or cls_ref.shape != (1, int(head.nc), positions):
            raise RuntimeError('Raw reference output contract differs from Detect head metadata')
        old_forward = patch_head(head, head_branch)
        try:
            with torch.no_grad():
                reg_t, cls_t = model(sample)
            reg_pt, cls_pt = reg_t.detach().cpu().numpy(), cls_t.detach().cpu().numpy()
            err_pytorch = dict(reg=compare(reg_ref, reg_pt), cls=compare(cls_ref, cls_pt))
            if max(v['max_abs'] for v in err_pytorch.values()) > 1e-5:
                raise RuntimeError(f'Patched regression/class outputs differ: {err_pytorch}')
            meta = dict(family=family, branch=head_branch, reg_max=int(head.reg_max), nc=int(head.nc),
                        feature_shapes=shapes, strides=strides)
            boxes, cls_scores = raw_decode(reg_pt, cls_pt, meta)
            if family != 'yolo26':
                if decoded_ref is None or decoded_ref.shape != (1, 4 + int(head.nc), positions):
                    raise RuntimeError(f'Original YOLO decoded output contract differs: {None if decoded_ref is None else decoded_ref.shape}')
                reconstructed = np.concatenate([boxes.T[None], cls_scores.T[None]], axis=1)
                decode_error = compare(decoded_ref, reconstructed)
            else:
                from ultralytics.utils.tal import dist2bbox
                anch, stride = anchors_and_strides(meta)
                with torch.no_grad():
                    reference_boxes = dist2bbox(torch.from_numpy(reg_pt),
                                               torch.from_numpy(anch.T).unsqueeze(0),
                                               xywh=True, dim=1) * torch.from_numpy(stride).reshape(1, 1, -1)
                decode_error = compare(reference_boxes.numpy(), boxes.T[None])
            if decode_error['max_abs'] > max_abs_tol:
                raise RuntimeError(f'Host raw decoder differs from Ultralytics: {decode_error}')
            output_dir.mkdir(parents=True, exist_ok=True)
            target = output_dir / '06_raw_head.onnx'
            torch.onnx.export(model, sample, str(target), input_names=['images'],
                              output_names=['reg_logits', 'cls_logits'], dynamic_axes=None,
                              opset_version=opset, do_constant_folding=True, dynamo=False)
        finally:
            head.forward = old_forward
        graph = onnx.load(str(target)); onnx.checker.check_model(graph)
        sess = ort.InferenceSession(str(target), providers=['CPUExecutionProvider'])
        if len(sess.get_inputs()) != 1 or len(sess.get_outputs()) != 2 or sess.get_inputs()[0].name != 'images':
            raise RuntimeError('ONNX must have one images input and two raw-head outputs')
        reg_ort, cls_ort = sess.run(None, {'images': sample.detach().cpu().numpy()})
        err_onnx = dict(reg=compare(reg_ref, reg_ort), cls=compare(cls_ref, cls_ort))
        if max(v['max_abs'] for v in err_onnx.values()) > max_abs_tol:
            raise RuntimeError(f'ONNX does not reproduce PyTorch raw heads: {err_onnx}')
        return dict(stage='06d_export_raw_ptq', status='EXPORT_PASS',
                    quantization_method='ptq', output_contract='raw_reg_cls_v1',
                    quantization_provenance='Original pretrained FP32 weights; no QAT; Step 07b static activation calibration',
                    onnx_path=str(target), onnx_sha256=sha256(target),
                    model_path=str(model_path), base_sha256=sha256(model_path),
                    family=family, branch=head_branch, reg_max=int(head.reg_max), nc=int(head.nc),
                    imgsz_hw=[h, w], feature_shapes=shapes, strides=strides,
                    output_shapes=[list(reg_ort.shape), list(cls_ort.shape)],
                    pt_raw_error=err_pytorch, onnx_raw_error=err_onnx,
                    host_decode_vs_pytorch_error=decode_error,
                    host_decode_equivalence_verified=True)
    finally:
        head.end2end = old_end2end
