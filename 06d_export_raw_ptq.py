#!/usr/bin/env python3
"""Step 06D: independent YOLO FP32 checkpoint -> raw-head ONNX for PTQ."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from ptq_dataset_common import save_json


def main():
    p = argparse.ArgumentParser(description='FP32 YOLOv8/11/26 -> verified raw-head ONNX (PTQ)')
    p.add_argument('--model', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path, help='PTQ experiment directory (e.g. runs/ptq/yolo11n)')
    p.add_argument('--family', choices=('auto', 'yolov8', 'yolo11', 'yolo26'), default='auto')
    p.add_argument('--head-branch', choices=('one2many', 'one2one'), default='one2many')
    p.add_argument('--imgsz', nargs=2, type=int, default=(640, 640), metavar=('H', 'W'))
    p.add_argument('--device', default='cpu')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--opset', type=int, default=13)
    p.add_argument('--max-abs-tol', type=float, default=3e-3)
    args = p.parse_args()
    work = args.output.expanduser().resolve()
    result = dict(stage='06d_export_raw_ptq', status='EXPORT_FAIL')
    try:
        from ptq_export_common import export_ptq
        result = export_ptq(model_path=args.model, output_dir=work, family=args.family,
                            head_branch=args.head_branch, imgsz=tuple(args.imgsz),
                            device=args.device, seed=args.seed, opset=args.opset,
                            max_abs_tol=args.max_abs_tol)
        print('EXPORT_PASS:', result['onnx_path'])
        print('Family:', result['family'], 'Branch:', result['branch'])
        print('Shapes:', result['output_shapes'], 'Host decode error:', result['host_decode_vs_pytorch_error'])
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
        print('EXPORT_FAIL:', result['error'], file=sys.stderr)
    finally:
        # Do not overwrite a successful prior export's metadata when an export fails:
        # delete it to fail closed and prevent downstream accidental reuse.
        work.mkdir(parents=True, exist_ok=True)
        save_json(work / '06d_export_raw_ptq.json', result)
        if result['status'] == 'EXPORT_PASS':
            save_json(work / '06_raw_head_metadata.json', result)
        else:
            (work / '06_raw_head_metadata.json').unlink(missing_ok=True)
    return 0 if result['status'] == 'EXPORT_PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
