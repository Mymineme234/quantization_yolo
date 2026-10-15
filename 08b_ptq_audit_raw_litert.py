#!/usr/bin/env python3
"""Step 08B PTQ full INT8 structural audit of executable LiteRT graph, fail-closed.

Allows integer shape/index metadata and int32 bias/accumulation, but refuses
floating-point ops/tensors or DEQUANTIZE in any path feeding graph outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from ptq_raw_head_postprocess import make_interpreter, quant_params, read_metadata


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(2 ** 20), b''):
            h.update(data)
    return h.hexdigest()


def reachable_ops(ops: list[dict], output_details: list[dict]) -> set[int]:
    producer = {}
    for oi, op in enumerate(ops):
        for t in op['outputs']:
            if int(t) >= 0:
                producer[int(t)] = oi
    seen_ops, pending = set(), [int(t['index']) for t in output_details]
    while pending:
        t = pending.pop()
        oi = producer.get(t)
        if oi is not None and oi not in seen_ops:
            seen_ops.add(oi)
            pending.extend(int(i) for i in ops[oi]['inputs'] if int(i) >= 0)
    return seen_ops


def audit_interpreter(itp, *, expected_outputs: int = 2) -> dict:
    tensors = {int(t['index']): t for t in itp.get_tensor_details()}
    inputs = itp.get_input_details()
    outputs = itp.get_output_details()
    ops = itp._get_ops_details()  # LiteRT private API, intentionally pinned in README.
    violations = []

    def problem(reason, **context):
        violations.append({'reason': reason, **context})

    if len(inputs) != 1 or len(outputs) != expected_outputs:
        problem('wrong_external_tensor_count', input_count=len(inputs), output_count=len(outputs))
    for kind, tensor_list in [('input', inputs), ('output', outputs)]:
        for t in tensor_list:
            if np.dtype(t['dtype']) != np.dtype('int8'):
                problem('non_int8_external', side=kind, tensor=t['name'], dtype=str(t['dtype']))
            try:
                quant_params(t)
            except Exception as exc:
                problem('missing_external_quantization', side=kind, tensor=t['name'], error=str(exc))

    seen = reachable_ops(ops, outputs)
    # Discover executable *data flow* including intermediates. Constants unused
    # after output pruning do not become a false positive.
    live_indices = {int(t['index']) for t in inputs + outputs}
    for i in seen:
        live_indices.update(int(t) for t in ops[i]['inputs'] if int(t) >= 0)
        live_indices.update(int(t) for t in ops[i]['outputs'] if int(t) >= 0)
    accepted = {'int8', 'int32', 'int64', 'bool'}
    for i in sorted(live_indices):
        t = tensors.get(i)
        if t is None:
            problem('missing_tensor_details', index=i)
            continue
        dtype = np.dtype(t['dtype']).name
        if dtype not in accepted:
            problem('noninteger_live_tensor', index=i, dtype=dtype, name=t['name'])
    hist = Counter()
    for i in sorted(seen):
        op = ops[i]
        name = str(op['op_name'])
        hist[name] += 1
        if name in {'DEQUANTIZE', 'CUSTOM', 'Flex', 'FlexDelegate', 'DELEGATE'} or name.startswith('Flex'):
            problem('forbidden_operator', op_index=i, op=name)
        ins = [tensors.get(int(t)) for t in op['inputs'] if int(t) >= 0]
        outs = [tensors.get(int(t)) for t in op['outputs'] if int(t) >= 0]
        if name == 'QUANTIZE':
            if any(t is None or np.dtype(t['dtype']) != np.dtype('int8') for t in ins + outs):
                problem('quantize_float_boundary', op_index=i)
        if name in ('CONV_2D', 'DEPTHWISE_CONV_2D', 'FULLY_CONNECTED'):
            if len(ins) < 2 or len(outs) < 1:
                problem('malformed_linear_op', op_index=i, op=name)
                continue
            if (ins[0] is None or ins[1] is None or outs[0] is None or
                any(np.dtype(t['dtype']) != np.dtype('int8') for t in (ins[0], ins[1], outs[0]))):
                problem('nonint8_linear_dataflow', op_index=i, op=name)
                continue
            scales = np.asarray(ins[1]['quantization_parameters'].get('scales', []))
            if scales.size == 0 or not np.isfinite(scales).all() or (scales <= 0).any():
                problem('invalid_quantized_weights', op_index=i, op=name)
            if len(ins) >= 3 and ins[2] is not None and np.dtype(ins[2]['dtype']) != np.dtype('int32'):
                problem('nonint32_bias', op_index=i, op=name)
        # All *activation* inputs/outputs of core kernels should stay int8.
        if name in ('LOGISTIC', 'MUL', 'ADD', 'SUB', 'MAX_POOL_2D', 'AVERAGE_POOL_2D',
                    'RESIZE_NEAREST_NEIGHBOR', 'RESIZE_BILINEAR', 'SOFTMAX', 'CONCATENATION',
                    'PAD', 'PADV2', 'MAXIMUM', 'MINIMUM'):
            # For PAD/RESIZE, shape/padding inputs may be int32; outputs always int8.
            if outs and outs[0] is not None and np.dtype(outs[0]['dtype']) != np.dtype('int8'):
                problem('nonint8_core_op_output', op_index=i, op=name)
    if hist.get('CONV_2D', 0) + hist.get('DEPTHWISE_CONV_2D', 0) == 0:
        problem('no_convolution_ops')
    return dict(status='STRUCTURAL_PASS' if not violations else 'STRUCTURAL_FAIL',
                violations=violations, input_count=len(inputs), output_count=len(outputs),
                executable_operator_count=len(seen),
                live_tensor_count=len(live_indices),
                executable_operator_histogram=dict(sorted(hist.items())),
                input_details=[{'name': t['name'], 'shape': list(map(int, t['shape'])),
                                'dtype': np.dtype(t['dtype']).name,
                                'quantization': list(map(float, t['quantization']))} for t in inputs],
                output_details=[{'name': t['name'], 'shape': list(map(int, t['shape'])),
                                 'dtype': np.dtype(t['dtype']).name,
                                 'quantization': list(map(float, t['quantization']))} for t in outputs])


def main():
    p = argparse.ArgumentParser(description='Fail-closed LiteRT executable graph INT8 audit')
    p.add_argument('--prepared-dir', required=True, type=Path)
    p.add_argument('--model', type=Path, help='Defaults to model recorded in Step 07')
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--max-violations', type=int, default=30)
    args = p.parse_args()
    result = {'status': 'STRUCTURAL_FAIL', 'stage': '08b_ptq_audit_raw_litert'}
    try:
        meta = read_metadata(args.prepared_dir)
        conversion = json.loads((args.prepared_dir / '07b_ptq_convert_raw_litert_int8.json').read_text(encoding='utf-8'))
        if meta.get('quantization_method') != 'ptq':
            raise RuntimeError('PTQ audit refuses QAT metadata')
        if conversion.get('status') != 'CONVERSION_PASS':
            raise RuntimeError('Step 07 conversion not passed')
        if conversion.get('source_onnx_sha256') != meta['onnx_sha256']:
            raise RuntimeError('Step 07 conversion was made from an outdated Step 06 ONNX')
        model = (args.model or Path(conversion['model_path'])).resolve()
        if sha256(model) != conversion['model_sha256']:
            raise RuntimeError('LiteRT model differs from Step 07 artifact')
        if int(meta['nc']) <= 0:
            raise RuntimeError('Invalid Step 06 metadata')
        itp = make_interpreter(model, threads=args.threads)
        checks = audit_interpreter(itp)
        result.update(checks)
        result.update(model_path=str(model), model_sha256=sha256(model))
        print('External input(s):', checks['input_details'])
        print('External output(s):', checks['output_details'])
        print('Executable ops:', checks['executable_operator_histogram'])
        print('Live tensors:', checks['live_tensor_count'])
        print('Violations:', len(checks['violations']))
        for row in checks['violations'][:args.max_violations]:
            print('VIOLATION:', row)
    except Exception as exc:
        result['status'] = 'STRUCTURAL_FAIL'
        result['error'] = f'{type(exc).__name__}: {exc}'
        print(f'STRUCTURAL_FAIL: {result["error"]}', file=sys.stderr)
    finally:
        path = args.prepared_dir / '08b_ptq_audit_raw_litert.json'
        path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        print('VERDICT:', result['status'])
    return 0 if result['status'] == 'STRUCTURAL_PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
