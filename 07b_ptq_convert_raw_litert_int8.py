#!/usr/bin/env python3
"""Step 07B: independent static PTQ calibration and onnx2tf full INT8 convert.

Input ONNX is floating-point from original FP32 model (06d), NOT QAT.
Representative 640^2 RGB samples are NHWC float32 [0,1]. For 3000 samples,
use disk-backed NPY: converter may still require ~14 GiB RAM when loading it.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from ptq_dataset_common import (create_representative_npy, images_from_manifest,
                                read_split_images, save_json, select_images,
                                sha256, validate_representative_npy)
from ptq_raw_head_postprocess import read_metadata


def cli():
    p = argparse.ArgumentParser(description='Static PTQ: raw-head float ONNX -> LiteRT full INT8')
    p.add_argument('--prepared-dir', type=Path, required=True, help='Output directory of 06d (PTQ)')
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--data', type=Path, help='Ultralytics data.yaml; select images from train split')
    group.add_argument('--calib-manifest', type=Path, help='Explicit calibration image list')
    group.add_argument('--calib-npy', type=Path, help='Already-prepared NHWC float32 calibration NPY')
    p.add_argument('--dataset-root', type=Path, help='Required with --calib-manifest')
    p.add_argument('--calib-samples', type=int, default=3000,
                   help='Number of train samples, deterministic random subset (default 3000)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output-dir', type=Path, help='Default: <prepared-dir>/07b_litert_int8')
    p.add_argument('--force', action='store_true')
    return p.parse_args()


def main():
    args = cli()
    work = args.prepared_dir.expanduser().resolve()
    report = work / '07b_ptq_convert_raw_litert_int8.json'
    result = dict(status='CONVERSION_FAIL', stage='07b_ptq_convert_raw_litert_int8')
    try:
        if args.calib_samples < 1:
            raise ValueError('--calib-samples must be positive')
        meta = read_metadata(work)
        if meta.get('quantization_method') != 'ptq':
            raise RuntimeError('Step 07B requires 06d PTQ metadata; refusing QAT experiment')
        source = Path(meta['onnx_path'])
        if not source.is_file() or sha256(source) != meta['onnx_sha256']:
            raise RuntimeError('ONNX missing or sha256 mismatch; re-run Step 06D')
        hw = tuple(int(x) for x in meta['imgsz_hw'])
        out = (args.output_dir or work / '07b_litert_int8').resolve()
        if out.exists() and any(out.iterdir()) and not args.force:
            raise RuntimeError(f'Non-empty PTQ output directory: {out}; use --force')
        if args.calib_npy:
            calib = args.calib_npy.expanduser().resolve()
            if not calib.is_file():
                raise FileNotFoundError(calib)
            source_desc = 'existing_npy'
        else:
            if args.data:
                available, data_spec, _ = read_split_images(args.data, 'train')
                if 'nc' in data_spec and int(data_spec['nc']) != int(meta['nc']):
                    raise ValueError(f'data.yaml nc={data_spec["nc"]} does not match model nc={meta["nc"]}')
                chosen = select_images(available, args.calib_samples, seed=args.seed, shuffle=True)
                source_desc = f'data_yaml_train:{args.data.expanduser().resolve()}'
            else:
                if args.dataset_root is None:
                    raise ValueError('--dataset-root is required with --calib-manifest')
                available = images_from_manifest(args.calib_manifest, args.dataset_root)
                chosen = select_images(available, args.calib_samples, seed=args.seed, shuffle=False)
                source_desc = f'manifest:{args.calib_manifest.expanduser().resolve()}'
            calib = (work / '07b_representative_nhwc_float32.npy').resolve()
            bytes_required = len(chosen) * hw[0] * hw[1] * 3 * 4
            free = shutil.disk_usage(work).free
            if free < bytes_required + 256 * 1048576:
                raise RuntimeError(f'Insufficient disk for calibration NPY: need {bytes_required / 1073741824:.2f} GiB plus safety margin, available {free / 1073741824:.2f} GiB')
            print(f'Calibration disk estimate: {bytes_required / 1073741824:.2f} GiB; converter may require more system RAM.', flush=True)
            # Prevent partially written calibrations after an interrupted run.
            create_representative_npy(chosen, hw, calib)
            (work / '07b_calibration_images.txt').write_text(
                '\n'.join(str(p) for p in chosen) + '\n', encoding='utf-8')
        count = validate_representative_npy(calib, hw)
        if args.data or args.calib_manifest:
            if count != args.calib_samples:
                raise RuntimeError(f'Calibration size mismatch {count} != {args.calib_samples}')
        else:
            if count != args.calib_samples:
                print(f'NOTE: existing NPY contains {count} samples, --calib-samples={args.calib_samples} '
                      'is ignored for --calib-npy', flush=True)
        if out.exists() and any(out.iterdir()):
            # We already checked above that --force is explicitly set.
            shutil.rmtree(out)
        out.mkdir(parents=True, exist_ok=True)
        command = [
            'onnx2tf', '-i', str(source), '-o', str(out), '-tb', 'tf_converter',
            '-oiqt', '-iqd', 'int8', '-oqd', 'int8', '-cind', 'images', str(calib),
            '[[[[0.0,0.0,0.0]]]]', '[[[[1.0,1.0,1.0]]]]',
        ]
        log_path = work / '07b_onnx2tf_convert.log'
        print('Static PTQ input:', source)
        print('Representative images:', count, 'source:', source_desc)
        print('Command:', ' '.join(command), flush=True)
        with log_path.open('w', encoding='utf-8') as f:
            proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, bufsize=1)
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                f.write(line)
            rc = proc.wait()
        result.update(source_onnx_sha256=meta['onnx_sha256'],
                      source_checkpoint_sha256=meta['base_sha256'],
                      representative_images=count, representative_file=str(calib),
                      representative_sha256=sha256(calib), representative_source=source_desc,
                      conversion_command=command, converter_exit_code=rc,
                      output_directory=str(out))
        if rc != 0:
            raise RuntimeError(f'onnx2tf exited {rc}; see {log_path}')
        matches = list(out.rglob('*full_integer_quant.tflite'))
        if len(matches) != 1:
            raise RuntimeError(f'Expected exactly one *full_integer_quant.tflite, found {len(matches)}')
        tflite = matches[0].resolve()
        result.update(status='CONVERSION_PASS', model_path=str(tflite),
                      model_sha256=sha256(tflite), size_bytes=tflite.stat().st_size,
                      note='Conversion alone is not proof of integer-only; run 08b audit')
        print(f'CONVERSION_PASS | {tflite} | {tflite.stat().st_size / 1048576:.2f} MiB')
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
        print('CONVERSION_FAIL:', result['error'], file=sys.stderr)
    finally:
        save_json(report, result)
    return 0 if result['status'] == 'CONVERSION_PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
