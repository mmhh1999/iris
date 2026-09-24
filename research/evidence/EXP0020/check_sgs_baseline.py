#!/usr/bin/env python3
"""Record a bounded official-SGS startup attempt and source completeness audit.

This is a reproducibility gate, not a BRDF benchmark or a replacement model.
Exit 2 means no scientific result may be reported from this attempt.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--python', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--extra-packages', type=Path)
    a = p.parse_args()
    repo, out = a.repo.resolve(), a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    required = ['rgb2x/pipeline_rgb2x_myversion.py', 'rgb2x/load_image.py',
                'r3dg-rasterization/cuda_rasterizer/rasterizer_impl.cu',
                'r3dg-rasterization/ext.cpp']
    source_checks = {name: (repo / name).is_file() for name in required}
    links = {str(x.relative_to(repo)): os.readlink(x) for x in repo.iterdir() if x.is_symlink()}
    manifest = a.data.resolve() / 'manifest.json'
    env = os.environ.copy()
    env['MPLCONFIGDIR'] = str(out / 'matplotlib')
    if a.extra_packages:
        env['PYTHONPATH'] = str(a.extra_packages.resolve())
    command = [str(a.python.absolute()), 'train.py', '--source_path', str(a.data.resolve() / 'sun_a'),
               '--model_path', str(out / 'model'), '--eval', '--n_views', '12',
               '--iterations', '1', '-t', 'sgs']
    started = time.monotonic()
    timed_out = False
    with (out / 'startup.log').open('w') as log:
        try:
            run = subprocess.run(command, cwd=repo, env=env, stdout=log,
                                 stderr=subprocess.STDOUT, timeout=90)
            rc = run.returncode
        except subprocess.TimeoutExpired:
            timed_out, rc = True, None
    result = dict(
        experiment='EXP0020', purpose='official baseline reproducibility gate',
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        command=command, seconds=time.monotonic()-started, returncode=rc,
        timeout=timed_out, source_files=source_checks, symlinks=links,
        input_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        input_adapter_status='not integrated: current data is IRIS layout, not SGS layout',
        status='startup_failed' if rc != 0 else 'startup_returned_requires_output_validation',
        benchmark_completed=False, material_metrics=None,
        note='Missing source paths are checkout audit findings; installed equivalents require separate verification. '
             'No replacement CUDA kernels, priors or GT supervision were injected.')
    (out / 'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    return 2

if __name__ == '__main__':
    raise SystemExit(main())
