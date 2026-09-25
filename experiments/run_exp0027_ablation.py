"""EXP0027: seed-repeated ablation of EXP0026 on the EXP0019 kitchen (data_03).

Separates the two EXP0026 ingredients -- the explicit sun term and the added
albedo-consistency regularizer -- so a small lit/shadow albedo gap can be
attributed to the sun term rather than to a flattened albedo field.
Arms: vanilla, sun_only, reg_only, full (= EXP0026) x conditions sun_a, sky x seeds.
Resumable: finished runs (stages.json containing 12_render) are skipped.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import disk_guard

ROOT = Path(__file__).resolve().parents[1]
SUN_A = '-0.383022221559489,0.42261826174069944,-0.8213938048432696'
ARMS = ['reg_only', 'vanilla', 'full', 'sun_only']
PER_RUN_GB = 1.2


def done(run):
    stages = run / 'stages.json'
    return stages.exists() and any(s['stage'] == '12_render' and s['returncode'] == 0
                                   for s in json.loads(stages.read_text())['stages'])


def main(a):
    base = a.base.resolve(); base.mkdir(parents=True, exist_ok=True)
    if not (base / 'data_03').exists():
        (base / 'data_03').symlink_to(a.data.resolve())
    disk_guard.check(base, need_gb=PER_RUN_GB * len(a.seeds) * len(a.arms) * 2)
    for seed in a.seeds:
        for arm in a.arms:
            suffix = f'_{arm}_s{seed}'
            for condition in ['sun_a', 'sky']:
                run = base / f'iris_{condition}{suffix}'
                if done(run):
                    continue
                if run.exists():
                    # Interrupted (e.g. wsl --shutdown); outputs are fully regenerable.
                    print('REMOVE incomplete', run.name, flush=True)
                    shutil.rmtree(run)
                cmd = [sys.executable, 'experiments/run_sunpatch_iris_daylight.py',
                       '--data', base / 'data_03' / condition, '--output', run,
                       '--steps', a.steps, '--seed', seed, '--stage-timeout', 1800]
                if arm != 'vanilla':
                    cmd += ['--sun-toward-world=' + SUN_A, '--sun-lr-scale', 0.1, '--sun-ablation', arm]
                print('RUN', run.name, flush=True)
                subprocess.run(list(map(str, cmd)), cwd=ROOT, check=True)
            ev = base / f'eval{suffix}'
            if not ev.exists():
                with (base / f'eval{suffix}.log').open('w') as log:
                    subprocess.run([sys.executable, 'experiments/evaluate_sunpatch_iris.py', '--base', str(base),
                                    '--run-suffix', suffix, '--out', ev.name, '--conditions', 'sun_a', 'sky'],
                                   cwd=ROOT, check=True, stdout=log, stderr=subprocess.STDOUT)
            for condition in ['sun_a', 'sky']:
                # Evaluation reads only last_1; drop the earlier stage handoffs (~0.3 GB each).
                model = base / f'iris_{condition}{suffix}' / 'checkpoints' / f'sunpatch_iris_{condition}{suffix}'
                for stale in ['init.ckpt', 'last_0.ckpt']:
                    (model / stale).unlink(missing_ok=True)
            print('EVAL', ev.name, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--base', type=Path, default=ROOT / 'experiments/out/EXP0027_ablation')
    p.add_argument('--data', type=Path, default=ROOT / 'experiments/out/EXP0019_sunpatch/data_03')
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--arms', nargs='+', choices=ARMS, default=ARMS)
    p.add_argument('--steps', type=int, default=200)
    main(p.parse_args())
