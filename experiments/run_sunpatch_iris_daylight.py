"""Same bounded stock-IRIS pipeline as run_sunpatch_iris.py, plus one new
opt-in stage (bake_sun_term.py) and threading --sun-toward-world through to
train_brdf_crf.py's stages, to test the daylight-aware explicit-sun term
(research/SUNPATCH_DAYLIGHT_METHOD_ZH.md) on the exact same EXP0019/EXP0024
benchmark scene, camera split, and step budget.

Passing --sun-toward-world is the only difference from run_sunpatch_iris.py's
behavior; omitting it reproduces vanilla IRIS exactly (same code path,
sun_vis stays absent, train_brdf_crf.py's added term is inert).
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--stage-timeout', type=int, default=300)
    p.add_argument('--output', type=Path, required=True)
    p.set_defaults(full_training=False)
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--skip-bake', action='store_true')
    p.add_argument('--start-at', type=int, default=1,
                   help='Resume at a numbered stage, retaining earlier artifacts and records.')
    p.add_argument('--sun-toward-world', type=str, default=None,
                   help='comma-separated x,y,z unit vector; enables the explicit daylight sun term')
    p.add_argument('--sun-lr-scale', type=float, default=1.0,
                   help='learning-rate multiplier for the sun_irradiance parameter (see EXP0026)')
    a = p.parse_args()
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    data = a.data.resolve()
    bake = out / 'bake'; shading = out / 'shading'; ckpts = out / 'checkpoints'
    exp = 'sunpatch_' + out.name
    model = ckpts / exp
    env = os.environ.copy()
    env.update(OPENCV_IO_ENABLE_OPENEXR='1', MPLCONFIGDIR='/tmp/iris-mpl',
               LD_LIBRARY_PATH=str(ROOT / '.toolchain/compiler/lib') + ':' + env.get('LD_LIBRARY_PATH',''))
    record_path = out/'stages.json'
    if record_path.exists() and a.start_at == 1:
        raise FileExistsError('Use a fresh output directory or explicitly resume with --start-at')
    records = json.loads(record_path.read_text())['stages'] if record_path.exists() else []
    def run(name, script, args):
        if int(name.split('_')[0]) < a.start_at:
            return False
        cmd = [sys.executable, script, *map(str, args)]
        start = time.time()
        print('START', name, flush=True)
        with (out / (name + '.log')).open('w') as f:
            r = subprocess.run(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=a.stage_timeout)
        records.append(dict(stage=name, command=cmd, returncode=r.returncode, seconds=time.time()-start))
        (out/'stages.json').write_text(json.dumps(dict(full_training=a.full_training,
            steps=None if a.full_training else a.steps, stages=records,
            sun_toward_world=a.sun_toward_world), indent=2))
        print('END', name, r.returncode, round(time.time()-start,1), 'seconds', flush=True)
        if r.returncode: raise SystemExit(r.returncode)
        return True
    def promote(filename):
        import torch
        final = model / 'final.ckpt'
        checkpoint = torch.load(final, map_location='cpu', weights_only=False)
        step = int(checkpoint['global_step'])
        if not a.full_training and step != a.steps:
            raise RuntimeError(f'Expected final step {a.steps}, got {step}: {final}')
        del checkpoint
        shutil.move(final, model / filename)
        audit_path = out / 'checkpoint_handoffs.json'
        audit = json.loads(audit_path.read_text()) if audit_path.exists() else []
        audit.append({'stage': records[-1]['stage'], 'checkpoint': filename, 'global_step': step})
        audit_path.write_text(json.dumps(audit, indent=2))

    ds = ['--scene', data, '--dataset', 'synthetic', '--ldr_img_dir', 'Image']
    if not a.skip_bake:
        run('01_slf_bake','slf_bake.py',ds+['--output',bake,'--voxel_num',128])
    elif not (bake/'vslf.npz').exists():
        raise FileNotFoundError('Cannot skip missing SLF bake')
    run('02_emitter_extract','extract_emitter_ldr.py',ds+['--output',bake,'--threshold',0.99])
    common = ['--experiment_name',exp,'--checkpoint_path',ckpts,
        '--dataset','synthetic',data,'--emitter_path',bake/'emitter.pth',
        '--has_part',1,'--ldr_img_dir','Image','--val_frame',0,
        '--SPP',32,'--spp',16,'--crf_basis',3,'--num_workers',0]
    if not a.full_training: common += ['--max_steps',a.steps]
    if run('03_initialize','initialize.py',common+['--max_epochs',1000,'--voxel_path',bake/'vslf.npz']):
        promote('init.ckpt')
    run('04_emitter_update','extract_emitter_ldr.py',ds+['--mode','update','--output',bake,'--ckpt',model/'init.ckpt'])
    run('05_bake_shading','bake_shading.py',ds+['--slf_path',bake/'vslf.npz','--emitter_path',bake/'emitter.pth','--output',shading])
    if a.sun_toward_world:
        run('051_bake_sun_term','bake_sun_term.py',ds+['--output',shading,'--sun-toward-world='+a.sun_toward_world])
    brdf = ['--max_epochs',1000,'--cache_dir',shading,'--lp',0.005,'--la',0.0,'--l_crf_weight',0.001,'--sun_lr_scale',a.sun_lr_scale]
    if run('06_brdf_crf','train_brdf_crf.py',common+brdf+['--dir_val','val_0','--ckpt_path',model/'init.ckpt','--voxel_path',bake/'vslf.npz']):
        promote('last_0.ckpt')
    run('07_slf_refine','slf_refine.py',ds+['--output',bake,'--load','vslf.npz','--save','vslf_0.npz','--ckpt',model/'last_0.ckpt','--crf_basis',3])
    if run('08_train_emitter','train_emitter.py',common+['--max_epochs',1000,'--dir_val','val_0_emitter','--ckpt_path',model/'last_0.ckpt','--voxel_path',bake/'vslf_0.npz']):
        promote('last_0.ckpt')
    run('09_emitter_update','extract_emitter_ldr.py',ds+['--mode','update','--output',bake,'--ckpt',model/'last_0.ckpt'])
    run('10_refine_shading','refine_shading.py',ds+['--slf_path',bake/'vslf_0.npz','--emitter_path',bake/'emitter.pth','--ckpt',model/'last_0.ckpt','--output',shading])
    if a.sun_toward_world:
        # Purely geometric (mesh-only); unaffected by the emitter/SLF refine above, but
        # refine_shading.py may not clear stale files, so rebake defensively into the
        # same directory for the second BRDF-CRF pass.
        run('101_bake_sun_term','bake_sun_term.py',ds+['--output',shading,'--sun-toward-world='+a.sun_toward_world])
    if run('11_brdf_crf','train_brdf_crf.py',common+brdf+['--dir_val','val_1','--ckpt_path',model/'init.ckpt','--voxel_path',bake/'vslf_0.npz']):
        promote('last_1.ckpt')
    run('12_render','render.py',['--experiment_name',exp,'--checkpoint_path',ckpts,'--ckpt','last_1.ckpt',
        '--dataset','synthetic',data,'--emitter_path',bake,'--output_path',out/'render',
        '--split','val','--ldr_img_dir','Image','--SPP',64,'--spp',16,'--crf_basis',3])

if __name__ == '__main__': main()
