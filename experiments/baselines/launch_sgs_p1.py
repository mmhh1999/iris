"""Run the isolated SGS P1 pilot with local CUDA toolchain."""
import os,sys,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[2]
tool=root/'.toolchain/compiler';nvidia=root/'.venv/lib/python3.10/site-packages/nvidia'
env=os.environ.copy()
env.update(CUDA_HOME=str(tool),CC=str(tool/'bin/x86_64-conda-linux-gnu-gcc'),CXX=str(tool/'bin/x86_64-conda-linux-gnu-g++'),MAX_JOBS='4',LD_LIBRARY_PATH=str(tool/'lib'),CPATH=':'.join(map(str,[tool/'targets/x86_64-linux/include',*sorted(nvidia.glob('*/include'))])),LIBRARY_PATH=':'.join(map(str,[tool/'targets/x86_64-linux/lib',*sorted(nvidia.glob('*/lib')),Path('/usr/lib/wsl/lib')])),PYTHONPATH=str(root/'.baseline_deps/sgs'),HF_HOME=str(root/'.baseline_deps/hf_cache'),SGS_RGBX_MODEL=str(root/'.baseline_deps/hf_cache/hub/models--zheng95z--rgb-to-x/snapshots/b38b3fd73a14ea62f3953fc54bc4ac67b067bae0'),TORCH_HOME=str(root/'.baseline_deps/torch_cache'),MPLCONFIGDIR='/tmp/sgs_matplotlib',SGS_SAM2_CHECKPOINT=str(root/'.baseline_deps/weights/sam2.1_hiera_large.pt'),TORCH_EXTENSIONS_DIR=str(root/'.baseline_deps/torch_extensions'),PATH=str(root/'.baseline_deps/sgs/bin')+':'+env['PATH'])
raise SystemExit(subprocess.call([str(root/'.venv/bin/python'),str(root/'experiments/baselines/run_sgs_p1_pilot.py'),*sys.argv[1:]],env=env,cwd=root))
