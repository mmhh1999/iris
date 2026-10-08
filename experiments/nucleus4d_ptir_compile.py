"""Compile and instantiate both official training backends on the local GPU."""
from pathlib import Path
import sys
import subprocess
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
import threedgrut.utils.misc
from threedgrt_tracer import Tracer as GS
from threedgptir_tracer import Tracer as PTIR
OmegaConf.register_new_resolver('int_list', lambda l: [int(x) for x in l], replace=True)
root = Path(__file__).resolve().parents[1]
if len(sys.argv)==1:
    for stage in ('gs', 'ptir'):
        subprocess.run([sys.executable, __file__, stage], check=True)
    print('ALL COMPILED', flush=True)
    sys.exit(0)
with initialize_config_dir(config_dir=str(root/'third_party/ptir_gs/configs'), version_base=None):
    for config, cls in ([('apps/colmap_3dgrt', GS)] if sys.argv[1]=='gs' else [('inversions/colmap_3dgptir', PTIR)]):
        print('Compiling', config, flush=True)
        obj = cls(compose(config_name=config))
        print('READY', config, flush=True)
