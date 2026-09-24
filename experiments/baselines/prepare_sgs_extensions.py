#!/usr/bin/env python3
"""Apply portability-only changes to separately cloned public extensions.

Run after cloning sources pinned in sgs_sources.lock.json. Idempotent headers;
feature copy is created once. No rendering equations are changed here.
"""
from pathlib import Path
import shutil
root=Path(__file__).resolve().parents[2]
r3dg=root/'third_party/r3dg_upstream'
for rel,header in [('submodules/simple-knn/simple_knn.cu','#include <cfloat>'),
                   ('r3dg-rasterization/cuda_rasterizer/rasterizer_impl.h','#include <cstdint>')]:
    p=r3dg/rel;s=p.read_text()
    if header not in s:p.write_text(header+'\n'+s)
src=root/'third_party/feature3dgs_upstream/submodules/diff-gaussian-rasterization-feature'
dst=root/'third_party/sgs_components/feature_rasterizer'
if not dst.exists():
    shutil.copytree(src,dst,symlinks=True)
    (dst/'diff_gaussian_rasterization').rename(dst/'diff_gaussian_rasterization_feature')
    p=dst/'setup.py';p.write_text(p.read_text().replace('diff_gaussian_rasterization','diff_gaussian_rasterization_feature'))
    p=dst/'cuda_rasterizer/config.h';p.write_text(p.read_text().replace('NUM_SEMANTIC_CHANNELS 128','NUM_SEMANTIC_CHANNELS 32'))
p=dst/'cuda_rasterizer/rasterizer_impl.h';s=p.read_text()
if '#include <cstdint>' not in s:p.write_text('#include <cstdint>\n'+s)
print('Prepared CUDA portability headers and 32-channel semantic extension')
