#!/usr/bin/env bash
# Isolated PTIR dependencies; reuse the existing Torch and CUDA installations.
PTIR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_HOME="$PTIR_ROOT/.toolchain/compiler"
export CC="$CUDA_HOME/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CUDA_HOME/bin/x86_64-conda-linux-gnu-g++"
export PATH="$PTIR_ROOT/.toolchain/ptir-slang/bin:$PTIR_ROOT/.baseline_deps/ptir/bin:$CUDA_HOME/bin:$PTIR_ROOT/.venv/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/targets/x86_64-linux/lib:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PTIR_ROOT/.baseline_deps/ptir:$PTIR_ROOT/.baseline_deps/sgs:$PTIR_ROOT/third_party/ptir_gs:$PTIR_ROOT/experiments${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_EXTENSIONS_DIR="$PTIR_ROOT/.baseline_deps/ptir_extensions"
export TORCH_CUDA_ARCH_LIST="12.0"
# Use CUDA's stream-ordered allocator alongside the native OptiX buffers.
# This changes allocation/reuse only; precision and rendering settings are unchanged.
export PYTORCH_ALLOC_CONF="backend:cudaMallocAsync"
export MAX_JOBS=2
for PTIR_CUDA_INCLUDE in "$PTIR_ROOT"/.venv/lib/python3.10/site-packages/nvidia/*/include; do
    if [ -d "$PTIR_CUDA_INCLUDE" ]; then
        export CPATH="$PTIR_CUDA_INCLUDE${CPATH:+:$CPATH}"
    fi
done
export OMP_NUM_THREADS=4
export MPLCONFIGDIR=/tmp/iris-matplotlib
export OPENCV_IO_ENABLE_OPENEXR=1
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export WANDB_MODE=disabled
