#!/usr/bin/env bash
# Build PyTorch main branch with ROCm 7.2.2 for gfx1201 (RX 9070).
# Goal: enable native expandable_segments support (lands in main, ROCm>=7.0).
# Verified working configure step on 2026-04-30.
set -euo pipefail

VENV=~/.venvs/torch-build
SRC=~/src/pytorch-build/pytorch
LOG=~/src/pytorch-build/build.log

cd "$SRC"

# Build configuration
export USE_ROCM=1
export USE_CUDA=0
export PYTORCH_ROCM_ARCH=gfx1201
export ROCM_PATH=/opt/rocm
export ROCM_HOME=/opt/rocm
export HIP_PATH=/opt/rocm

# Trim unused features to speed build
export USE_KINETO=0
export USE_FBGEMM=0
export USE_NNPACK=0
export USE_QNNPACK=0
export USE_XNNPACK=0
export USE_MKLDNN=1
export BUILD_TEST=0
export USE_DISTRIBUTED=1
export USE_GLOO=1
export USE_MPI=0
export USE_TENSORPIPE=0

# Memory-bound: 17 GiB RAM, link stage may peak ~10 GiB per job
export MAX_JOBS=6

# Use venv's compilers/cmake/ninja
export PATH="$VENV/bin:/opt/rocm/bin:/opt/rocm/llvm/bin:$PATH"
export CMAKE_PREFIX_PATH="$VENV"

# Clean any leftover build state
rm -rf build || true

# Run hipify to generate ROCm sources from CUDA originals (MANDATORY).
# Without this, cmake configure fails with:
#   "File c10/hip/impl/hip_cmake_macros.h.in does not exist"
echo "=== running tools/amd_build/build_amd.py (hipify) at $(date) ==="
"$VENV/bin/python" tools/amd_build/build_amd.py 2>&1 | tail -20
echo "=== hipify done at $(date) ==="

echo "=== build env ==="
env | grep -E "USE_|PYTORCH_|ROCM|HIP|MAX_JOBS|CMAKE_PREFIX" | sort
echo "=== python ==="
"$VENV/bin/python" -V
"$VENV/bin/python" -c "import sys; print(sys.executable)"
echo "=== starting build at $(date) ==="

"$VENV/bin/pip" install -e . --no-build-isolation -v 2>&1 | tee "$LOG"

echo "=== finished at $(date) ==="
