#!/bin/bash
# ==========================================
# gpu_preflight.sh — GPU environment validation
# ==========================================
# Runs at the start of every GPU Slurm job to verify:
#   1. GPU is visible and accessible
#   2. CUDA driver version
#   3. PyTorch can see the GPU
#   4. cuDNN is available (if installed)
#
# Exits with code 1 if any check fails, allowing Slurm to abort the job
# before spending wall time on a misconfigured node.
# ==========================================

set -euo pipefail

echo "==========================================="
echo "  GPU PREFLIGHT CHECK"
echo "  $(date)"
echo "  Host: $(hostname)"
echo "  Job ID: ${SLURM_JOB_ID:-N/A}"
echo "==========================================="

# ── 1. nvidia-smi check ────────────────────────────────────────────────
if ! command -v nvidia-smi &>/dev/null; then
    echo "[FAIL] nvidia-smi not found — no GPU driver on this node"
    exit 1
fi

echo ""
echo "[1/5] nvidia-smi"
nvidia-smi --query-gpu=name,memory.total,driver_version,cuda_version \
    --format=csv,noheader 2>/dev/null || {
    echo "[FAIL] nvidia-smi query failed — GPU may be unavailable"
    exit 1
}
echo "  GPU present and responding"

# ── 2. CUDA_HOME / LD_LIBRARY_PATH ─────────────────────────────────────
echo ""
echo "[2/5] CUDA environment"
CUDA_HOME="${CUDA_HOME:-${CUDA_PATH:-}}"
if [ -z "$CUDA_HOME" ] && command -v nvcc &>/dev/null; then
    CUDA_HOME=$(dirname "$(dirname "$(which nvcc)")")
fi
echo "  CUDA_HOME: ${CUDA_HOME:-not set}"
echo "  LD_LIBRARY_PATH (cuda): $(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -i cuda | head -3 | tr '\n' ' ')"

# ── 3. PyTorch GPU check ───────────────────────────────────────────────
echo ""
echo "[3/5] PyTorch GPU availability"
python3 -c "
import torch
print(f'  PyTorch version: {torch.__version__}')
print(f'  CUDA available:  {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA version:    {torch.version.cuda}')
    print(f'  GPU count:       {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'  GPU[{i}]: {props.name} ({props.total_memory // 1024**2} MiB)')
    # Quick tensor test
    x = torch.randn(1000, 1000, device='cuda')
    y = x @ x.T
    print(f'  Tensor ops:      OK (1000x1000 matmul passed)')
else:
    print('  [FAIL] CUDA not available in PyTorch')
    exit(1)
" || {
    echo "[FAIL] PyTorch GPU check failed"
    exit 1
}

# ── 4. cuDNN check (optional, non-fatal) ───────────────────────────────
echo ""
echo "[4/5] cuDNN availability"
python3 -c "
import torch
try:
    cudnn_available = torch.backends.cudnn.is_available()
    print(f'  cuDNN available: {cudnn_available}')
    if cudnn_available:
        print(f'  cuDNN version:   {torch.backends.cudnn.version()}')
        # Benchmark a small conv to verify cuDNN is working
        conv = torch.nn.Conv2d(3, 64, 3, padding=1).cuda()
        x = torch.randn(8, 3, 224, 224, device='cuda')
        y = conv(x)
        print(f'  Conv2d test:     OK (3→64, 224×224, batch=8)')
except Exception as e:
    print(f'  [WARN] cuDNN check failed (non-fatal): {e}')
" || echo "  [WARN] cuDNN check skipped — continuing"

# ── 5. VRAM headroom ───────────────────────────────────────────────────
echo ""
echo "[5/5] VRAM headroom"
python3 -c "
import torch
if torch.cuda.is_available():
    free, total = torch.cuda.mem_get_info(0)
    free_gb = free / 1024**3
    total_gb = total / 1024**3
    print(f'  Free VRAM:  {free_gb:.1f} GiB / {total_gb:.1f} GiB')
    if free_gb < 4.0:
        print(f'  [WARN] Less than 4 GiB free — check for zombie processes')
" || echo "  [WARN] VRAM check skipped"

echo ""
echo "==========================================="
echo "  GPU PREFLIGHT: ALL CHECKS PASSED"
echo "  $(date)"
echo "==========================================="
