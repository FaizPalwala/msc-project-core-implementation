# HPC CookBook — Machine Unlearning on Aire

## Environment Setup

```bash
# 1. Load modules
module purge
module load miniforge
module load cuda/12.6.2

# 2. Create and activate the environment
conda create -n core python=3.10 -y
conda activate core

# 3. Install PyTorch for CUDA 12.4 (must precede -e . so pip doesn't
#    resolve torch>=2.0 from PyPI and pull the CPU build)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 4. Remaining deps + console scripts from pyproject.toml (single source of truth)
pip install -e .

# 5. cuDNN: PyTorch ships its own — no extra install needed for ResNet-18 training

```

## Storage Layout

```
/nobackup/<username>/unlearning_project/
├── data/
│   ├── dataset/                      ← SFHQ-InstantID CSVs + images
│   │   ├── dataset.csv               ← balanced (600 IDs, 75 imgs/ID)
│   │   ├── dataset_imbalanced.csv    ← 85:40:20 gradient
│   │   └── images/                   ← 128×128 aligned face crops
│   └── ...
├── results/
│   ├── checkpoints/                  ← trained models (.pt)
│   ├── single_shot/                  ← single-shot evaluation output
│   ├── iterative/                    ← iterative protocol output
│   ├── hparam/                       ← HP search trials
│   └── ablation/                     ← ablation study results
└── logs/                             ← Slurm .out/.err files
```

## Pipeline Stages

```
train ──→ single_shot ──→ iterative ──→ stability
  │            │               │              │
  └────────────┴───────────────┘              │
    (single-shot + HP search                  │
     run in parallel after train)             │
                                              │
                         ┌────────────────────┘
                         ▼
                   13 publication-quality PNGs
```

## Job Submission

### Full Pipeline (automatic chaining)

```bash
# Submit the entire pipeline — Slurm handles dependencies
sbatch scripts/hpc_full_pipeline.sh

# Monitor progress
squeue -u $USER
```

### Individual Stages

```bash
# 1. Train original model (30 epochs, ~4-6 hr on L40S)
JOB_TRAIN=$(sbatch --parsable \
    --job-name=unlearn_train \
    --time=12:00:00 --partition=gpu --gres=gpu:1 \
    --cpus-per-task=8 --mem=32G \
    scripts/slurm_train.sh)

# 2a. Single-shot evaluation (~12-24 hr, all 10 methods)
JOB_SINGLE=$(sbatch --parsable \
    --dependency=afterok:$JOB_TRAIN \
    --job-name=unlearn_single \
    --time=24:00:00 --partition=gpu --gres=gpu:1 \
    --cpus-per-task=8 --mem=32G \
    scripts/slurm_single_shot.sh)

# 2b. HP search (parallel with single-shot, ~8-12 hr per method)
JOB_HP=$(sbatch --parsable \
    --dependency=afterok:$JOB_TRAIN \
    --job-name=unlearn_hp \
    --time=24:00:00 --partition=gpu --gres=gpu:1 \
    --cpus-per-task=8 --mem=32G \
    scripts/slurm_hparam.sh)

# 3. Iterative protocol (~24-48 hr, 15 steps × multiple methods)
JOB_ITER=$(sbatch --parsable \
    --dependency=afterok:$JOB_SINGLE:$JOB_HP \
    --job-name=unlearn_iter \
    --time=48:00:00 --partition=gpu --gres=gpu:1 \
    --cpus-per-task=8 --mem=32G \
    scripts/slurm_iterative.sh)

# 4. Stability plots (~1 hr, CPU-only after iterative)
sbatch --dependency=afterok:$JOB_ITER \
    --job-name=unlearn_stab \
    --time=2:00:00 --partition=gpu --gres=gpu:1 \
    --cpus-per-task=4 --mem=16G \
    scripts/slurm_stability.sh
```

### Smoke Testing (Before Full Run)

```bash
# Single-shot, scaled 10× down, 1 seed, 2 methods, ~15 min
python src/single_shot.py \
    --csv data/dataset/dataset.csv \
    --model results/checkpoints/original_model_best.pt \
    --scale 0.1 --n_seeds 1 --skip_retrain \
    --methods ga adaptiformet

# Iterative, 3 steps, smoke scale, ~30 min
python src/iterative.py \
    --csv data/dataset/dataset.csv \
    --model results/checkpoints/original_model_best.pt \
    --n_steps 3 --scale 0.1 --n_seeds 1 \
    --methods adaptiformet
```

### Multi-Seed Runs (for error bars)

```bash
# Single-shot with 5 seeds → μ ± σ in aggregated table
python src/single_shot.py \
    --csv data/dataset/dataset.csv \
    --model results/checkpoints/original_model_best.pt \
    --n_seeds 5 --seed 42

# Iterative with 5 seeds → independent seed_N/ subdirectories
python src/iterative.py \
    --csv data/dataset/dataset.csv \
    --model results/checkpoints/original_model_best.pt \
    --n_seeds 5 --seed 42
```

## Resource Estimates

| Stage | GPU | CPUs | Memory | Wall time (full) | Wall time (smoke) |
|-------|-----|------|--------|-------------------|--------------------|
| Train | 1 L40S | 8 | 32 GB | 4-6 hr | 30 min |
| Single-shot | 1 L40S | 8 | 32 GB | 12-24 hr | 1 hr |
| HP search (per method) | 1 L40S | 8 | 32 GB | 8-12 hr | 1 hr |
| Iterative | 1 L40S | 8 | 32 GB | 24-48 hr | 2 hr |
| Stability plots | CPU | 4 | 16 GB | 30 min | 5 min |
| **Total pipeline** | — | — | — | **~48-72 hr** | **~4 hr** |

## Parallelisation Strategy

```
                    ┌─ single_shot ──────────┐
train ──────────────┤                        ├── iterative ── stability
                    └─ hparam_search ────────┘
                           (1 job per method,
                            all parallel)
```

- Single-shot and HP search are independent after training — submit both simultaneously.
- HP search can be further parallelised: one Slurm job per method (ng_plus, msg, msg_kd, ct, adaptiformet).
- Iterative depends on both — uses best HP configs from search + single-shot as baseline.

## Common Issues

### Out of memory (L40S, 46 GB)
- 224×224 ResNet-18 with batch_size=64: ~12 GB VRAM
- Reduce `--batch_size` to 32 if running multiple processes per GPU
- Aire L40S nodes have 3 GPUs; request `--gres=gpu:1` to avoid sharing

### Checkpoint bloat
- 134 MB per checkpoint (includes optimizer state)
- 10 methods × 15 iterative steps × checkpoints every 5 = ~30 checkpoints = ~4 GB
- Set `--checkpoint_every 15` for final-step only, or `0` to disable

### Job killed (pre-emption)
- Aire may pre-empt long-running GPU jobs on shared nodes
- All stages use incremental JSONL logging — restart picks up where it left off
- HP search: re-run with same `--out` dir, completed trials are skipped
- Iterative: per-step JSONL is append-only, survives partial runs

## Aire-Specific Notes

- **Partition**: `gpu` (28 nodes × 3 L40S GPUs, 168 cores each)
- **Flash storage**: `$TMP_SHARED` (1 TB/job, auto-purged). Stage datasets there for I/O-heavy phases.
- **Lustre scratch**: `/scratch/$USER/` — large capacity, slower I/O. Store persistent results here.
- **Home**: `$HOME` — small quota, versioned backups. Store code and small configs only.
- **Max wall time**: 72 hr on gpu partition.
- **Interactive**: `srun --partition=gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=4:00:00 --pty bash`
