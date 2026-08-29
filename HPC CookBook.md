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
```

## Storage Layout

```
/mnt/scratch/<username>/core/
├── code/                          ← repo (this project)
│   ├── scripts/                   ← all slurm_*.sh entry points
│   ├── src/                       ← the evaluation suite
│   └── results/                   ← namespaced by dataset (gitignored)
│       ├── balanced/
│       └── imbalanced/
├── bench/                         ← sister dir: metadata + images
│   └── metadata/
│       ├── dataset.csv            (balanced, 750-id, 12 cols)
│       └── dataset_imbalanced.csv (imbalanced, 12 cols, no schedule cols)
└── logs/                          ← merged .out streams (job-id suffix)
```

Results are namespaced by dataset (`results/balanced` vs `results/imbalanced`)
so both artifacts can run and be analysed without clobbering.  Logs are the
single merged stream: every Python invocation uses `2>&1` into the job's
`.out`; `.err` is reserved for Slurm-level errors.

## Dataset Selection

The one pipeline override is the `DATASET` env var (balanced | imbalanced):

```bash
DATASET=balanced   bash scripts/hpc_full_pipeline.sh   # default
DATASET=imbalanced bash scripts/hpc_full_pipeline.sh
```

Identity classes, forget-step count and per-step sizes are **inferred from
the CSV** at runtime (600-id vs 750-id needs no code change).

## Pipeline Stages

**Balanced lane** — Slurm dependency graph (edges = `sbatch --dependency`):

```
                                  ┌──> canary ────────────────────────────────────────────────────────┐
                                  │                                                                   │
                                  ├──> ablation ──────────────────────────────────────────────────────┤
                                  │                                                                   │
[ train ] ──> single_shot ──> hparam ×9 ──> single_shot_best ──┐                                      │
                                  │                            │                                      │
                                  └──> iterative ×2 ───────────┴──> stability ×2 ─────────────────────┴──> [ report ]

[ feasibility ] (parallel, no dependency)────────────────────────────────────────────────────────────────> [ report ]
```

*GPU counts: train 1 GPU; single_shot 1 GPU; hparam 1 GPU × 9 jobs (parallel);
single_shot_best 1 GPU; iterative 1 GPU × 2 lanes; stability CPU; ablation,
canary, feasibility 1 GPU each; report CPU.*

**Imbalanced lane** (replaces the schedule lanes — axis is the popularity
gradient, not time):

```
                                  ┌──> canary ──────────────────────────────────────────────────────────────────────────┐
                                  │                                                                                     │
                                  ├──> ablation ────────────────────────────────────────────────────────────────────────┤
                                  │                                                                                     │
[ train ] ──> single_shot ──> hparam ×9 ──> single_shot_best ────┬──> equity plots ─────────────────────────────────────┤
                                                                 │                                                      │
                                                                 ├──> per-bin oracles (B) ────┐                         │
                                                                 │                            │                         │
                                                                 └──> Protocol C select ──────┴──> budget sweep (C)─────┴──> [ report ]

[ feasibility ] (parallel, no dependency) ─────────────────────────────────────────────────────────────────────────────────> [ report ]
```

**Config routing:** only `single_shot` uses default configs. Everything
downstream — `single_shot_best`, iterative, stability inputs, equity plots,
ablation, canary unlearning — uses the HP-tuned best configs.  `iterative`
waits on `single_shot` + `hparam`; `stability` waits on its `iterative` lane +
`single_shot_best`; the report waits on every lane.

| Stage                             | Script                                    | GPU        | Description                                                                                                                                                                   |
| --------------------------------- | ----------------------------------------- | ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Train                             | `slurm_train.sh`                        | 1 L40S     | Original model on retain+forget (train subset)                                                                                                                                |
| Single-shot                       | `slurm_single_shot.sh`                  | 1 L40S     | All methods, all forget IDs,**default configs** + per-identity/demographic CSVs                                                                                         |
| HP search                         | `slurm_hparam.sh`                       | 1 L40S ×9 | **One job per method, all parallel**; each writes `{method}_best_config.json`                                                                                         |
| Single-shot best                  | `slurm_single_shot.sh <out> <best_dir>` | 1 L40S     | Same eval with**tuned** configs (headline results)                                                                                                                      |
| Iterative                         | `slurm_iterative.sh`                    | 1 L40S     | Schedule protocol (uniform 5×15 or Poisson),**tuned** configs, `--order_seed`                                                                                        |
| Stability                         | `slurm_stability.sh`                    | CPU        | 16 publication plots; per-identity + demographic plots read the**top-level aggregated** single-shot CSVs (plots 10/11 were silently skipped before the aggregation fix) |
| Equity plots (imbalanced)         | `slurm_imbalanced_plots.sh`             | CPU        | 6 per-bin equity plots + Kruskal-Wallis (**after single_shot_best — tuned**; was default-config)                                                                       |
| Ablation                          | `slurm_ablation.sh`                     | 1 L40S     | AdaptiForget component ablation (6 variants);**tuned AdaptiForget base** (after hparam)                                                                                 |
| Canary                            | `slurm_canary.sh`                       | 1 L40S     | Pixel-level ground-truth deletion proof (**tuned unlearning**, after hparam)                                                                                            |
| Report                            | `slurm_report.sh`                       | CPU        | LaTeX/Markdown synthesis of all results                                                                                                                                       |
| Per-bin oracles (imbalanced)      | `slurm_per_bin_oracle.sh`               | 1 L40S     | Protocol B: 3 retrains, each excluding only one bin's forgets                                                                                                                 |
| Protocol C selection (imbalanced) | `slurm_select_protocol_c.sh`            | CPU        | 1 best method per category by UF score                                                                                                                                        |
| Budget sweep (imbalanced)         | `slurm_budget_sweep.sh`                 | 1 L40S     | Per-bin distance-to-oracle vs unlearning budget                                                                                                                               |

The imbalanced chain skips iterative + stability (no schedule axis — the
popularity gradient IS the axis) and instead runs Protocols B + C.

## Job Submission

### Full Pipeline (automatic chaining)

```bash
# Submit the entire pipeline — Slurm handles dependencies
DATASET=balanced bash scripts/hpc_full_pipeline.sh

# Poisson schedule INSTEAD of uniform (results → results/balanced/iterative_poisson);
# both lanes run by default — override with SCHEDULES="uniform" or "poisson"
SCHEDULES=poisson DATASET=balanced bash scripts/hpc_full_pipeline.sh

# HP method set override (default: ng_plus msg ct msg_kd adaptiforget ga srl ft budget_scaled)
HP_METHODS="ng_plus ft" bash scripts/hpc_full_pipeline.sh

# Monitor progress
squeue -u $USER
```

### Individual Stages

```bash
CSV=../bench/metadata/dataset.csv
MODEL=results/balanced/checkpoints/original_model_best.pt

# 1. Train original model (30 epochs, ~4-6 hr on L40S)
sbatch scripts/slurm_train.sh "$CSV" results/balanced/checkpoints

# 2a. Single-shot evaluation (default configs, ~12-24 hr)
sbatch scripts/slurm_single_shot.sh "$CSV" "$MODEL" results/balanced/single_shot

# 2b. HP search — ONE job per method, all parallel (9 jobs for the default set)
for M in ng_plus msg ct msg_kd adaptiforget ga srl ft budget_scaled; do
    sbatch scripts/slurm_hparam.sh "$CSV" "$MODEL" results/balanced/hparam "$M" grid
done

# 2c. Single-shot with BEST configs (after all HP jobs finish)
sbatch scripts/slurm_single_shot.sh "$CSV" "$MODEL" \
    results/balanced/single_shot_best results/balanced/hparam

# 3. Iterative protocol (uniform, tuned configs, ~24-48 hr)
sbatch scripts/slurm_iterative.sh "$CSV" "$MODEL" \
    results/balanced/iterative uniform results/balanced/hparam

# 3b. Poisson schedule (independent experiment, parallel with uniform if desired)
sbatch scripts/slurm_iterative.sh "$CSV" "$MODEL" \
    results/balanced/iterative_poisson poisson results/balanced/hparam

# 4. Stability plots (CPU-only after iterative; per-identity/demographic
#    CSVs are the TOP-LEVEL aggregated files written by single_shot_best)
sbatch scripts/slurm_stability.sh \
    results/balanced/iterative/iterative_combined_aggregated.csv \
    results/balanced/iterative/plots \
    results/balanced/single_shot_best/single_shot_per_identity.csv \
    results/balanced/single_shot_best/single_shot_demographic.csv

# 5. Ablation study (AFTER hparam — the "Full" variant uses the tuned
#    AdaptiForget config; pass the hparam dir as arg 3)
sbatch scripts/slurm_ablation.sh "$CSV" results/balanced/ablation results/balanced/hparam

# 6. Canary verification (AFTER hparam — unlearning uses tuned GA +
#    AdaptiForget configs; pass the hparam dir as arg 3)
sbatch scripts/slurm_canary.sh "$CSV" results/balanced/canary results/balanced/hparam

# 7. Report (after stability + single_shot_best + canary + ablation)
sbatch scripts/slurm_report.sh results/balanced
```

### Order-Stability Runs (P3) — parallel seeds

Order-stability: same pretrained model, different forget ORDERINGS per seed,
μ±σ across seeds.  Each seed is an independent job — run them in parallel:

```bash
OUT=results/balanced/iterative_order
for SEED in 42 43 44 45 46; do
    sbatch --job-name=unlearn_ord_$SEED \
        --time=48:00:00 --partition=gpu --gres=gpu:1 \
        --cpus-per-task=8 --mem=32G \
        --wrap="python src/iterative.py --csv $CSV --model $MODEL \
                 --out $OUT/seed_$SEED --seed $SEED --n_seeds 1 \
                 --order_seed $SEED --subset holdout \
                 --best_configs results/balanced/hparam"
done
# Then aggregate: python src/iterative.py --aggregate-only ... (or
# re-run main() with n_seeds=5 pointing at $OUT — it aggregates seed_*/)
```

### Smoke Testing (Before Full Run)

```bash
# End-to-end v1.1 smoke proof (synthetic 6-identity dataset, CPU-only, ~10 min)
sbatch scripts/slurm_smoke.sh

# Or run interactively from the repo root
python tests/smoke_test.py
```

### Method Feasibility Gate (triage — before the expensive full run)

Multi-scale gate: subsamples real identities (12-id fast-fail, or 100-id ≈
13% of full for the head-width column), trains a tiny model, and sweeps each
method's budget at 1×/3×/10× through the real method registries, returning
GO / TUNE / TUNE(retain-collapse) / BROKEN verdicts per method.  Use it to
decide whether a method is healthy at scale (GO), needs config work
(TUNE), or is structurally broken (BROKEN — remove/redesign) *before*
spending the full-run GPU budget.  Locally on Apple Silicon:

```bash
UNLEARN_NUM_WORKERS=0 PYTHONPATH="" python3 -u src/feasibility_study.py \
    --src_csv /path/to/metadata/dataset.csv \
    --out results/feasibility_balanced_12id \
    --epochs 3 --step_scale 1.0 --device mps --scale 12id
# 100-id column (90 retain + 10 forget): add --n_retain 90 --n_forget 10 --scale 100id
```

On Aire (GPU, full budgets, ~4 h per scale) — **also wired into the full
pipeline** (Stage 2c, runs in parallel with single-shot/HP; the report
renders the multi-scale trajectory table):

```bash
sbatch scripts/slurm_feasibility.sh balanced 12id
sbatch scripts/slurm_feasibility.sh balanced 100id
# → results/feasibility_balanced_<scale>/feasibility_results.json (rows + verdicts)
```

Verdicts (v2): GO = forgets+retains at some budget; TUNE = responds to
budget, needs config work; TUNE(retain-collapse) = forgets but kills
retain; BROKEN = no response even at 10× (remove/redesign).  The 750-id
column of the trajectory table comes from the hparam grids.  `--step_scale`
shrinks base budgets for CPU checks; `--methods` restricts the sweep.

## Resource Estimates

| Stage                               | GPU    | CPUs | Memory | Wall time (full)    | Wall time (smoke) |
| ----------------------------------- | ------ | ---- | ------ | ------------------- | ----------------- |
| Train                               | 1 L40S | 8    | 32 GB  | 4-6 hr              | 30 min            |
| Single-shot                         | 1 L40S | 8    | 32 GB  | 12-24 hr            | 1 hr              |
| HP search (per method)              | 1 L40S | 8    | 32 GB  | 8-12 hr             | 1 hr              |
| Single-shot best                    | 1 L40S | 8    | 32 GB  | 12-24 hr            | 1 hr              |
| Iterative                           | 1 L40S | 8    | 32 GB  | 24-48 hr            | 2 hr              |
| Stability plots                     | CPU    | 4    | 16 GB  | 30 min              | 5 min             |
| Ablation                            | 1 L40S | 8    | 32 GB  | 2-4 hr              | 15 min            |
| Canary                              | 1 L40S | 8    | 32 GB  | 4-6 hr              | 20 min            |
| Per-bin oracles (×3)               | 1 L40S | 8    | 32 GB  | 12-18 hr            | —                |
| Budget sweep                        | 1 L40S | 8    | 32 GB  | 8-12 hr             | —                |
| **Total pipeline (balanced)** | —     | —   | —     | **~48-72 hr** | **~4 hr**   |

Wall-clock note: with 8 parallel HP jobs + single-shot sharing the GPU
partition, the HP stage's wall time is one method's search (~12 hr), not
8× — this is the parallelisation win.

## Parallelisation Strategy

```
                     ┌─ single_shot (default) ────────────────┐
train ───────────────┤                                        ├─ single_shot_best
                     └─ hparam ×9 methods (ALL parallel) ────┘      │
                                                              ├─ iterative ── stability
                                                              └─ canary (parallel)
```

- **HP search: one Slurm job per method, all in parallel** — each writes its
  own `{method}_best_config.json`, so jobs never race on a shared file.
  Default set: ng_plus msg ct msg_kd adaptiforget ga srl ft budget_scaled (methods with
  grids in `hparam_search.GRIDS`; no_unlearning/retrain have no tunable
  params).  Override with `HP_METHODS`.
- **Single-shot and HP run concurrently** after train (both depend only on
  the trained model).
- **Ablation runs concurrently too** — it reuses the train checkpoint
  (`afterok` on train, no retrain), so it costs only 1 extra GPU slot and
  ~2-4 hr, parallel with single-shot/HP; the report waits on it.
- **`single_shot_best` runs after all HP jobs** — tuned configs override
  YAML defaults (`config_loader` merges; the log states
  `[config] DEFAULT` vs `[config] OPTIMIZED` per run).
- **Uniform and Poisson iterative are independent experiments** — submit
  both; they write to separate dirs (`iterative` vs `iterative_poisson`).
- **Order-stability seeds are independent jobs** — run 5 seeds in parallel,
  then aggregate μ±σ (see above).
- **Canary has its own train but waits on HP** — it trains a canary-tagged
  model independently of the main train, but its UNLEARNING stage uses the
  tuned GA/AdaptiForget configs, so in the pipeline it runs after the HP
  jobs (`afterok:HP_DEPS`).
- **Imbalanced Protocols B + C** — per-bin oracles (3 retrains in one job)
  and Protocol C selection run in parallel after single_shot_best; the
  budget sweep needs both.

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

### macOS local verification (dev machine)

- Use `PYTHONPATH="" /opt/anaconda3/envs/ml_spec/bin/python3` — the Hermes
  venv leaks site-packages otherwise.
- The smoke test is slow on CPU (~10 min) but completes — don't kill it early.
- DataLoader `num_workers>0` with two concurrent loaders can intermittently
  deadlock under macOS spawn; this is macOS-only, Linux/fork HPC is unaffected.

## Aire-Specific Notes

- **Partition**: `gpu` (28 nodes × 3 L40S GPUs, 168 cores each)
- **Flash storage**: `$TMP_SHARED` (1 TB/job, auto-purged). Stage datasets
  there for I/O-heavy phases.
- **Lustre scratch**: `/mnt/scratch/$USER/` — large capacity, slower I/O.
  The `core/` project tree (code + bench + results) lives here.
- **Home**: `$HOME` — small quota, versioned backups. Keep only dotfiles
  and small configs; the repo lives on scratch (code is backed up via git
  remote; results are regenerable).
- **Max wall time**: 48 hr on gpu partition.
- **Interactive**: `srun --partition=gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=4:00:00 --pty bash`