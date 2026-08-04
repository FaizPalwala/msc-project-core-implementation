# Machine Unlearning Methods & Evaluation Framework
## Identity-Level Unlearning on SFHQ-InstantID

Dual-head ResNet-18 framework for evaluating machine unlearning methods on synthetic
facial identity data. Models a realistic GDPR "right to be forgotten" scenario:
600 synthetic identity clusters, 15 sequential deletion steps, 4 identities per step.

---

## Task

**Primary**: 600-class identity classification (MUFAC-aligned, Choi & Na 2023).
**Secondary**: 4-class age-group classification (Handover-aligned, from proxy labels).
**Unlearning target**: The model must not recognise the deleted person (identity head)
nor their age group (age head) — a dual test of identity-level and attribute-level
forgetting.

### Architecture

```
ResNet-18 backbone (ImageNet-pretrained, 224×224)
  → AdaptiveAvgPool2d → 512-d features
  → fc_identity : 512 → 600   (primary task)
  → fc_age      : 512 →   4   (secondary task)
```

**Training**: Joint loss `L = L_id + λ · L_age` (default λ = 0.5). AMP on CUDA.

### Dataset

| Property | Balanced | Imbalanced |
|----------|----------|------------|
| Identities | 600 | 600 |
| Images/ID | 75 | 85:40:20 gradient |
| Total images | 45,000 | ~19,500 |
| Split | 450 retain / 90 test / 60 forget | Same IDs, pruned |
| Forget protocol | 15 steps × 4 IDs | Same |

### Schema 

Balanced — `dataset.csv` / `dataset.parquet` (11 columns):

| Column | Type | Notes |
|--------|------|-------|
| `image_path` | string | relative path (resolved against CSV dir, then data root) |
| `identity_id` | int | primary key (renamed from `clusterid`) |
| `age_group` | int (0–3) | age-label for the age head |
| `age` | int | raw age estimate |
| `gender` | int (0/1) | confound control / fairness analysis |
| `split` | string | `retain` / `test` / `forget` |
| `forget_step` | int | −1 for non-forget rows |
| `forget_variant` | int | −1 for non-forget rows |
| `arcface_similarity` | float | within-identity outlier confound control |
| `laplacian_variance` | float | image-quality confound |
| `detection_confidence` | float | alignment quality |

Imbalanced — `dataset_imbalanced.csv` / `.parquet`: same 11 + `popularity_bin`
(string) + `images_per_identity` (int) = 13 columns.

File format (CSV or Parquet) is auto-detected from the extension.

---

## Pipeline Stages

```
train → single_shot → iterative → stability ──┐
  │          │                                 │
  │          └→ hparam (parallel) → iterative  │
  │                                            │
  └──→ canary (independent) ───────────────────┴──→ report
```

| Stage | Script | Description | Parallel? |
|-------|--------|-------------|-----------|
| Train | `train.py` | Train original model M on retain+forget | — |
| Single-shot | `single_shot.py` | All methods, all 60 forget IDs, complete eval | After train |
| HP search | `hparam_search.py` | Grid/random search, UF-score ranking | Parallel with single-shot |
| Iterative | `iterative.py` | 15 step × 4 ID, cumulative + fresh, re-emergence | After best configs |
| Stability | `stability.py` | 13 publication-quality plots | After iterative |
| Canary | `canary.py` | Pixel-level ground-truth deletion proof | Independent |
| Report | `report.py` | LaTeX/Markdown synthesis of all results | After stability + canary |

## Quick Start

```bash
# Install dependencies (on Aire: install torch cu124 wheels first, see HPC CookBook)
pip install -e .

# Train original model
python src/train.py --csv ../bench/metadata/dataset.csv --save_dir results/checkpoints

# Single-shot evaluation (smoke test)
python src/single_shot.py --csv ../bench/metadata/dataset.csv \
    --model results/checkpoints/original_model_best.pt \
    --scale 0.1 --skip_retrain --methods ga ft ng_plus

# Full single-shot
python src/single_shot.py --csv ../bench/metadata/dataset.csv \
    --model results/checkpoints/original_model_best.pt

# Iterative protocol (15 steps)
python src/iterative.py --csv ../bench/metadata/dataset.csv \
    --model results/checkpoints/original_model_best.pt \
    --methods ng_plus msg_kd adaptiformet

# Stability plots
python src/stability.py --combined results/iterative/iterative_combined_aggregated.csv
```

## Methods

| Method | Registry key | Type | Reference |
|--------|-------------|------|-----------|
| No-Unlearning | `no_unlearning` | Control | — |
| Retrain Oracle | `retrain` | Gold standard | — |
| Gradient Ascent | `ga` | Baseline | — |
| Random Relabelling | `srl` | Baseline | — |
| Fine-Tune Retain | `ft` | Baseline | — |
| NG+ | `ng_plus` | SOTA | Cadet et al. (2024) |
| MSG | `msg` | SOTA | Cadet et al. (2024) |
| CT | `ct` | SOTA | Cadet et al. (2024) |
| MSG-KD | `msg_kd` | Novel | — |
| AdaptiForget | `adaptiformet` | Novel | — |

## Evaluation Framework

Five-tier evaluation referenced to:

- **Cadet et al. (2024)** — Deep Unlearn benchmark (NG+, MSG, CT specifications)
- **Choi & Na (2023)** — MUFAC identity-level unlearning benchmark
- **Golatkar et al. (2020)** — Representation-level forgetting, re-emergence
- **Grimes et al. (2024)** — Iterative protocol critique
- **Carlini et al. (2022)** — LiRA (approximated via iterative checkpoints)

| Tier | Test | What it catches | Reference |
|------|------|-----------------|-----------|
| 0 | Threshold MIA (confidence + loss) | Output-level privacy | Yeom et al. (2018) |
| 1 | Per-identity MIA + max-confidence | Single-ID leakage, cluster averaging | Choi & Na (2023) |
| 2 | Linear probes (identity/age/gender) | Latent identity encoding | Golatkar et al. (2020) |
| 3 | Pseudo-LiRA (iterative checkpoints) | Stronger attack sensitivity | Carlini et al. (2022) |
| 4 | Canary insertion | Ground-truth deletion proof | Thudi et al. (2022) |
| 5 | Temporal re-emergence | Memory recovery after later updates | Golatkar et al. (2020) |

## Metrics

| Metric | Head | Target | Reported |
|--------|------|--------|----------|
| Identity accuracy (retain/test) | Identity | High, stable | Per-split, per-step |
| Age accuracy (retain/test) | Age | High, stable | Per-split, per-step |
| MIA AUC (mean ± std per identity) | Identity | 0.50 ± 0.05 | Per-method, per-step |
| Max per-identity MIA AUC | Identity | < 0.55 | Per-method |
| Max single-image confidence | Identity | Low | Per-method |
| Fraction leaked (AUC > 0.55) | Identity | 0.00 | Per-method, per-step |
| Identity probe accuracy | 512-d features | ≈ chance (0.17%) | Per-method |
| Age probe accuracy | 512-d features | Comparison metric | Per-method |
| Gender probe accuracy | 512-d features | Fairness metric | Per-method |
| Model drift (L2) | Backbone | Controlled growth | Per-step |
| Step time | — | Informational | Per-step |
| Re-emergence rate | Identity | 0.00 | Cumulative checkpoints |
| MIA AUC by age group | Identity | Uniform across groups | Per-method |
| MIA AUC by popularity bin | Identity | Uniform across bins | Per-method (imbalanced) |

## Config Structure

```
configs/
  methods/*.yaml           → per-method hyperparameters (single source of truth)
```

- Method HPs loaded via `config_loader.load_method_configs(scale=...)`
- Run-scale knobs (`--scale`, `--seed`, `--n_seeds`) are explicit CLI flags on
  each stage script — see the Slurm templates for the canonical full-scale values.

## Acknowledgements

This work was originally undertaken on the Aire HPC system at the University of Leeds, UK.
All experiments (training, unlearning, evaluation) ran on Aire's GPU partition (L40S nodes). 

## References

```
Cadet et al. (2024) — Deep Unlearn: Benchmarking Machine Unlearning. arXiv:2410.01276.
Choi & Na (2023) — MUFAC/MUCAC identity-level unlearning benchmark. arXiv:2311.02240.
Grimes et al. (2024) — Gone but Not Forgotten: Improved Benchmarks. DLS 2024.
Golatkar et al. (2020) — Eternal Sunshine of the Spotless Net. CVPR 2020.
Carlini et al. (2022) — Membership Inference Attacks From First Principles. IEEE S&P.
Thudi et al. (2022) — Unrolling SGD: Understanding Limits of Machine Unlearning.
Yeom et al. (2018) — Privacy Risk in Machine Learning. IEEE CSF.
```
