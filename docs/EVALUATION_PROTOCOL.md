# Evaluation Protocol

**Version:** 1.0 (2026-08) · **Status:** Release artifact (tracked, versioned)
**Applies to:** `msc-project-core-implementation` @ `dev/evaluation-refactor`
**Companion dataset:** `release_v1_bench` v1.1 (750-identity, 15-step protocol)

This document is the formal, pre-registered evaluation protocol for the
identity-level machine-unlearning framework. It specifies, for every metric
reported by the pipeline: what it measures, exactly how it is computed, the
thresholds that were fixed *before* the reported runs, the scientific rationale
for each design choice, and the canonical citation. It is the reference of
record for the results chapters of the dissertation and for reproducing or
extending the experiments.

> **Pre-registration statement.** All thresholds marked **[P]** in this
> document were fixed before the v1.3 full run (jobs 7102706–7102745) and
> before the final run. They are evaluation targets, not post-hoc fits; where
> a method misses a target the result is reported as measured, and the
> interpretation section of each tier explains how to read the miss.

---

## 1. Scope and Evaluation Design

### 1.1 The question

The framework evaluates *identity-level* machine unlearning on synthetic
facial data: a model is trained on 750 synthetic identities; 75 identities are
then "deleted" under a GDPR-style right-to-be-forgotten protocol; an unlearning
method produces a model that (a) no longer recognises the forgotten
identities, (b) retains utility on the 675 remaining identities, (c) does not
re-identify forgotten people via *any* attack surface (output confidence,
latent features, temporal re-acquisition), and (d) does so at lower cost than
the retrain-from-scratch oracle.

### 1.2 Why a five-tier hierarchy

Single metrics are gameable: a method can suppress the classification head
while leaving identity information in the features (output obfuscation), or
forget today's batch while re-learning it from the next update (temporal
re-emergence). The five-tier hierarchy — adapted from the multi-tier framing of
Cadet et al. (2024) and the identity-level benchmark of Choi & Na (2023) —
tests progressively stronger attack surfaces, so that a claim of "erased"
must survive not just the weak output-space attack but also feature-space
probing, checkpoint-level attacks, and causal ground-truth verification.

| Tier | Name | Attack surface | Question | Primary statistic |
|---|---|---|---|---|
| 0 | Threshold MIA | Output confidence (pooled) | Can a *weak* attacker distinguish members? | `retain_test_auc`, `forget_test_auc` |
| 1 | Per-Identity MIA | Identity-level output confidence | Can an attacker target a *specific* person? | per-identity AUC (μ±σ), `mia_fraction_leaked` |
| 2 | Representation probes | 512-d backbone features | Did unlearning reach the latent space? | `probe_identity_acc`, `probe_age_acc`, `probe_gender_acc` |
| 3 | Iterative-checkpoint MIA | Checkpoints over the schedule | Does forgetting survive the unlearning trajectory? | per-step `mia_mean_auc` trend |
| 4 | Canary ground-truth | Injected pixel watermark | Is a *known, inserted* signal causally erased? | membership gap `id_conf(canary)−id_conf(clean)` |
| 5 | Temporal re-emergence | Later retain-only updates | Does forgetting persist after subsequent updates? | `re_emergence_rate` |

### 1.3 Config routing as an experimental control

A central methodological decision, verified end-to-end in the pipeline: **only
the original single-shot evaluation runs with default (untuned) method
configurations.** Every downstream stage — the tuned single-shot, the
iterative protocol, the canary unlearning, the ablation study, and the
imbalanced equity plots — runs with hyperparameters selected by the
UF-ranked grid search. This makes the default-vs-tuned contrast a *controlled
comparison* (Section 5.2): the default-config results isolate each method's
out-of-the-box behaviour, and the tuned results isolate the method's ceiling,
so scale- and configuration-fragility are measured rather than confounded.

### 1.4 Evaluation integrity invariants

1. **Holdout isolation.** Unlearning methods never observe holdout images;
   `prepare_method_call` pins `subset="train"` at the dataset layer. All
   reported forgetting/utility numbers are on held-out images.
2. **Runtime-inferred splits.** Identity classes, forget schedule, and per-step
   sizes are inferred from the CSV (`infer_identity_classes`,
   `infer_forget_schedule`); there are no hardcoded dataset counts in the
   evaluation path.
3. **Multi-seed statistics.** Every single-shot and iterative result is
   reported as μ±σ over 5 seeds. Iterative order-stability uses 5 forget
   orderings over one shared checkpoint.
4. **Fail-closed pipeline.** Every Slurm stage uses `set -euo pipefail`; the
   report waits on every upstream lane and degrades loudly (warns) if an
   artifact is missing, never silently omitting a section.

---

## 2. Tier 0 — Threshold Membership Inference Attack

**What it measures.** Whether a weak attacker who sees only the model's output
can distinguish training members from held-out images, pooled across all
identities. This is the *easiest* test in the hierarchy and the appropriate
baseline for "no privacy at all".

**Method.** Score every image by the model's maximum output confidence on the
identity head. Compute two AUCs against the holdout-confidence distribution
(`src/mia.py:132`):
- `retain_test_auc` — retain-train confidence vs test confidence;
- `forget_test_auc` — forget-train confidence vs test confidence.

| Statistic | Range | Target [P] | Interpretation of miss |
|---|---|---|---|
| `retain_test_auc` | 0.50–1.00 | ≤ 0.55 | > 0.60: members distinguishable — unlearning incomplete |
| `forget_test_auc` | 0.50–1.00 | ≤ 0.55 | > 0.60: forgotten members still confident |
| `forget_advantage` | −0.50–+0.50 | ≈ 0.00 | > +0.10: forgetting lags the test baseline |

**Rationale.** Yeom et al. (2018) showed that overfitting directly translates
into threshold-MIA advantage; a threshold attack is the minimal privacy audit.
Tier 0 pools identities, so it is necessary-but-not-sufficient: a Tier-0 pass
says nothing about *which* identities leak (Tier 1) or *where* the information
lives (Tier 2).

**Citation.** Yeom et al. (2018), "Privacy Risk in Machine Learning," *IEEE
CSF*. DOI: 10.1109/CSF.2018.00027.

---

## 3. Tier 1 — Per-Identity Membership Inference

**What it measures.** Whether an attacker can single out *a specific person*:
"was this identity in the training set?" This is the attack that matters for
identity-level unlearning — if any one forgotten identity remains
distinguishable, the deletion failed for that person.

**Method.** For each of the 75 forget identities, build the identity's
holdout-confidence distribution and compare it to the pooled retain-split
holdout-confidence distribution; report the per-identity AUC. Statistics
computed in `src/mia.py` (`run_mia_per_identity`):
`mia_mean_auc`, `mia_std_auc`, `mia_max_auc`, `mia_min_auc`,
`mia_fraction_leaked` (share of identities with AUC > 0.55), and
`max_conf_auc` (the max-confidence attack, §3.2).

| Statistic | Range | Target [P] | Interpretation |
|---|---|---|---|
| `mia_mean_auc` | 0.50–1.00 | ≤ 0.55 | mean attacker advantage over all identities |
| `mia_std_auc` | 0.00–0.30 | ≤ 0.15 | **the equity metric** — high σ = uneven protection |
| `mia_max_auc` | 0.50–1.00 | ≤ 0.70 | worst-case identity (one person still identifiable) |
| `mia_min_auc` | 0.00–0.50 | < 0.30 | some identities are genuinely invisible |
| `mia_fraction_leaked` | 0.00–1.00 | ≤ 0.30 (≤ 23/75) | share of identities still attackable |

### 3.1 Why mean *and* std

A mean AUC of 0.52 can hide an identity at 0.90 and another at 0.10. The
dissertation reports μ±σ *together* — the std is the equity statistic that
converts "average privacy" into "per-person privacy". This follows Cadet et
al. (2024), whose fraction-leaked criterion is the count of identities above
the 0.55 threshold rather than the average.

### 3.2 Max-confidence attack

`max_conf_auc` uses the single highest confidence *across all output classes*
rather than the identity-head confidence. If the model is confident about
*something* for forget images — even a wrong identity — this attack catches
the residual familiarity that a correctly-suppressed head can hide. A gap
where `max_conf_auc > mia_mean_auc` indicates output-level suppression without
representation-level erasure (see Tier 2).

**Citation.** Carlini et al. (2022), "Membership Inference Attacks From First
Principles," *IEEE S&P*; Cadet et al. (2024), "Deep Unlearn," *arXiv:2410.01276*.

---

## 4. Tier 2 — Representation Probes

**What they measure.** Whether identity information survives in the 512-d
backbone features after unlearning. A method that only suppresses the
classification head performs *output obfuscation*; an attacker with feature
access (e.g., a downstream fine-tuner) can still recover the identity.

**Method.** Extract backbone features for held-out images; train a linear
LogisticRegression probe for each target attribute (identity / age / gender);
report held-out probe accuracy (`src/probes.py`). Chance baselines: identity
≈ 1/750, age = 0.25, gender = 0.50.

| Statistic | Chance | Target [P] | Interpretation |
|---|---|---|---|
| `probe_identity_acc` (retain) | 1/750 | ≥ 0.70 | utility preserved for retained identities |
| `probe_identity_acc` (forget) | 1/750 | ≤ 0.30 | identity features erased from the representation |
| `probe_age_acc` | 0.25 | ≥ 0.70 | shared attributes preserved |
| `probe_gender_acc` | 0.50 | ≥ 0.80 | shared attributes preserved |

**Rationale and the key reviewer check.** If the identity probe on forget
images stays high (≥ 0.70) while the head's forget accuracy drops, the method
performed *feature-level* unlearning failure disguised as success — the
classic "output obfuscation" critique. This probe is therefore the single most
important check that unlearning reached the representation, following Golatkar
et al. (2020), who first formalised representation-level erasure. The age and
gender probes test the *specificity* of unlearning: a method that damages
shared attributes on forget identities is not identity-specific enough.

**Citation.** Golatkar et al. (2020), "Eternal Sunshine of the Spotless Net,"
*CVPR*, DOI: 10.1109/CVPR42600.2020.01254.

---

## 5. Stage A/C — The Single-Shot Comparison

### 5.1 The UF score (ranking metric)

Because no single metric summarises the utility-forgetting-time trade-off, the
framework ranks methods with a composite score, `uf_score` (src/hparam_search.py:176):

```
UF = w_r · retain_acc
   + w_f · max(0, 1 − 2·mia_auc) · [forget_acc ≤ 0.15]   # v2, 2026-08
   − w_t · min(time_s / 300, 1)
```

with **pre-registered weights** `w_r = 0.5`, `w_f = 0.4`, `w_t = 0.1` and time
budget 300 s [P].

**Rationale for the forget term.** Empirically, the retrain oracle (identity
erased) sits at MIA AUC ≈ 0.00 and the no-unlearning control at ≈ 0.50 in this
framework, so the forget term *rewards AUC below 0.5* and zeroes at/above 0.5
(leak). This is the C1-corrected sign convention: it rewards genuine erasure
rather than penalising it. The weights favour utility and forgetting equally
(0.5/0.4) and treat time as a tie-breaker (0.1) — deliberate, because the
methods are primarily compared on *what they forget and keep*, with compute as
a secondary axis (retrain oracle is the cost baseline).

**v2 erasure-conditioned bracket (2026-08, post-v1.3).** The final run exposed
a false optimum: a config with tiny ascent LR collapsed confidence on forget
images → MIA AUC ≈ 0.026 while forget acc stayed ≈ 0.82 and probe-identity
1.0 — *output suppression, not erasure*. The MIA term alone cannot distinguish
"confidence collapse" from "identity erased", so since the correction the
forget term earns credit **only when the config actually erases**
(`forget_acc ≤ 0.15`, matching the pre-registered forget-acc target [P]).
Suppressors earn forget credit exactly 0 and cannot out-rank a genuine eraser.
`src/report.py` `_uf_score` applies the same bracket, so the tuned-table UF
column is consistent with the search that selected the configs. (The hard
rejection layer is Section 6.1.)

### 5.2 Default vs tuned comparison

- **Stage A (defaults):** every method at its YAML-documented default
  configuration — the out-of-the-box behaviour a practitioner would get.
- **Stage C (tuned):** every method at its UF-best configuration from the grid
  search (Section 6) — the method's ceiling.

The contrast is the *scale- and config-fragility* result: e.g., MSG retains
0.0142 at defaults but 0.9983 tuned; GA/NG+ erase nothing (forget 0.9993) at
defaults at 750 identities. These are measured properties of the methods, not
implementation artefacts — and they motivate the feasibility gate (Section 7)
and the budget-scaled variant.

**Citation for the fragility framing.** Choi & Na (2023), *arXiv:2311.02240*
(default-vs-tuned protocol); Grimes et al. (2024), "Gone but Not Forgotten,"
*DLS* (scale-dependence critique).

---

## 6. Hyperparameter Search

**Method.** One grid/random search per method (`src/hparam_search.py`), each
trial evaluated on holdout images with the UF score; best config saved as
`{method}_best_config.json`. The v1.3 run executed 507 trials total
(192 + 36 + 72 + 9 + 9 + 36 + 36 + 108 + 9 across the 9 methods).

**Grid design.** Budgets are the primary axis (step/epoch counts), plus
method-specific knobs:
- `budget_scaled`: `base_steps ∈ {150, 300, 600}`, `budget_exponent ∈ {0.25, 0.5, 0.75}`,
  `reference_forget_count ∈ {75, 150}` — the exponent sweep is the novel
  variant's calibration axis;
- `adaptiforget`: `max_steps`, `kl_weight_init/max`, `mask_refresh_every`,
  `topk_fraction`, `early_stop_adv`.

**Rationale.** Fixed step budgets do not transfer across dataset sizes
(Section 7) — the feasibility gate's central finding — so budget scaling is
searched explicitly rather than assumed. Each method's grid is recorded in
`GRIDS` and the per-trial JSONL is append-only (never re-read by the pipeline,
by design — a poisoned or partial JSONL cannot corrupt later stages).

### 6.1 Erasure gate (hard rejection layer, 2026-08)

**Purpose.** Prevent the search from exporting *suppression* configs as
"best". The final run's AdaptiForget tuned config (lr_ascent 1e-5, 10× below
default) collapsed confidence on forget images → MIA AUC ≈ 0.026 (UF read it
as erased) while forget acc stayed ≈ 0.82 and probe-identity 1.0 — the model
was *unsure*, not *unlearned*. The v2 UF bracket (5.1) fixes the score; the
gate is the complementary hard layer:

- `ERASURE_FORGET_ACC_MAX = 0.15`, `ERASURE_PROBE_ACC_MAX = 0.30` [P] —
  aligned with the pre-registered forget-acc target and the probe semantics
  (features still reading the forget split ⇒ not erased).
- Each trial's `forget_id_acc` (and identity-probe accuracy on the forget
  split, `probe_identity_forget_acc`) is checked after evaluation. Trials
  with `forget_acc > 0.15` or `probe > 0.30` are **REJECTED**: they remain in
  the append-only JSONL (audit trail, tagged `"erasure_gate": "REJECTED"`)
  but are excluded from ranking and best-config export.
- If **no** trial passes, any stale `{method}_best_config.json` from an
  earlier run is deleted and the method runs at its YAML default downstream
  (logged loudly; a rejected config can never be resurrected by
  `config_loader`'s glob). This is the documented fallback for methods that
  cannot erase at full scale (e.g. ng_plus at 750 identities).

**Why gate + v2 bracket.** The bracket makes the *score* suppression-proof;
the gate makes the *selection* suppression-proof (a suppressor can never be
exported even if retain/time dominate its UF). Both thresholds are
pre-registered targets, so the guard is a protocol control, not a
post-hoc patch.

---

## 7. Feasibility Gate (C2/C3)

**Purpose.** A cheap, small-scale pre-flight that answers, before a full run:
is a method's failure at full scale a *configuration* problem (it responds to
more budget) or *structural* (no response even at 10×)? Runs as pipeline
Stage 2c on a ~12-identity subsample, sweeping each method's budget at 1×/3×/10×
(`src/feasibility_study.py`, `scripts/slurm_feasibility.sh`).

**Verdict semantics.** GO = forgets and retains at some budget; TUNE = responds
to budget but needs config work; TUNE(retain-collapse) = forgets but destroys
retain; BROKEN = no response even at 10× (provisional until confirmed at full
scale).

**Central finding.** Fixed step budgets do not transfer across dataset sizes —
the direct motivation for the Budget-Scaled GA variant (`budget_scaled`), whose
step budget scales with forget-set size:
`steps = clip(base · (n_forget / reference)^exponent)`.

**The worked example of a verdict being overturned.** The gate marked
fine-tuning (FT) BROKEN at 12 identities (it did not respond to budget there);
at full 750-identity scale FT forgets perfectly (forget accuracy 0.0000). The
verdict was a *scale artefact* — BROKEN verdicts are explicitly provisional.
The gate also caught a real bug: AdaptiForget's first gate pass returned BROKEN
(pre-fix no-op); the early-stop fix flipped it to GO. The gate is thus both a
triage tool and a bug-catcher, and its verdicts are recorded per run in
`feasibility_results.json` and rendered by the report.

**Citation.** Tarun et al. (2023), "Class-Specific Unlearning," *CVPR* (budget
scaling intuition); Bourtoule et al. (2021), "Machine Unlearning," *IEEE S&P*
(retrain oracle as the cost baseline).

---

## 8. Iterative Protocol and Temporal Re-Emergence (Tier 3 + Tier 5)

**What it measures.** Forgetting in the *streaming* regime: identities are
forgotten in 15 sequential steps (5 identities each), and the model is updated
at each step. Tier 3 tracks per-identity MIA at each checkpoint along the
schedule; Tier 5 re-tests identities forgotten in *earlier* steps at later
steps (5, 10, 15) to detect re-acquisition.

**Step-local vs cumulative.** Cumulative `forget_acc`/`mia_mean_auc` at step k
aggregate over all identities forgotten so far — a method that fails on step
7's identities can hide behind the improving average. The step-local columns
(`step_forget_acc`, `step_forget_train_acc`) report only the *current* step's
identities, so a failing step cannot hide.

| Metric | Scope | Target [P] | Interpretation |
|---|---|---|---|
| `forget_acc` | cumulative | ≤ 0.15 | trajectory |
| `step_forget_acc` | this step, holdout | ≤ 0.15 | can't hide a failing step |
| `step_forget_train_acc − step_forget_acc` | gap | ≤ 0.10 | genuine erasure; ≥ 0.15 = overfit to the unlearning signal |
| `mia_mean_auc` per step | cumulative | → 0.50 | approach indistinguishability |
| `re_emergence_rate` | earlier identities at later steps | ≤ 0.05 | durable forgetting; ≥ 0.10 = temporary |

**Rationale.** The forget-train/forget-holdout gap (Grimes et al. 2024) is the
overfitting-to-forgetting detector: if forget accuracy is 0.00 on training
images but 0.30 on holdout, the method memorised the 75 unlearning images
rather than erasing the identity concept. Re-emergence (Golatkar et al. 2020;
Grimes et al. 2024) is the durability check — a method that forgets today but
re-learns the identity from the next retain-only update fails the GDPR
"erasure is permanent" requirement.

**Schedules.** Two parallel lanes:
- **Uniform** — 15 steps × 5 identities; supports multi-seed μ±σ and
  order-stability (Section 9).
- **Seeded Poisson (λ=5)** — a single fixed, seeded schedule with variable
  batch sizes (2–9 identities), modelling GDPR erasure requests as an
  independent stochastic process (Online Forgetting Process, *arXiv:2012.01668*).
  One-shot stress test by design: re-ordering it would defeat its purpose.

**Analysis rule (cross-schedule/cross-bin).** All comparisons use
*cumulative forgotten count* as the x-axis, never raw step index — streaming
difficulty scales with the total variation V_T between consecutive forget sets
(Shen et al. 2025, *arXiv:2507.15280*).

---

## 9. Order-Stability Protocol (P3/P4)

**Question.** Does *which* identities are forgotten at *which* step matter,
holding the pretrained model fixed?

**Protocol.** One shared pretrained checkpoint × 5 forget orderings
(`--order_seed` per run, deterministic permutation of the uniform schedule);
report μ±σ across orderings. The retrain oracle is order-independent (it
never touches forget order) and runs once.

**Reading the result.** Tight μ±σ → behaviour is driven by model/identity
properties, not schedule luck. Wide μ±σ → order-sensitive forgetting — the
streaming regime Shen et al. (2025) warns about; report the spread, never
average it away. The step-local gap (§8) is the per-ordering diagnostic: an
order-stable method keeps gap ≤ 0.10 at every step regardless of arrival
order. Poisson + `--order_seed` raises by design.

---

## 10. Tier 4 — Canary Ground-Truth Verification

**What it measures.** *Causal* evidence of deletion: a known, identity-specific
pixel pattern is inserted into all images of 4 forget identities *before*
training; after unlearning, the verifier checks whether the model still uses
the pattern as a membership cue.

**Canary design.** A 32×32 pattern repeated on a **3×3 grid** (9 placements,
`src/canary.py` `_grid_positions`) at magnitude ±60 per channel. The
multi-location grid is load-bearing: random-crop augmentation destroys a
single-corner pattern (the original design was never learned — positive
control det-acc 0.478 ≈ chance); a grid survives any crop, and the positive
control confirms the pattern is encoded (4/4 identities, det-acc 1.0).

**The erasure statistic — membership gap.** The headline signal is the
output-space membership gap:
`gap = id_conf(canary-tagged) − id_conf(clean)`,
where "clean" is the same identity's *original* images (recovered by
stripping the `canary_images/` path segment). A gap ≈ 0 means both groups are
treated identically — the canary identity is erased. A gap ≫ 0 means the model
still scores canary-tagged images as the identity — the pattern remains a
membership cue.

**Why not feature-space detection.** `canary_detection_acc` (LR detector) is
*not* the headline: the pattern is physically present in the pixels, so a
feature-space detector finds it ≈1.0 both before AND after unlearning. It
measures presence, not erasure. Empirically (v1.3): AdaptiForget closes the
gap to 0.000 on all 4 identities; GA leaves gaps of 0.21–0.45 (pattern
persists, consistent with GA's weak forgetting).

**Rationale.** This is the strongest available privacy guarantee: if the
inserted signal is gone, deletion is *verified*, not merely inferred. It is
the ground-truth complement to the correlational MIA tiers (Thudi et al. 2022),
and it motivated the report's Tier-4 rendering (membership-gap table, not
det-acc).

**Citation.** Thudi et al. (2022), "Unrolling SGD: Understanding Limits of
Machine Unlearning," *arXiv:2109.10672*.

---

## 11. AdaptiForget Ablation

**Purpose.** Attribute the novel method's behaviour to its components. Each of
6 variants disables exactly one component on identical data (same trained
model, same seed, same forget set); the "Full" variant uses the **tuned**
best-config (the method as deployed), so contributions are measured from the
deployed baseline, not the default one (`src/ablation_study.py`, pipeline
Stage 2a).

| Variant | Override | Component removed |
|---|---|---|
| Full AdaptiForget | — | none (baseline) |
| w/o adaptive λ | `kl_weight_init/max = 0.5` | KL weight schedule → constant |
| w/o mask refresh | `mask_refresh_every = 999999` | periodic saliency recompute → one-shot mask |
| w/o early stop | `early_stop_adv = -1.0` | forget-holdout early stop → full budget |
| w/o KL distil | `kl_weight = 0.0` | KL anchor → pure MSG |
| w/o masking (NG+) | `topk_fraction = 1.0` | saliency mask → all parameters updated |

**Headline finding.** Early stop is the load-bearing component: removing it
raises MIA from 0.0255 to 0.5139 and leaks 52% of identities at full scale —
a result that holds at both the 12-identity gate and 750 identities.

**Rationale.** Component ablations (Golatkar et al. 2020's "ablations of
unlearning"), not just whole-method comparisons, are what turn a novel method
into an *explained* method: each variant's drop in forget advantage / retain
accuracy quantifies that component's contribution to the deployed behaviour.

---

## 12. Imbalanced Stress Protocol (H1 + Protocols B/C)

The imbalanced dataset (750 identities; popularity bins **75 high / 225
medium / 450 low**, 5:1 train-image gradient, single-shot axis — no time
schedule) tests the long-tail hypothesis:

**H1 (FaLW-grounded).** High-popularity identities are over-learned (82 train
images) → hardest to forget; low-popularity (16 images) → easiest to scrub.

| Protocol | What it does | Where it lands |
|---|---|---|
| A. Baseline gate | per-bin MIA AUC on the *original* model — a bin at ≈0.5 was never learned, not scrubbed | `single_shot` per-bin block |
| B. Distance-to-oracle | **one oracle per bin** — retrain excluding only that bin's forget identities; per-bin behavioural distance after a fixed budget | `per_bin_oracle.py` → `oracles/` + `per_bin_distance.json` |
| C. Budget sweep | sweep unlearning budget per method until each bin reaches its oracle = per-bin difficulty; high-bin needing more → H1 supported | `select_protocol_c_methods.py` (1 best method per category by UF) → `budget_sweep.py` |
| D. Balanced cross-check | same 75 forget IDs through the balanced artifact (90 images/ID) — hard-in-both ⇒ identity-tracks; easier in balanced ⇒ count-tracks | shared split map |
| E. Intensity modulation | fixed-intensity baseline vs bins; expect FaLW skewed deviation | report |

On the imbalanced axis "budget" = method-internal iterations (`ga_steps`,
`ft_epochs`, `srl_epochs`, `ng_steps`, `msg_steps`, `ct_steps`, `max_steps`,
`base_steps` — the `BUDGET_KEY` map in `src/budget_sweep.py`).

**Equity statistics.** Per-bin MIA AUC (mean per bin) and Kruskal-Wallis
H-test on per-identity AUC by bin (fail to reject p > 0.05 for retain
metrics = equity-preserving; report p for forget metrics). Six equity plots
from `src/imbalanced_plots.py`, including AUC-vs-images-per-ID regression
(data-poverty → privacy-poverty) and the utility-forgetting frontier by bin.

**Rationale.** Long-tailed identity distributions are the realistic GDPR
setting (Cao et al. 2018; Liu et al. 2019); H1 follows the FaLW skewed-
unlearning deviation result (Yu et al. 2026, *arXiv:2601.18650*); per-bin
oracles make "difficulty" a *measured* quantity per popularity stratum
rather than a global average. Empirically (v1.3): FT is high-bin-hardest
(forget 0.911 high vs 0.066 low — H1 confirmed for saliency-agnostic
methods), while saliency-mask methods invert (high-bin easiest).

---

## 13. Reporting and the Artifact Trail

**Report** (`src/report.py`) synthesises: tuned single-shot table (headline) +
default-config table (evidence), feasibility verdict matrix, iterative tables
per schedule, canary membership-gap table, and the ablation table — all from
JSON/CSV artifacts, never recomputed.

**Stability plots** (`src/stability.py`, 16 plots): retain-acc, MIA-AUC,
forget-advantage, model drift, step time, Pareto frontier + hypervolume,
heatmap, radar, cumulative time, per-identity signatures, demographic
heatmap, phase space, fraction leaked, total-time bar, forget-train/holdout
gap by step, step-local forget acc.

**Reproducibility invariants.**
1. Every number in the report is traceable to a JSON/CSV artifact in
   `results/<dataset>/`.
2. `checksums.sha256` + `RELEASE_MANIFEST.json` pin the dataset release;
   `slurm_dataset_helper.sh` resolves the CSV by construction
   (`$(dirname repo)/bench/metadata/*.csv`).
3. Seeds are recorded per run; the Poisson schedule is fixed and seeded
   (recorded in the manifest) by design.

---

## References

- Yeom et al. (2018). "Privacy Risk in Machine Learning." *IEEE CSF*.
  DOI: 10.1109/CSF.2018.00027. — Threshold MIA (Tier 0).
- Choi & Na (2023). "MUFAC/MUCAC: Identity-Level Unlearning Benchmark."
  *arXiv:2311.02240*. — Per-identity MIA, identity probe, default-vs-tuned.
- Cadet et al. (2024). "Deep Unlearn: Benchmarking Machine Unlearning."
  *arXiv:2410.01276*. — Multi-tier evaluation, fraction-leaked.
- Golatkar et al. (2020). "Eternal Sunshine of the Spotless Net." *CVPR*.
  — Representation probes, re-emergence, ablations.
- Grimes et al. (2024). "Gone but Not Forgotten: Improved Benchmarks for
  Machine Unlearning." *DLS*. — Iterative-protocol critique, forget-train gap.
- Carlini et al. (2022). "Membership Inference Attacks From First Principles."
  *IEEE S&P*. — LiRA, max-confidence attack.
- Thudi et al. (2022). "Unrolling SGD: Understanding Limits of Machine
  Unlearning." *arXiv:2109.10672*. — Canary ground-truth verification.
- Bourtoule et al. (2021). "Machine Unlearning." *IEEE S&P*. — Retrain oracle.
- Tarun et al. (2023). "Class-Specific Unlearning for Efficient Removal of
  Object Classes." *CVPR*. — Budget/scale intuition.
- Shen et al. (2025). "Machine Unlearning for Streaming Forgetting."
  *arXiv:2507.15280*. — Cumulative-count axis, V_T rule.
- Online Forgetting Process. *arXiv:2012.01668*. — Poisson arrival model.
- Yu et al. (2026). "Forgetting-aware Loss Reweighting for Long-tailed
  Unlearning" (FaLW). *arXiv:2601.18650*. — Long-tail H1.
- Cao et al. (2018). "VGG-Face2: A Dataset for Recognising Faces across Pose
  and Age." *FG*. — 5:1 popularity gradient calibration.
- Liu et al. (2019). "Large-Scale Long-Tailed Recognition." *CVPR*. — Long-tail
  justification.
- GENIU (class-level imbalance unlearning), *arXiv:2406.07885*; CUFG (ordered
  vs random forgetting), *arXiv:2509.14633* — future schedule variants.
