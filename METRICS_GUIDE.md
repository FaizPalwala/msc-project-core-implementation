# Machine Unlearning Evaluation Guide

**Companion document for results interpretation — dissertation reference.**  
**Status:** Living document — untracked, expandable as experiments progress.

---

## 1. How to Use This Guide

- Read it **alongside** your aggregated results (`single_shot_aggregated.json`,
  `iterative_combined_aggregated.csv`, stability plots).
- Every metric has a row: *what it measures*, *how it's computed*, *what "good"
  looks like*, *what "bad" looks like*, and the *canonical citation*.
- The **Quick‑Reference Table** (§9) is designed to be pasted into your
  dissertation's methods section or appendix.
- The **Per‑Method Expectations** (§8) tells you what each method *should*
  produce if the implementation is correct — use it to sanity‑check runs.

---

## 2. Evaluation Tiers (overview)

Our framework follows a five‑tier evaluation hierarchy adapted from Cadet et al.
(2024), Choi & Na (2023), and Grimes et al. (2024):

| Tier | Name | Key Question | Primary Metric | Attack Surface |
|---|---|---|---|---|
| 0 | Threshold MIA | Can a weak attacker distinguish members? | AUC (retain vs forget) | Output confidence |
| 1 | Per‑Identity MIA | Can an attacker target a *specific* identity? | Per‑ID AUC (μ ± σ), fraction leaked | Identity‑level confidence |
| 2 | Representation Probes | Did unlearning reach the latent space? | Probe accuracy (identity/age/gender) | 512‑d backbone features |
| 3 | Pseudo‑LiRA | Would a strong, reference‑model attack succeed? | AUC across iterative checkpoints | Shadow‑model calibration |
| 4 | Canary Ground‑Truth | Is a known, inserted signal erased? | Canary detection accuracy | Injected pixel watermark |
| 5 | Temporal Re‑emergence | Does forgetting survive later updates? | Forget‑holdout AUC at later checkpoints | Decay over training steps |

---

## 3. Tier 0 — Threshold Membership Inference Attack (MIA)

**What it measures.**  The most basic privacy test: can an attacker who only sees
the model's output confidence distinguish a randomly‑selected training image
from a held‑out image?

**Method.**  Train a threshold classifier on the model's maximum output
confidence for retain‑split images vs holdout‑split images.  Report AUC.

| Statistic | Expected range | Good | Bad |
|---|---|---|---|
| `retain_test_auc` | 0.50 — 1.00 | ≤ 0.55 | > 0.60 |
| `forget_test_auc` | 0.50 — 1.00 | ≤ 0.55 | > 0.60 |
| `forget_advantage` | −0.50 — +0.50 | ≈ 0.00 (no advantage over test) | > +0.10 (forget still distinguishable) |

**Interpretation.**  Tier 0 is a weak attack — it pools all identities together,
so it's the *easiest* test to pass.  If a method fails Tier 0 (AUC > 0.60), it
almost certainly fails Tier 1.  Conversely, a Tier 0 pass is necessary but not
sufficient.

**Citation.**  Yeom et al. (2018), "Privacy Risk in Machine Learning," *IEEE CSF*.

---

## 4. Tier 1 — Per‑Identity MIA

**What it measures.**  A stronger attack that targets a *single* forget identity
at a time, asking: *"Was this specific person in the training set?"*  This is
the attack most relevant to identity‑level unlearning — if the model still
encodes *any one person* more confidently than random, the unlearning failed for
that person.

**Method.**  For each of the 75 forget identities, train a binary classifier on
that identity's holdout‑confidence vs the retain‑split holdout‑confidence
distribution.  Report per‑identity AUC (mean ± std, max, min), fraction with
AUC > 0.55, and the worst‑case identity's AUC.

### 4.1 Per‑Identity AUC Statistics

| Statistic | Expected range | Good | Bad |
|---|---|---|---|
| `mia_mean_auc` | 0.50 — 1.00 | ≤ 0.55 | > 0.60 |
| `mia_std_auc` | 0.00 — 0.30 | ≤ 0.15 (tight — all identities similar) | > 0.20 (some identities leak much more) |
| `mia_max_auc` | 0.50 — 1.00 | ≤ 0.70 | > 0.80 (at least one identity is clearly identifiable) |
| `mia_min_auc` | 0.00 — 0.50 | < 0.30 (some identities are genuinely invisible) | — |
| `mia_fraction_leaked` | 0.00 — 1.00 | ≤ 0.30 (≤ 23 of 75 identities leaked) | > 0.50 (> 38 identities leaked) |

**Interpretation.**  `mia_std_auc` is the equity metric: a high std means
forgetting is *uneven* — some identities are well‑protected, others are
vulnerable.  The dissertation should report both mean AND std, not just mean.

### 4.2 Max‑Confidence Attack

| Statistic | Expected range | Good | Bad |
|---|---|---|---|
| `max_conf_auc` | 0.50 — 1.00 | ≤ 0.55 | > 0.65 |

**Interpretation.**  A variant of Tier 0 that uses the single highest confidence
*across all output classes* rather than the identity‑head confidence.  If the
model is confident about *something* (even a wrong identity) for forget images,
this attack catches it.  A gap between `max_conf_auc` and `mia_mean_auc` (the
former higher) suggests the model still "recognises" forget images as familiar
even if it misclassifies them.

### 4.3 Forget‑Train vs Forget‑Holdout Gap

**Only available after the per‑identity `image_subset` redesign (§2 of the plan).**

| Statistic | Good | Bad |
|---|---|---|
| `forget_train_id_acc − forget_holdout_id_acc` | ≤ 0.10 | > 0.15 |

**Interpretation.**  If forget accuracy drops to 0.00 on training images but
stays at 0.30 on holdout images, the method **overfit the unlearning signal**
rather than genuinely erasing the identity — it learned to suppress the specific
75 training images but the identity concept survived.  A gap ≤ 0.10 across all
popularity bins is strong evidence of genuine forgetting.

**Citation.**  Choi & Na (2023), "MUFAC/MUCAC," *arXiv:2311.02240*.

---

## 5. Tier 2 — Representation Probes

**What they measure.**  Does the model's internal representation still encode
the forget identity?  If yes, an attacker with access to the features (not just
the output) can recover identity information even when the classification head
has been "unlearned."

**Method.**  Extract 512‑d backbone features for held‑out images of each split.
Train a linear LogisticRegression probe to predict identity / age / gender from
those features.  Report accuracy and the delta vs chance.

### 5.1 Identity Probe

| Statistic | Chance | Good | Bad |
|---|---|---|---|
| `probe_identity_acc` (retain) | 1/n_ids ≈ 0.002 | ≥ 0.70 (model still knows retain identities) | < 0.50 (collateral damage to utility) |
| `probe_identity_acc` (forget) | 1/n_ids ≈ 0.017 | ≤ 0.30 (identity features erased) | ≥ 0.70 (identity still encoded in features) |

**Interpretation.**  **This is the single most important probe.**  If the
identity probe on forget images stays high (≥ 0.70) while the classification
head's forget accuracy drops, the method performed *output‑level obfuscation*,
not true feature‑level unlearning.  A reviewer will flag this immediately.

### 5.2 Age Probe

| Statistic | Chance | Good | Bad |
|---|---|---|---|
| `probe_age_acc` (any split) | 0.25 (4 age groups) | ≥ 0.70 (age knowledge preserved — utility retained) | < 0.50 (age features damaged — collateral harm) |

**Interpretation.**  Age should be equally probeable on retain, forget, and
holdout splits.  If age probe accuracy drops on forget images, the unlearning
method damaged shared attributes — a negative side‑effect.

### 5.3 Gender Probe

| Statistic | Chance | Good | Bad |
|---|---|---|---|
| `probe_gender_acc` (any split) | 0.50 (binary) | ≥ 0.80 (gender preserved) | < 0.60 (fairness concern) |

**Interpretation.**  Same logic as age — gender is a shared attribute that
should survive unlearning.  A drop on forget identities indicates the method is
not identity‑specific enough.

### 5.4 Measure‑Forgetting (Probe Delta)

| Statistic | Good |
|---|---|
| Identity probe Δ (retain − forget) | Large positive (≥ 0.5) — identity features erased on forget, preserved on retain |

**Citation.**  Golatkar et al. (2020), "Eternal Sunshine of the Spotless Net," *CVPR*.

---

## 6. Tier 3 — Pseudo‑LiRA (Iterative Checkpoint MIA)

**What it measures.**  A stronger, reference‑model‑calibrated MIA that
approximates Carlini et al.'s LiRA attack using checkpoints saved at 5‑step
intervals during iterative unlearning.

**Method.**  At each checkpoint step, compute per‑identity AUC using the current
model as the target and the original (pre‑unlearning) model as the reference.
Track the trend over 15 forget steps.

| Trend | Interpretation |
|---|---|
| AUC drops toward 0.50 and stays flat | Genuine forgetting — the model's behaviour on forget identities becomes indistinguishable from unseen identities. |
| AUC drops then rebounds at later steps | Re‑emergence — the model "re‑learns" the forget identity from later retain‑only updates. This is the key Tier 5 signal. |
| AUC never drops below 0.60 | The method never achieves meaningful forgetting for this identity. |

**Citation.**  Carlini et al. (2022), "Membership Inference Attacks From First
Principles," *IEEE S&P*.

---

## 7. Tier 4 — Canary Ground‑Truth

**What it measures.**  Absolute proof that a known signal was erased.  An
identity‑specific pixel canary (a 32×32 pattern repeated on a 3×3 grid —
multi‑location so random crops during training always retain a full copy;
a single‑corner pattern is cropped out and never learned) is inserted into
all images of 4 forget identities before training.  After unlearning, a
verifier compares the model's behaviour on canary‑tagged images vs the
**clean originals** of the same identities (recovered by stripping the
`canary_images/` path segment).

| Statistic | Good | Bad |
|---|---|---|
| Membership gap = `id_conf(canary‑tagged) − id_conf(clean)` | ≈ 0 (both groups treated identically — the canary identity is erased) | ≫ 0 (canary‑tagged images still scored as the identity — the model still uses the pattern as a membership cue) |
| Canary detection (feature‑space LR) | NOT used as the headline — the pattern is physically in the pixels, so a detector always finds it (≈1.0 pre AND post unlearning). Measures presence, not erasure. | — |

**Interpretation.**  This is the strongest possible privacy guarantee: if the
canary is gone, we have *causal* evidence of deletion (not just correlational
MIA evidence).  A Tier 4‑passing method can claim "verified deletion" — a
higher bar than any MIA.  Empirically (v1.3 run): AdaptiForget closes the
membership gap to 0.000 on all 4 canary identities; GA leaves a 0.21–0.45 gap
(pattern persists — consistent with GA's weak forgetting).

**Citation.**  Thudi et al. (2022), "Unrolling SGD: Understanding Limits of
Machine Unlearning," *arXiv*.

---

## 8. Tier 5 — Temporal Re‑Emergence

**What it measures.**  Does forgetting survive later updates to the model?  At
steps 5, 10, and 15 of the iterative protocol, re‑test identities that were
forgotten in *earlier* steps.  If their MIA AUC rises, the model has
re‑acquired information about them.

| Trend | Interpretation |
|---|---|
| Re‑emergence rate ≤ 0.05 | Forgetting is durable — later retain‑only updates do not resurrect forgotten identities. |
| Re‑emergence rate ≥ 0.10 | Forgetting is temporary — the model reconstructs identity information from the retain distribution alone. This is a critical weakness. |

**Citation.**  Golatkar et al. (2020) for the re‑emergence concept; Grimes et
al. (2024) for the iterative‑protocol critique.

---

## 9. Cross‑Cutting Metrics

### 9.1 Model Drift (`model_drift_l2`)

| Statistic | Good |
|---|---|
| L2 distance of backbone parameters from original model | Monitored for growth; small drift (≤ 2.0) indicates the method didn't destroy overall representations. |

### 9.2 Step Time

| Statistic | Meaning |
|---|---|
| `total_time_s` | Total wall time per seed per method. Use to compare compute efficiency: AdaptiForget should be faster than retrain, while MSG/CT may be slower due to saliency computation. |

### 9.3 Statistical Tests (Imbalanced Only)

| Test | Null Hypothesis | Desired Outcome |
|---|---|---|
| Kruskal‑Wallis H‑test on per‑identity MIA AUC by popularity bin | AUC is equal across bins | Fail to reject (p > 0.05) for retain metrics; report p for forget metrics |

---

## 10. Per‑Method Expected Behaviour

| Method | Forget‑holdout acc | Retain‑holdout acc | MIA AUC | Forget‑train gap | Notes |
|---|---|---|---|---|---|
| **no_unlearning** | High (~0.72) | High (~0.70) | ~0.66 (baseline leak) | ≈ 0.00 | Control — everything leaks; the yardstick for all other methods. |
| **retrain** (oracle) | ~0.00 | ~0.99 | ~0.50 (near‑perfect) | ≈ 0.00 | Gold standard. Upper bound on utility, lower bound on leakage. |
| **ga** (Gradient Ascent) | ≤ 0.60 | ≥ 0.70 | 0.55 — 0.65 | ≤ 0.10 | Unlearns by climbing the loss — should reduce forget acc noticeably. If still 0.56, increase `ga_steps` or `ga_lr`. |
| **srl** (Random Relabeling) | ≤ 0.15 | ≥ 0.70 | ≤ 0.50 | ≤ 0.10 | Relabels forget identities → confuses identity head. Should preserve retain well. |
| **ft** (Fine‑Tune Retain) | ≤ 0.05 | ≥ 0.80 | ≤ 0.50 | ≤ 0.05 | Finetunes on retain only — should "drift away" from forget identities. Often the strongest simple baseline. |
| **ct** (Saliency Threshold) | ≤ 0.05 | ≥ 0.70 | ≤ 0.50 | ≤ 0.05 | Dampens high‑saliency weights — similar performance to FT. |
| **ng_plus** (Negative Gradient +) | ≤ 0.60 | ≥ 0.70 | 0.55 — 0.65 | ≤ 0.10 | Ascent + retain descent + KL reg. Should outperform vanilla GA; if not, check KL weight. |
| **msg** (Masked Small Gradients) | ≤ 0.10 | ≥ 0.70 | ≤ 0.55 | ≤ 0.10 | Masks the top‑k saliency gradients — designed to protect retain. If retain collapses (0.17), reduce `topk_fraction` or increase `retain_reg_every`. |
| **msg_kd** (MSG + KD) | ≤ 0.10 | ≥ 0.70 | ≤ 0.55 | ≤ 0.10 | MSG with KL distillation. Same retain‑collapse risk as MSG. |
| **adaptiforget** (AdaptiForget) | ≤ 0.15 | ≥ 0.70 | ≤ 0.55 | ≤ 0.10 | Adaptive mask shrinks during unlearning — should balance forgetting and retention. The primary novel contribution. |
| **budget_scaled** (Budget-Scaled GA) | ≤ 0.60 | ≥ 0.70 | 0.55 — 0.65 | ≤ 0.10 | GA whose step budget adapts to forget-set size (`steps = clip(base·(n_forget/ref)^exp)`). Should forget at scale where fixed-budget GA under-forgets, and protect retain where GA over-erases. |

---

## 11. Quick‑Reference Table

| Metric | What it means | Good value | Primary citation |
|---|---|---|---|
| `retain_id_acc` | Utility: does the model still recognise retained identities? | ≥ 0.65 | Choi & Na (2023) |
| `forget_id_acc` (holdout) | Forgetting: does the model fail to recognise forgotten identities on unseen images? | ≤ 0.15 | Choi & Na (2023) |
| `forget_train_id_acc` | Overfitting‑to‑forgetting check | Gap to holdout ≤ 0.10 | Grimes et al. (2024) |
| `mia_mean_auc` | Per‑identity MIA: can an attacker distinguish forget members? | ≤ 0.55 | Carlini et al. (2022) |
| `mia_fraction_leaked` | How many identities are still identifiable? | ≤ 0.30 | Cadet et al. (2024) |
| `probe_identity_acc` (forget) | Are identity features still in the representation? | ≤ 0.30 | Golatkar et al. (2020) |
| `probe_age_acc` | Is age knowledge preserved? | ≥ 0.70 | — |
| `model_drift_l2` | Has the model drifted substantially? | ≤ 2.0 | — |
| `re_emergence_rate` | Does forgetting survive later updates? | ≤ 0.05 | Golatkar et al. (2020) |
| `canary_membership_gap` | `id_conf(canary‑tagged) − id_conf(clean)` — does the model still use the canary as a membership cue? | ≈ 0.00 (erased) | Thudi et al. (2022) |

---

## 12. Graph Gallery

### 12.1 Stability Plots (16 publication‑quality PNGs)

Generated by `src/stability.py` from the aggregated iterative‑protocol CSV.
Each plot is a multi‑method × multi‑step comparison:

1. **Retain Identity Accuracy over Steps** — Utility trajectory. Flat or
   gently declining = good.
2. **MIA Mean AUC over Steps** — Privacy trajectory. Should approach 0.50.
3. **Forget Advantage over Steps** — |AUC − 0.5|. Should drop toward 0.
4. **Model Drift (L2)** — Semantic drift of backbone parameters.
5. **Step Time** — Per‑method wall time per step.
6. **Pareto Frontier + Hypervolume** — Utility vs forgetting trade‑off with
   hypervolume; the frontier identifies the best methods.
   **The most important single plot for the dissertation.**
7. **Forget Advantage Heatmap** — Method × step, |AUC − 0.5| (↓ better).
8. **Final‑Step Multi‑Metric Radar** — retain acc, step‑forget quality,
   forget quality, MIA quality, speed (all outer = better).
9. **Cumulative Time** — Total wall time vs step.
10. **Per‑Identity Signatures** — Per‑identity MIA AUC bar charts, one panel
    per method (needs `single_shot_per_identity.csv` — now passed by the
    pipeline).
11. **Demographic Heatmap** — MIA AUC by demographic group × method (needs
    `single_shot_demographic.csv` — now passed by the pipeline).
12. **Phase Space** — Retain acc vs forget acc trajectory over steps.
13. **Fraction Leaked** — Fraction of identities with AUC > 0.55 over steps.
14. **Total Time Bar** — End‑of‑run wall time per method (deployment cost).
15. **Forget‑Train / Forget‑Holdout Gap by Step** — The overfitting‑to‑forgetting
    detector.  Step‑local: `step_forget_train_acc − step_forget_acc` per step.
    Gap ≤ 0.10 (green line) = genuine forgetting; gap ≥ 0.15 (red line) = the
    method memorised the unlearning images rather than erasing the identity.
16. **Step‑Local Forget Accuracy** — THIS step's identities only.  Unlike the
    cumulative forget acc (averaged over all identities forgotten so far), a
    failing step cannot hide behind an improving average.

### 12.2 Per‑Identity MIA Distribution (Violin Plot)

Shows the distribution of per‑identity AUC values across the forget identities
(60 at 600‑id, 75 at 750‑id) for each method.  A narrow distribution centred at
0.50 = uniform protection.  A wide or bimodal distribution = uneven forgetting.

### 12.3 Popularity‑Bin Stratification (Imbalanced Only)

Bar / line chart showing MIA AUC and forget‑holdout accuracy for high, medium,
and low popularity bins separately.  Parallel bars across bins = the method is
equity‑preserving.  At the 750‑id redesign the bins become 82:41:16 train
images (5:1 gradient, calibrated from VGG‑Face2's real range — Cao et al. 2018,
Wang et al. 2019, Liu et al. 2019).

### 12.4 Imbalanced Equity Plots (`src/imbalanced_plots.py`)

Six PNGs read from `single_shot_aggregated.json` + `dataset_imbalanced.csv`:

1. `01_forget_acc_by_bin` — forget‑holdout accuracy by popularity bin.
2. `02_mia_auc_by_bin` — per‑identity MIA AUC by bin.
3. `03_forget_train_gap_by_bin` — train‑minus‑holdout gap per method.
4. `04_auc_vs_images_per_id` — AUC vs images/identity scatter + regression
   (r, p): does data‑poverty predict privacy‑poverty?
5. `05_forget_utility_frontier` — retain vs forget scatter coloured by bin.
6. `06_kruskal_pvalues` — Kruskal‑Wallis p‑values per method × metric.

---

## 13. Iterative Protocol — Step‑Local vs Cumulative Metrics

**Why step‑local matters.**  The cumulative `forget_acc`/`mia_mean_auc` at
step *k* aggregates over *all* identities forgotten so far (5, 10, 15, …
growing).  A method that fails badly on step 7's identities but succeeds
elsewhere can hide the failure behind the cumulative average.  The step‑local
columns (`step_forget_acc`, `step_forget_train_acc`) report the *current* step's
identities only (from the `forget_step_N` split).

| Metric | Scope | Interpretation |
|---|---|---|
| `forget_acc` | All forget identities seen so far | Cumulative trajectory (existing) |
| `step_forget_acc` | THIS step's identities, holdout | Can't hide a failing step |
| `step_forget_train_acc` | THIS step's identities, train | Overfitting‑to‑forgetting detector: train ≈ 0 while holdout > 0 ⇒ the method memorised the unlearning images |

**Gap rule:** `step_forget_train_acc − step_forget_acc` ≤ 0.10 = genuine
identity‑level erasure; ≥ 0.15 = overfit to the unlearning signal
(Grimes et al. 2024).

---

## 14. Poisson Schedule & Long‑Tail Axes (750‑id design)

The 750‑id dataset iteration (see `.hermes/plans/new design context.md`)
adds two experimental axes grounded in the streaming‑forgetting literature:

| Axis | Column | Design | Citation |
|---|---|---|---|
| Temporal (balanced) | `forget_step` (uniform, 5 ids/step × 15) | ordinal‑safe, run in order, multi‑seed μ±σ | NeurIPS 2023 ML Unlearning Challenge; Tarun et al. 2023 |
| Temporal stress (balanced) | `forget_step_poisson` (seeded Poisson, λ=5) | single fixed schedule, variable batch sizes | Online Forgetting Process (arXiv:2012.01668) |
| Popularity (imbalanced) | `popularity_bin` / `images_per_identity` (82:41:16) | 5:1 gradient, single‑shot only (no time axis) | FaLW — Yu et al. 2026 (arXiv:2601.18650) |

**Analysis rule (Shen et al. 2025, arXiv:2507.15280):** all cross‑schedule /
cross‑bin comparisons use **cumulative forgotten count**, never raw step index
— streaming difficulty scales with the total variation V_T between consecutive
forget sets.

**Poisson interpretation:** GDPR‑style erasure requests arrive as an
independent stochastic process, so batch sizes vary step‑to‑step.  The Poisson
schedule is a **one‑shot stress test** (fixed, seeded, recorded in the
manifest), unlike the uniform schedule which supports multi‑seed μ±σ
statistics.

**Imbalanced hypothesis (H1, FaLW‑grounded):** high‑bin identities
(82 train images) are over‑learned → hardest to forget; low‑bin (16 images)
under‑learned → easiest to scrub.

**Protocols (imbalanced stress‑test chain, implemented in the pipeline):**
| Protocol | What it does | Where it lands |
|---|---|---|
| A. Baseline gate (free) | per‑bin MIA AUC on the ORIGINAL model before unlearning — a bin at ≈0.5 was never learned, not scrubbed | `single_shot` per‑bin block |
| B. Distance‑to‑oracle | **one oracle PER bin** — retrain excluding only that bin's forget identities (the oracle *does* change per bin); per‑bin behavioral distance after a fixed budget | `per_bin_oracle.py` → `oracles/` |
| C. Budget sweep (cleanest causal test) | sweep unlearning budget per method until each bin reaches its oracle = "difficulty"; high‑bin needing more → H1 supported | `select_protocol_c_methods.py` (dynamic 1‑best‑per‑category) → `budget_sweep.py` |
| D. Balanced cross‑check (confound control) | same 75 forget IDs through balanced (90/img); hard‑in‑balanced ⇒ identity‑tracks, easier ⇒ count‑tracks | shared split map across artifacts |
| E. Intensity modulation | fixed‑intensity baseline vs bins; expect FaLW skewed deviation | report |

On imbalanced, "budget" = method‑internal unlearning iterations
(ga_steps / ft_epochs / srl_epochs / ng_steps / msg_steps / ct_steps /
max_steps) — there is no forget_step schedule axis to sweep.

---

## 15. Order‑Stability Protocol (P3/P4)

**Question.**  Does *which* identities are forgotten at *which* step matter,
holding the pretrained model fixed?  A method is order‑stable if its final
forget/utility profile is insensitive to the permutation of the forget set
into steps.

**Protocol.**  One pretrained checkpoint (constant across seeds) × 5 runs,
each with a **different forget ordering** (`--order_seed` per run); report
μ±σ across runs for every metric.  Implementation: `VirtualIdentityDataset(
order_seed=N)` deterministically remaps which forget identities sit at which
uniform step (contiguous chunks, sizes preserved); the permutation is a pure
function of (sorted identity list, seed), so methods, step‑local evals and
the retrain oracle all see one consistent remap.

**Why it costs no retraining (P4).**  The pretrained model is trained once
and shared; only the *order of unlearning steps* changes.  The retrain
oracle is order‑independent (it retrains on retain‑only, never touches
forget order) and runs once, not 5×.

**Reading the result.**
- **Tight μ±σ across orderings** → the method's behaviour is driven by the
  model/identity properties, not by schedule luck (robust; CUFG‑style
  ordered‑vs‑random support).
- **Wide μ±σ** → order‑sensitive: later‑forgotten identities may benefit
  from more accumulated interference (or suffer from it).  This is the
  streaming‑forgetting regime Shen et al. (2025) warns about — report the
  spread, never average it away silently.
- **Per‑step gap metric** (`step_forget_train_acc − step_forget_acc`) is
  the natural per‑ordering diagnostic: an order‑stable method keeps the gap
  ≤ 0.10 at every step regardless of which identities arrive when.

**Caveats.**  Order‑stability applies to the **uniform** schedule only;
`order_seed` with `--schedule poisson` raises by design (the Poisson
schedule is a single fixed GDPR‑arrival stress test — remapping its batch
composition would defeat its purpose).  All cross‑ordering comparisons use
`cumulative_forgotten` as the x‑axis (Shen et al. 2025 V_T rule), not raw
step index.

---

## 16. AdaptiForget Ablation Protocol

Runs in the full pipeline as `slurm_ablation.sh` (after the HP jobs —
it reuses the main train checkpoint and the **tuned** AdaptiForget
best-config as the "Full" base, so the component contributions are
measured from the method as deployed; the report stage renders the
results as the "AdaptiForget Ablation" section).

Purpose: attribute the novel method's behaviour to its components.
Each of 6 variants disables exactly one component, on identical data
(same trained model, same seed, same forget set):

| Variant | Override | Component removed |
|---|---|---|
| Full AdaptiForget | — | none (baseline) |
| w/o adaptive λ | `kl_weight_init/max = 0.5` | KL weight schedule → constant |
| w/o mask refresh | `mask_refresh_every = 999999` | periodic saliency recompute → one-shot mask |
| w/o early stop | `early_stop_adv = -1.0` | forget-holdout early stop → full budget |
| w/o KL distil | `kl_weight = 0.0` | KL anchor → pure MSG |
| w/o masking (NG+) | `topk_fraction = 1.0` | saliency mask → all params updated |

Metrics per variant (all evaluated on holdout): `retain_id_acc`,
`forget_id_acc`, `mia_mean_auc`, **`forget_advantage`** (signed erasure
signal `max(0, 1 − 2·MIA)` — C1 semantics, NOT `abs(MIA − 0.5)`),
`fraction_leaked`, `steps_used`, `early_stopped`.

Interpretation: the drop in forget advantage / retain acc vs Full
AdaptiForget quantifies each component's contribution.  Early stop is
the compute-efficiency component (expect `steps_used` to hit
`max_steps` when disabled); masking + KL are the retain-protection
components (expect retain collapse when removed).

## 17. Verification Status (2026‑08)

- **Smoke test (v1.1 schema): PASSED on macOS** — `tests/smoke_test.py`
  trains 2 epochs on a 6‑identity synthetic set (3 retain / 3 forget, with
  `forget_step` + `forget_step_poisson` columns), runs single‑shot GA
  through schedule‑aware splits, and asserts: MIA AUC 0.31 (forgetting
  signal present), retain acc 0.56 (above chance).  ~10 min on CPU —
  the run is slow, not hung.
- **Order‑stability permutation: 13/13 ad‑hoc checks** — deterministic
  (same seed → same composition), seed‑distinct, covers all forget
  identities, preserves per‑step sizes; poisson+order_seed guard present.
- **HPC rerun pending** — requires the 750‑id CSVs (`forget_step_poisson`
  column on balanced; `image_subset` on both).

---

## References (abbreviated)

- **Yeom et al. (2018).** "Privacy Risk in Machine Learning." *IEEE CSF*.
  — Threshold MIA definition.
- **Choi & Na (2023).** "MUFAC/MUCAC: Identity‑Level Unlearning Benchmark."
  *arXiv:2311.02240*. — Per‑identity MIA, identity probe, forget‑holdout gap.
- **Cadet et al. (2024).** "Deep Unlearn: Benchmarking Machine Unlearning."
  *arXiv:2410.01276*. — Multi‑tier evaluation, fraction leaked, method pool.
- **Golatkar et al. (2020).** "Eternal Sunshine of the Spotless Net." *CVPR*.
  — Representation probes, temporal re‑emergence.
- **Grimes et al. (2024).** "Gone but Not Forgotten: Improved Benchmarks for
  Machine Unlearning." *DLS*. — Iterative protocol critique, forget‑train gap.
- **Carlini et al. (2022).** "Membership Inference Attacks From First
  Principles." *IEEE S&P*. — LiRA, max‑confidence attack.
- **Thudi et al. (2022).** "Unrolling SGD: Understanding Limits of Machine
  Unlearning." *arXiv*. — Canary‑based ground‑truth deletion verification.
- **Ginart et al. (2019).** "Making AI Forget You: Data Deletion in Machine
  Learning." *arXiv:1907.05012*. — Deletion foundation; single‑batch baseline.
- **Bourtoule et al. (2021).** "Machine Unlearning." *IEEE S&P*. — Retrain‑oracle
  gold standard.
- **Shen et al. (2025).** "Machine Unlearning for Streaming Forgetting."
  *arXiv:2507.15280*. — Difficulty scales with V_T; cumulative‑count axis.
- **Online Forgetting Process.** *arXiv:2012.01668*. — Poisson arrival model
  for GDPR‑style erasure requests.
- **Yu et al. (2026).** "Forgetting‑aware Loss Reweighting for Long‑tailed
  Unlearning" (FaLW). *arXiv:2601.18650*. — Heterogeneous + skewed unlearning
  deviation on long tails; grounds H1.
- **GENIU.** *arXiv:2406.07885*. — Class‑level imbalance unlearning.
- **CUFG.** *arXiv:2509.14633*. — Ordered vs random forgetting (future
  schedule variants).
- **Cao et al. (2018).** "VGG‑Face2." — 5:1 popularity gradient calibration.
- **Liu et al. (2019).** "Large‑Scale Long‑Tailed Recognition." *CVPR*. —
  Long‑tail distribution justification.
