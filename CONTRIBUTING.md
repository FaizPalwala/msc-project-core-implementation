# Contributing

This is a research-grade machine unlearning evaluation framework designed to be
open-source and extensible after the dissertation is complete. Contributions are
welcome in the following areas.

## Adding a new unlearning method

1. Implement your method in a new file under `src/` following the standard
   signature used by all methods in `src/baselines.py`:

   ```python
   from interfaces import UnlearningResult

   def my_method(
       model: nn.Module,
       csv_path: str,
       device: torch.device,
       **kwargs,
   ) -> UnlearningResult:
       ...
       return {"model": unlearned_model, "method": "MyMethod", "metrics": {...}}
   ```

   The return contract (`UnlearningResult`) requires `model`, `method`, and
   `metrics` keys — see `src/interfaces.py`.  Every runner validates the
   contract after calling your method, so violations surface immediately.

2. The model is a dual-head ResNet-18.  `model(x)` returns
   `(identity_logits, age_logits)`.  Use the shared `_combined_loss()` helper
   from `src/baselines.py` to compute the joint loss.

3. Register your method in the appropriate registry at the bottom of your
   module, or add a new registry and merge it in `src/single_shot.py`
   and `src/iterative.py`.

4. Add a config file under `configs/methods/<your_method>.yaml` with default
   hyperparameter values.

5. (Optional) Add a hyperparameter search grid in `src/hparam_search.py`
   under `GRIDS` and/or `RANDOM_RANGES`.

## Adding a new evaluation metric

1. Add the metric function to the appropriate module:
   - Output-level metrics → `src/evaluate.py`
   - Privacy attacks → `src/mia.py`
   - Representation probes → `src/probes.py`

2. Wire it into the single-shot runner (`src/single_shot.py`) and/or the
   iterative runner (`src/iterative.py`).

3. If the metric produces data suitable for visualisation, add a corresponding
   plot function to `src/stability.py` and a numbered entry in
   `run_stability_analysis()`.

## Code style

- Python 3.10+, fully type-annotated.
- Follow the existing function signatures, docstring conventions, and the
  `from __future__ import annotations` pattern used throughout.
- Run `python -m py_compile src/*.py` before submitting.
- Keep methods self-contained — each unlearning method file should import
  only `torch`, `numpy`, and the shared helpers from `baselines.py` and
  `dataset.py`.

## Project structure

```
configs/
  methods/                 ← per-method hyperparameters (single source of truth)
scripts/
  slurm_*.sh               ← per-stage Slurm templates
  hpc_full_pipeline.sh     ← job-chaining launcher
src/
  model.py                 ← dual-head ResNet-18
  dataset.py               ← SFHQ-InstantID dataset + transforms
  train.py                 ← training loop
  baselines.py             ← classical unlearning methods (No-Op, Retrain, GA, SRL, FT)
  sota_methods.py          ← SOTA methods (NG+, MSG, CT)
  novel_variant.py         ← novel variants (MSG-KD, AdaptiForget)
  single_shot.py           ← single-shot evaluation runner
  iterative.py             ← iterative protocol (15 steps × 4 IDs)
  stability.py             ← publication-quality plots
  hparam_search.py         ← hyperparameter grid/random search
  ablation_study.py        ← AdaptiForget component ablation
  canary.py                ← pixel-level canary insertion + verification
  evaluate.py / mia.py / probes.py  ← evaluation framework
  config_loader.py         ← loads method HPs with smoke scaling
  device_utils.py          ← CUDA/MPS/CPU device selection
```

## License

Code: MIT.  See [LICENSE](LICENSE).
