"""Unified Hydra entrypoint for phase 2, phase 3, and phase 4/5 runs.

This file centralizes the current runner logic so workflows can be
selected from config instead of separate CLI scripts.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Optional

import hydra
from omegaconf import DictConfig, OmegaConf

from device_utils import resolve_device


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(__file__).resolve().parents[2]


def _resolve_path(path_value: str | Path | None, base: Path = PROJECT_ROOT) -> Optional[Path]:
    if path_value is None:
        return None
    path = Path(path_value)
    return path if path.is_absolute() else (base / path).resolve()


def _print_block(title: str) -> None:
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print(f"{'=' * 72}")


def _apply_smoke_overrides(mapping: dict, overrides: dict) -> None:
    """Merge smoke-test overrides into a nested defaults mapping."""
    for key, value in overrides.items():
        if key in mapping and isinstance(mapping[key], dict) and isinstance(value, dict):
            mapping[key].update(value)
        else:
            mapping[key] = value


def _run_phase2(cfg: DictConfig) -> None:
    from mia import run_mia_full
    from model import load_model
    from single_shot_experiment import run_single_shot
    from train import train

    phase = cfg.workflow.phase2
    device = resolve_device(cfg.runtime.device)
    csv_path = _resolve_path(cfg.paths.csv, base=DATA_ROOT)
    out_root = _resolve_path(cfg.paths.out)
    assert csv_path is not None
    assert out_root is not None
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    ckpt_dir = out_root / "checkpoints"
    shot_dir = out_root / "single_shot"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    shot_dir.mkdir(parents=True, exist_ok=True)

    model_path = _resolve_path(phase.train.model_path) if phase.train.model_path else ckpt_dir / "original_model_best.pt"
    print(OmegaConf.to_yaml(cfg.workflow, resolve=True))
    _print_block("PHASE 2: BASELINES + MIA")
    print(f"  Device     : {device}")
    print(f"  CSV        : {csv_path}")
    print(f"  Output dir : {out_root}")
    print(f"  Model path  : {model_path}")

    if phase.train.enabled:
        if phase.train.skip_train and model_path.exists():
            print(f"[SKIP] Training — loading existing model: {model_path}")
        else:
            print("\n[STEP 1] Training original model M on retain + forget data...")
            train(
                csv_path=str(csv_path),
                save_dir=str(ckpt_dir),
                epochs=int(phase.train.epochs),
                lr=float(phase.train.lr),
                batch_size=int(phase.train.batch_size),
                seed=int(cfg.runtime.seed),
                run_name="original_model",
            )

    if phase.single_shot.enabled:
        print("\n[STEP 2] Running single-shot unlearning experiments...")
        methods = list(phase.single_shot.methods)
        if phase.single_shot.skip_retrain_oracle and "retrain" in methods:
            methods = [m for m in methods if m != "retrain"]
            print("[INFO] Skipping retrain oracle")

        run_single_shot(
            csv_path=str(csv_path),
            model_path=str(model_path),
            forget_step=int(phase.single_shot.forget_step),
            methods=methods,
            out_dir=str(shot_dir),
            device_str=cfg.runtime.device,
            seed=int(cfg.runtime.seed),
            ga_steps=int(phase.single_shot.ga_steps),
            ga_lr=float(phase.single_shot.ga_lr),
            srl_epochs=int(phase.single_shot.srl_epochs),
            srl_lr=float(phase.single_shot.srl_lr),
            ft_epochs=int(phase.single_shot.ft_epochs),
            ft_lr=float(phase.single_shot.ft_lr),
            retrain_epochs=int(phase.single_shot.retrain_epochs),
        )

    if phase.mia.enabled:
        print("\n[STEP 3] Running MIA on original model (pre-unlearning baseline)...")
        original_model = load_model(str(model_path), device=str(device))
        mia_baseline = run_mia_full(
            original_model,
            str(csv_path),
            device,
            forget_step=int(phase.single_shot.forget_step),
            score_type=str(phase.mia.score_type),
            verbose=True,
        )

        import json

        mia_path = out_root / "mia_baseline.json"
        with open(mia_path, "w") as handle:
            json.dump({"original_model": mia_baseline}, handle, indent=2)
        print(f"[OK] MIA baseline saved → {mia_path}")

    print(f"\n[OK] Phase 2 complete → {out_root}")


def _run_phase3(cfg: DictConfig) -> None:
    from comprehensive_eval import DEFAULT_CONFIGS, run_comprehensive_eval
    from hparam_search import GRIDS, run_search

    phase = cfg.workflow.phase3
    device = resolve_device(cfg.runtime.device)
    csv_path = _resolve_path(cfg.paths.csv, base=DATA_ROOT)
    model_path = _resolve_path(cfg.paths.model)
    out_root = _resolve_path(cfg.paths.out)
    assert csv_path is not None and model_path is not None and out_root is not None

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    out_path = out_root / phase.out_subdir
    hparam_dir = _resolve_path(phase.hparam_dir) if phase.hparam_dir else out_path / "hparam"
    stage_a_dir = out_path / "stage_a_default"
    stage_c_dir = out_path / "stage_c_best"
    for directory in [out_path, hparam_dir, stage_a_dir, stage_c_dir]:
        if directory is not None:
            directory.mkdir(parents=True, exist_ok=True)

    _print_block("PHASE 3: STATE-OF-THE-ART UNLEARNING")
    print(f"  Stages   : {list(phase.stages)}")
    print(f"  Device   : {device}")
    print(f"  Forget   : step {phase.forget_step}")

    defaults_backup = deepcopy(DEFAULT_CONFIGS)
    grids_backup = deepcopy(GRIDS)
    try:
        if phase.ng_steps is not None:
            DEFAULT_CONFIGS["ng_plus"]["ng_steps"] = int(phase.ng_steps)
        if phase.msg_steps is not None:
            DEFAULT_CONFIGS["msg"]["msg_steps"] = int(phase.msg_steps)
            DEFAULT_CONFIGS["msg_kd"]["msg_steps"] = int(phase.msg_steps)
        if phase.ct_steps is not None:
            DEFAULT_CONFIGS["ct"]["ct_steps"] = int(phase.ct_steps)

        if "A" in phase.stages:
            _print_block("STAGE A: Default-Config Evaluation")
            run_comprehensive_eval(
                csv_path=str(csv_path),
                model_path=str(model_path),
                forget_step=int(phase.forget_step),
                methods=phase.methods,
                out_dir=str(stage_a_dir),
                device_str=cfg.runtime.device,
                seed=int(cfg.runtime.seed),
                skip_retrain=bool(phase.skip_retrain),
                hparam_dir=None,
            )

        if "B" in phase.stages:
            _print_block(f"STAGE B: HP Sensitivity ({phase.search} search)")
            for method in phase.hp_methods:
                print(f"\n  → Searching {method}...")
                if phase.ng_steps is not None and method == "ng_plus":
                    GRIDS["ng_plus"]["ng_steps"] = [int(phase.ng_steps)]
                if phase.msg_steps is not None and method in ("msg", "msg_kd"):
                    GRIDS["msg"]["msg_steps"] = [int(phase.msg_steps)]
                    GRIDS["msg_kd"]["msg_steps"] = [int(phase.msg_steps)]
                if phase.ct_steps is not None and method == "ct":
                    GRIDS["ct"]["ct_steps"] = [int(phase.ct_steps)]

                run_search(
                    method_name=method,
                    csv_path=str(csv_path),
                    model_path=str(model_path),
                    forget_step=int(phase.forget_step),
                    search_type=str(phase.search),
                    n_random_trials=int(phase.n_random),
                    out_dir=str(hparam_dir),
                    device_str=cfg.runtime.device,
                    seed=int(cfg.runtime.seed),
                )

        if "C" in phase.stages:
            _print_block("STAGE C: Best-Config Comprehensive Evaluation")
            run_comprehensive_eval(
                csv_path=str(csv_path),
                model_path=str(model_path),
                forget_step=int(phase.forget_step),
                methods=phase.methods,
                out_dir=str(stage_c_dir),
                device_str=cfg.runtime.device,
                seed=int(cfg.runtime.seed),
                skip_retrain=bool(phase.skip_retrain),
                hparam_dir=str(hparam_dir),
            )
    finally:
        DEFAULT_CONFIGS.clear()
        DEFAULT_CONFIGS.update(defaults_backup)
        GRIDS.clear()
        GRIDS.update(grids_backup)

    print(f"\n[OK] Phase 3 complete → {out_path}")


def _run_phase45(cfg: DictConfig) -> None:
    from ablation_study import run_ablation
    from comprehensive_eval import DEFAULT_CONFIGS, run_comprehensive_eval
    from iterative_unlearning import DEFAULT_ITER_CONFIGS, run_all_iterative
    from stability_analysis import run_stability_analysis

    phase = cfg.workflow.phase45
    device = resolve_device(cfg.runtime.device)
    csv_path = _resolve_path(cfg.paths.csv, base=DATA_ROOT)
    model_path = _resolve_path(cfg.paths.model)
    out_root = _resolve_path(cfg.paths.out)
    assert csv_path is not None and model_path is not None and out_root is not None

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    out_path = out_root / phase.out_subdir
    s4a_dir = out_path / "phase4_single_shot"
    s4b_dir = out_path / "phase4_ablation"
    s5a_dir = out_path / "phase5_iterative"
    s5b_dir = out_path / "phase5_plots"
    for directory in [out_path, s4a_dir, s4b_dir, s5a_dir, s5b_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    smoke_cfg = {
        "ng_plus": {"ng_steps": 30},
        "msg": {"msg_steps": 30},
        "msg_kd": {"msg_steps": 30},
        "ct": {"ct_steps": 30},
        "adaptiformet": {"max_steps": 50},
        "ga": {"ga_steps": 30},
        "srl": {"srl_epochs": 1},
        "ft": {"ft_epochs": 1},
    }

    _print_block("PHASE 4 & 5: NOVEL VARIANT + ITERATIVE UNLEARNING")
    print(f"  Stages   : {list(phase.stages)}")
    print(f"  N steps  : {phase.n_steps}")
    print(f"  Device   : {device}")

    defaults_backup = deepcopy(DEFAULT_CONFIGS)
    iter_backup = deepcopy(DEFAULT_ITER_CONFIGS)
    try:
        if phase.smoke_test:
            print("\n[SMOKE TEST MODE] Reducing step counts…")
            _apply_smoke_overrides(DEFAULT_CONFIGS, smoke_cfg)
            _apply_smoke_overrides(DEFAULT_ITER_CONFIGS, smoke_cfg)

        if "4A" in phase.stages:
            _print_block("STAGE 4A: AdaptiForget Single-Shot Evaluation")
            methods_4a = ["ng_plus", "msg", "msg_kd", "ct", "adaptiformet"]
            if phase.skip_retrain:
                methods_4a = [m for m in methods_4a if m != "retrain"]

            run_comprehensive_eval(
                csv_path=str(csv_path),
                model_path=str(model_path),
                forget_step=0,
                methods=methods_4a,
                out_dir=str(s4a_dir),
                device_str=cfg.runtime.device,
                seed=int(cfg.runtime.seed),
                skip_retrain=bool(phase.skip_retrain),
                hparam_dir=str(_resolve_path(phase.hparam_dir)) if phase.hparam_dir else None,
            )

        if "4B" in phase.stages:
            _print_block("STAGE 4B: AdaptiForget Ablation Study")
            run_ablation(
                csv_path=str(csv_path),
                model_path=str(model_path),
                out_dir=str(s4b_dir),
                device_str=cfg.runtime.device,
                seed=int(cfg.runtime.seed),
                smoke_test=bool(phase.smoke_test),
            )

        if "5A" in phase.stages:
            _print_block(f"STAGE 5A: Iterative Unlearning ({phase.n_steps} steps, {phase.mode})")
            methods_5a = list(phase.methods) if phase.methods is not None else [
                "no_unlearning", "ga", "srl", "ft",
                "ng_plus", "msg", "msg_kd", "ct", "adaptiformet",
            ]
            if phase.skip_retrain and "retrain" in methods_5a:
                methods_5a = [m for m in methods_5a if m != "retrain"]

            run_all_iterative(
                csv_path=str(csv_path),
                model_path=str(model_path),
                methods=methods_5a,
                n_steps=int(phase.n_steps),
                mode=str(phase.mode),
                out_dir=str(s5a_dir),
                device_str=cfg.runtime.device,
                seed=int(cfg.runtime.seed),
                checkpoint_every=int(phase.checkpoint_every),
            )

        if "5B" in phase.stages:
            _print_block("STAGE 5B: Stability Analysis Plots")
            combined_csv = s5a_dir / "iterative_combined.csv"
            if not combined_csv.exists():
                print(f"  [WARN] {combined_csv} not found — run Stage 5A first.")
            else:
                run_stability_analysis(str(combined_csv), str(s5b_dir))
    finally:
        DEFAULT_CONFIGS.clear()
        DEFAULT_CONFIGS.update(defaults_backup)
        DEFAULT_ITER_CONFIGS.clear()
        DEFAULT_ITER_CONFIGS.update(iter_backup)

    print(f"\n[OK] Phase 4/5 complete → {out_path}")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg, resolve=True))
    workflow_name = str(cfg.workflow.name)

    if workflow_name == "phase2":
        _run_phase2(cfg)
    elif workflow_name == "phase3":
        _run_phase3(cfg)
    elif workflow_name == "phase45":
        _run_phase45(cfg)
    else:
        raise ValueError(f"Unknown workflow '{workflow_name}'. Expected phase2, phase3, or phase45.")


if __name__ == "__main__":
    main()