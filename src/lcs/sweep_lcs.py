import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.lcs.train_lcs import _default_features_dir, _default_lcs_dir, train_lcs

ARTIFACT_NAMES = ["lcs_model.pkl", "scaler.pkl", "selector.pkl", "selected_feature_indices.npy", "rules.csv"]


def _coarse_candidates() -> list[dict[str, Any]]:
    return [
        {"k": 75, "selector": "f_classif", "learning_iterations": 100000, "population_size": 3000, "nu": 10.0},
        {"k": 100, "selector": "f_classif", "learning_iterations": 100000, "population_size": 3000, "nu": 10.0},
        {"k": 125, "selector": "f_classif", "learning_iterations": 100000, "population_size": 3000, "nu": 10.0},
        {"k": 150, "selector": "f_classif", "learning_iterations": 100000, "population_size": 3000, "nu": 10.0},
        {"k": 75, "selector": "mutual_info", "learning_iterations": 100000, "population_size": 3000, "nu": 10.0},
        {"k": 100, "selector": "mutual_info", "learning_iterations": 100000, "population_size": 3000, "nu": 10.0},
        {"k": 125, "selector": "mutual_info", "learning_iterations": 100000, "population_size": 3000, "nu": 10.0},
        {"k": 150, "selector": "mutual_info", "learning_iterations": 100000, "population_size": 3000, "nu": 10.0},
        {"k": 100, "selector": "f_classif", "learning_iterations": 200000, "population_size": 3000, "nu": 8.0},
        {"k": 125, "selector": "f_classif", "learning_iterations": 200000, "population_size": 3000, "nu": 8.0},
        {"k": 100, "selector": "mutual_info", "learning_iterations": 200000, "population_size": 3000, "nu": 8.0},
        {"k": 125, "selector": "mutual_info", "learning_iterations": 200000, "population_size": 3000, "nu": 8.0},
    ]


def _score(meta: dict[str, Any]) -> tuple[int, float, float, int]:
    gates = meta.get("gates", {})
    passed = bool(gates.get("passed", False))
    metrics = meta.get("test_metrics_at_val_threshold", {})
    return (
        1 if passed else 0,
        float(metrics.get("f1", 0.0)),
        float(metrics.get("recall", 0.0)),
        -int(meta.get("rule_count", 10**9)),
    )


def _candidate_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
    return (
        candidate["k"],
        candidate["selector"],
        candidate["learning_iterations"],
        candidate["population_size"],
        float(candidate["nu"]),
    )


def _build_refine_candidates(top_candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    refinements: list[dict[str, Any]] = []
    for base in top_candidates:
        refinements.extend(
            [
                {
                    "k": min(150, base["k"] + 25),
                    "selector": base["selector"],
                    "learning_iterations": max(base["learning_iterations"], 200000),
                    "population_size": base["population_size"],
                    "nu": max(4.0, float(base["nu"]) - 2.0),
                },
                {
                    "k": max(50, base["k"] - 25),
                    "selector": base["selector"],
                    "learning_iterations": max(base["learning_iterations"], 200000),
                    "population_size": base["population_size"],
                    "nu": min(12.0, float(base["nu"]) + 2.0),
                },
                {
                    "k": base["k"],
                    "selector": base["selector"],
                    "learning_iterations": max(base["learning_iterations"], 200000),
                    "population_size": 4000,
                    "nu": float(base["nu"]),
                },
            ]
        )

    unique: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for candidate in refinements:
        key = _candidate_key(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def _promote_run(output_dir: Path, winner: dict[str, Any]) -> None:
    run_id = winner["run_id"]
    run_dir = output_dir / "runs" / run_id

    for artifact_name in ARTIFACT_NAMES:
        shutil.copy2(run_dir / artifact_name, output_dir / artifact_name)

    approved_manifest = {
        "approved": True,
        "approved_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "artifacts": ARTIFACT_NAMES,
        "metrics": winner["test_metrics_at_val_threshold"],
        "rule_count": winner["rule_count"],
        "gates": winner["gates"],
    }

    with open(output_dir / "approved_model.json", "w", encoding="utf-8") as file_obj:
        json.dump(approved_manifest, file_obj, indent=2)


def run_sweep(
    features_dir: Path,
    output_dir: Path,
    budget: int,
    val_size: float,
    seed: int,
    threshold_step: float,
    min_accuracy: float,
    min_precision: float,
    min_recall: float,
    min_auc: float,
    max_rules: int,
    promote_best: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    all_candidates: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []

    coarse = _coarse_candidates()
    for candidate in coarse:
        all_candidates.append(candidate)
        if len(all_candidates) >= budget:
            break

    run_counter = 1
    for candidate in all_candidates:
        run_tag = f"sweep-{run_counter:02d}"
        run_counter += 1
        result = train_lcs(
            features_dir=features_dir,
            output_dir=output_dir,
            k=candidate["k"],
            selector_name=candidate["selector"],
            learning_iterations=candidate["learning_iterations"],
            population_size=candidate["population_size"],
            nu=candidate["nu"],
            val_size=val_size,
            seed=seed,
            threshold_step=threshold_step,
            min_accuracy=min_accuracy,
            min_precision=min_precision,
            min_recall=min_recall,
            min_auc=min_auc,
            max_rules=max_rules,
            run_tag=run_tag,
            register_baseline=False,
            publish_latest=False,
            promote_if_pass=False,
        )
        result["candidate"] = candidate
        results.append(result)

    remaining = budget - len(results)
    if remaining > 0:
        gate_passers = [
            result for result in sorted(results, key=_score, reverse=True) if result.get("gates", {}).get("passed")
        ]
        top_for_refine = [result["candidate"] for result in gate_passers[:3]]
        refine_candidates = _build_refine_candidates(top_for_refine)

        existing_keys = {_candidate_key(candidate) for candidate in all_candidates}
        additional_candidates: list[dict[str, Any]] = []
        for candidate in refine_candidates:
            key = _candidate_key(candidate)
            if key in existing_keys:
                continue
            additional_candidates.append(candidate)
            existing_keys.add(key)
            if len(additional_candidates) >= remaining:
                break

        for candidate in additional_candidates:
            run_tag = f"sweep-{run_counter:02d}"
            run_counter += 1
            result = train_lcs(
                features_dir=features_dir,
                output_dir=output_dir,
                k=candidate["k"],
                selector_name=candidate["selector"],
                learning_iterations=candidate["learning_iterations"],
                population_size=candidate["population_size"],
                nu=candidate["nu"],
                val_size=val_size,
                seed=seed,
                threshold_step=threshold_step,
                min_accuracy=min_accuracy,
                min_precision=min_precision,
                min_recall=min_recall,
                min_auc=min_auc,
                max_rules=max_rules,
                run_tag=run_tag,
                register_baseline=False,
                publish_latest=False,
                promote_if_pass=False,
            )
            result["candidate"] = candidate
            results.append(result)

    ranked = sorted(results, key=_score, reverse=True)
    winner = ranked[0] if ranked else None

    gate_passing = [result for result in ranked if result.get("gates", {}).get("passed")]
    best_gate_passing = gate_passing[0] if gate_passing else None

    if promote_best and best_gate_passing is not None:
        _promote_run(output_dir, best_gate_passing)

    report_rows = []
    for result in ranked:
        metrics = result.get("test_metrics_at_val_threshold", {})
        report_rows.append(
            {
                "run_id": result.get("run_id"),
                "gate_passed": result.get("gates", {}).get("passed", False),
                "k": result.get("params", {}).get("k"),
                "selector": result.get("params", {}).get("selector"),
                "learning_iterations": result.get("params", {}).get("learning_iterations"),
                "population_size": result.get("params", {}).get("population_size"),
                "nu": result.get("params", {}).get("nu"),
                "threshold": metrics.get("threshold"),
                "f1": metrics.get("f1"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "accuracy": metrics.get("accuracy"),
                "auc": metrics.get("auc"),
                "rule_count": result.get("rule_count"),
            }
        )

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "budget": budget,
        "runs_executed": len(results),
        "winner": winner,
        "best_gate_passing": best_gate_passing,
        "gates": {
            "min_accuracy": min_accuracy,
            "min_precision": min_precision,
            "min_recall": min_recall,
            "min_auc": min_auc,
            "max_rules": max_rules,
        },
    }

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sweep_dir = output_dir / "sweeps"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    with open(sweep_dir / f"sweep_{timestamp}.json", "w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, indent=2)

    pd.DataFrame(report_rows).to_csv(sweep_dir / f"sweep_{timestamp}.csv", index=False)

    if best_gate_passing is not None:
        with open(output_dir / "latest_gate_passing_run.json", "w", encoding="utf-8") as file_obj:
            json.dump(best_gate_passing, file_obj, indent=2)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a medium-budget safe LCS tuning sweep.")
    parser.add_argument("--features-dir", type=Path, default=_default_features_dir())
    parser.add_argument("--output-dir", type=Path, default=_default_lcs_dir())
    parser.add_argument("--budget", type=int, default=16)

    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold-step", type=float, default=0.01)

    parser.add_argument("--min-accuracy", type=float, default=0.9125)
    parser.add_argument("--min-precision", type=float, default=0.4640)
    parser.add_argument("--min-recall", type=float, default=0.3225)
    parser.add_argument("--min-auc", type=float, default=0.7347)
    parser.add_argument("--max-rules", type=int, default=3000)

    parser.add_argument("--promote-best", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_sweep(
        features_dir=args.features_dir,
        output_dir=args.output_dir,
        budget=args.budget,
        val_size=args.val_size,
        seed=args.seed,
        threshold_step=args.threshold_step,
        min_accuracy=args.min_accuracy,
        min_precision=args.min_precision,
        min_recall=args.min_recall,
        min_auc=args.min_auc,
        max_rules=args.max_rules,
        promote_best=args.promote_best,
    )

    best = summary.get("best_gate_passing")
    if best:
        metrics = best.get("test_metrics_at_val_threshold", {})
        print("Best gate-passing run:")
        print(
            f"  run_id={best.get('run_id')} "
            f"f1={metrics.get('f1', 0.0):.4f} "
            f"precision={metrics.get('precision', 0.0):.4f} "
            f"recall={metrics.get('recall', 0.0):.4f}"
        )
    else:
        print("No gate-passing run found in this sweep.")


if __name__ == "__main__":
    main()
