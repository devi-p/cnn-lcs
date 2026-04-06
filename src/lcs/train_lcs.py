import argparse
import json
import os
import pickle
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from skExSTraCS import ExSTraCS
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.cnn.eval_utils import find_best_f1_threshold, metrics_at_threshold

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _colab_drive_root() -> Path | None:
    drive_path = Path("/content/drive/MyDrive/cnn-lcs")
    return drive_path if drive_path.exists() else None


def _default_features_dir() -> Path:
    colab_root = _colab_drive_root()
    if colab_root:
        return colab_root / "features"
    return PROJECT_ROOT / "outputs" / "features"


def _default_lcs_dir() -> Path:
    colab_root = _colab_drive_root()
    if colab_root:
        return colab_root / "lcs"
    return PROJECT_ROOT / "outputs" / "lcs"


def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return 0.0


def _selector_from_name(name: str, k: int, seed: int) -> SelectKBest:
    if name == "f_classif":
        return SelectKBest(f_classif, k=k)
    if name == "mutual_info":
        def _score_func(x_values: np.ndarray, y_values: np.ndarray) -> np.ndarray:
            return mutual_info_classif(x_values, y_values, random_state=seed)

        return SelectKBest(_score_func, k=k)
    raise ValueError(f"Unsupported selector: {name}")


def _extract_rule_table(model: ExSTraCS, selected_indices: np.ndarray) -> tuple[pd.DataFrame, int]:
    pop = model.population.popSet
    rules = sorted(pop, key=lambda rule: rule.accuracy * rule.numerosity, reverse=True)

    rows: list[dict[str, Any]] = []
    for rule in rules:
        mapped_attributes = [int(selected_indices[attr]) for attr in rule.specifiedAttList]
        rows.append(
            {
                "condition": str(rule.condition),
                "prediction": "Anomalous" if rule.phenotype == 1 else "Normal",
                "accuracy": float(rule.accuracy),
                "numerosity": int(rule.numerosity),
                "fitness": float(rule.fitness),
                "selected_feature_indices": str(rule.specifiedAttList),
                "original_feature_indices": str(mapped_attributes),
                "correct_count": int(rule.correctCount),
                "match_count": int(rule.matchCount),
            }
        )

    return pd.DataFrame(rows), len(pop)


def _copy_artifacts(src_dir: Path, dest_dir: Path, artifacts: list[str]) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    for artifact_name in artifacts:
        shutil.copy2(src_dir / artifact_name, dest_dir / artifact_name)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as file_obj:
        file_obj.write(json.dumps(payload) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2)


def train_lcs(
    features_dir: Path,
    output_dir: Path,
    k: int = 50,
    selector_name: str = "f_classif",
    learning_iterations: int = 100000,
    population_size: int = 3000,
    nu: float = 10.0,
    val_size: float = 0.2,
    seed: int = 42,
    threshold_step: float = 0.01,
    min_accuracy: float = 0.9125,
    min_precision: float = 0.4640,
    min_recall: float = 0.3225,
    min_auc: float = 0.7347,
    max_rules: int = 3000,
    run_tag: str | None = None,
    register_baseline: bool = False,
    publish_latest: bool = True,
    promote_if_pass: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = run_tag or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    x_train_all = np.load(features_dir / "train_features.npy")
    y_train_all = np.load(features_dir / "train_labels.npy")
    x_test = np.load(features_dir / "test_features.npy")
    y_test = np.load(features_dir / "test_labels.npy")

    print(
        "Loaded features: "
        f"train={x_train_all.shape}, test={x_test.shape}, "
        f"train_normal={(y_train_all == 0).sum()}, train_anomalous={(y_train_all == 1).sum()}"
    )

    x_train, x_val, y_train, y_val = train_test_split(
        x_train_all,
        y_train_all,
        test_size=val_size,
        random_state=seed,
        stratify=y_train_all,
    )

    print(
        f"Split sizes: train={x_train.shape[0]}, val={x_val.shape[0]}, test={x_test.shape[0]} "
        f"(seed={seed})"
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    x_test_scaled = scaler.transform(x_test)

    if k > x_train_scaled.shape[1]:
        raise ValueError(f"k={k} cannot be larger than feature dimension {x_train_scaled.shape[1]}")

    selector = _selector_from_name(selector_name, k=k, seed=seed)
    x_train_selected = selector.fit_transform(x_train_scaled, y_train)
    x_val_selected = selector.transform(x_val_scaled)
    x_test_selected = selector.transform(x_test_scaled)

    selected_indices = selector.get_support(indices=True)
    print(f"Selected features: k={len(selected_indices)} via {selector_name}")

    print("Training ExSTraCS...")
    model = ExSTraCS(learning_iterations=learning_iterations, N=population_size, nu=nu)
    model.fit(x_train_selected, y_train)

    val_prob = model.predict_proba(x_val_selected)[:, 1]
    val_best = find_best_f1_threshold(y_val, val_prob, step=threshold_step)
    chosen_threshold = float(val_best["threshold"])

    y_test_prob = model.predict_proba(x_test_selected)[:, 1]
    test_metrics = metrics_at_threshold(y_test, y_test_prob, chosen_threshold)
    test_auc = _safe_roc_auc(y_test, y_test_prob)
    test_metrics["auc"] = test_auc

    metrics_05 = metrics_at_threshold(y_test, y_test_prob, 0.5)
    metrics_05["auc"] = test_auc

    y_test_pred = test_metrics["y_pred"]
    test_confusion = confusion_matrix(y_test, y_test_pred).tolist()

    rule_extraction_error = None
    try:
        rules_df, rule_count = _extract_rule_table(model, selected_indices)
    except Exception as error:
        rule_extraction_error = str(error)
        rules_df = pd.DataFrame(
            columns=[
                "condition",
                "prediction",
                "accuracy",
                "numerosity",
                "fitness",
                "selected_feature_indices",
                "original_feature_indices",
                "correct_count",
                "match_count",
            ]
        )
        # If rules cannot be extracted, force non-promotion by violating the rule-count gate.
        rule_count = max_rules + 1

    gate_checks = {
        "accuracy": bool(test_metrics["accuracy"] >= min_accuracy),
        "precision": bool(test_metrics["precision"] >= min_precision),
        "recall": bool(test_metrics["recall"] >= min_recall),
        "auc": bool(test_metrics["auc"] >= min_auc),
        "rule_count": bool(rule_count <= max_rules),
    }
    gate_passed = all(gate_checks.values())

    metadata = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "features_dir": str(features_dir),
        "output_dir": str(output_dir),
        "params": {
            "k": int(k),
            "selector": selector_name,
            "learning_iterations": int(learning_iterations),
            "population_size": int(population_size),
            "nu": float(nu),
            "val_size": float(val_size),
            "seed": int(seed),
            "threshold_step": float(threshold_step),
        },
        "val_metrics": {
            "threshold": float(val_best["threshold"]),
            "precision": float(val_best["precision"]),
            "recall": float(val_best["recall"]),
            "f1": float(val_best["f1"]),
            "accuracy": float(val_best["accuracy"]),
        },
        "test_metrics_at_val_threshold": {
            "threshold": float(test_metrics["threshold"]),
            "precision": float(test_metrics["precision"]),
            "recall": float(test_metrics["recall"]),
            "f1": float(test_metrics["f1"]),
            "accuracy": float(test_metrics["accuracy"]),
            "auc": float(test_metrics["auc"]),
        },
        "test_metrics_at_0_5": {
            "threshold": 0.5,
            "precision": float(metrics_05["precision"]),
            "recall": float(metrics_05["recall"]),
            "f1": float(metrics_05["f1"]),
            "accuracy": float(metrics_05["accuracy"]),
            "auc": float(metrics_05["auc"]),
        },
        "test_confusion_matrix": test_confusion,
        "rule_count": int(rule_count),
        "rule_extraction_error": rule_extraction_error,
        "gates": {
            "min_accuracy": float(min_accuracy),
            "min_precision": float(min_precision),
            "min_recall": float(min_recall),
            "min_auc": float(min_auc),
            "max_rules": int(max_rules),
            "checks": gate_checks,
            "passed": bool(gate_passed),
        },
    }

    print("LCS test evaluation (threshold tuned on validation):")
    print(f"  threshold = {test_metrics['threshold']:.2f}")
    print(f"  Accuracy  = {test_metrics['accuracy']:.4f}")
    print(f"  F1 Score  = {test_metrics['f1']:.4f}")
    print(f"  Precision = {test_metrics['precision']:.4f}")
    print(f"  Recall    = {test_metrics['recall']:.4f}")
    print(f"  AUC       = {test_metrics['auc']:.4f}")
    print("  Confusion Matrix:")
    print(np.array(test_confusion))
    print("  (rows=actual, cols=predicted | 0=normal, 1=anomalous)")
    print(f"Rule count: {rule_count}")
    if rule_extraction_error:
        print(f"Rule extraction warning: {rule_extraction_error}")
    print(f"Gate passed: {gate_passed}")

    artifact_names = ["lcs_model.pkl", "scaler.pkl", "selector.pkl", "selected_feature_indices.npy", "rules.csv"]

    with open(run_dir / "lcs_model.pkl", "wb") as file_obj:
        pickle.dump(model, file_obj)
    with open(run_dir / "scaler.pkl", "wb") as file_obj:
        pickle.dump(scaler, file_obj)
    with open(run_dir / "selector.pkl", "wb") as file_obj:
        pickle.dump(selector, file_obj)
    np.save(run_dir / "selected_feature_indices.npy", selected_indices)
    rules_df.to_csv(run_dir / "rules.csv", index=False)
    _write_json(run_dir / "run_metrics.json", metadata)

    _append_jsonl(output_dir / "experiment_runs.jsonl", metadata)

    if register_baseline:
        _write_json(output_dir / "baseline_metrics.json", metadata)
        print(f"Wrote baseline registry: {output_dir / 'baseline_metrics.json'}")

    if publish_latest:
        _copy_artifacts(run_dir, output_dir, artifact_names)
        _write_json(output_dir / "latest_run.json", metadata)

    if promote_if_pass and gate_passed:
        _copy_artifacts(run_dir, output_dir, artifact_names)
        approved_manifest = {
            "approved": True,
            "approved_at": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id,
            "artifacts": artifact_names,
            "metrics": metadata["test_metrics_at_val_threshold"],
            "rule_count": rule_count,
            "gates": metadata["gates"],
        }
        _write_json(output_dir / "approved_model.json", approved_manifest)
        print(f"Promoted run and wrote approval manifest: {output_dir / 'approved_model.json'}")
    elif promote_if_pass:
        print("Run did not satisfy all gates; no approval manifest was written.")

    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ExSTraCS with safe, validation-driven tuning.")
    parser.add_argument("--features-dir", type=Path, default=_default_features_dir())
    parser.add_argument("--output-dir", type=Path, default=_default_lcs_dir())

    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--selector", choices=["f_classif", "mutual_info"], default="f_classif")
    parser.add_argument("--learning-iterations", type=int, default=100000)
    parser.add_argument("--population-size", type=int, default=3000)
    parser.add_argument("--nu", type=float, default=10.0)

    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold-step", type=float, default=0.01)

    parser.add_argument("--min-accuracy", type=float, default=0.9125)
    parser.add_argument("--min-precision", type=float, default=0.4640)
    parser.add_argument("--min-recall", type=float, default=0.3225)
    parser.add_argument("--min-auc", type=float, default=0.7347)
    parser.add_argument("--max-rules", type=int, default=3000)

    parser.add_argument("--run-tag", type=str, default=None)
    parser.add_argument("--register-baseline", action="store_true")
    parser.add_argument("--promote-if-pass", action="store_true")

    parser.add_argument(
        "--no-publish-latest",
        dest="publish_latest",
        action="store_false",
        help="Do not copy this run into output_dir root artifacts.",
    )
    parser.set_defaults(publish_latest=True)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_lcs(
        features_dir=args.features_dir,
        output_dir=args.output_dir,
        k=args.k,
        selector_name=args.selector,
        learning_iterations=args.learning_iterations,
        population_size=args.population_size,
        nu=args.nu,
        val_size=args.val_size,
        seed=args.seed,
        threshold_step=args.threshold_step,
        min_accuracy=args.min_accuracy,
        min_precision=args.min_precision,
        min_recall=args.min_recall,
        min_auc=args.min_auc,
        max_rules=args.max_rules,
        run_tag=args.run_tag,
        register_baseline=args.register_baseline,
        publish_latest=args.publish_latest,
        promote_if_pass=args.promote_if_pass,
    )


if __name__ == "__main__":
    main()
