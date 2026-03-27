#!/usr/bin/env python3
"""Evaluate four-class action separability from segmented OpenBCI recordings."""

from __future__ import annotations

import argparse
import json
import os
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs/.matplotlib").resolve()))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters
from tsfresh.utilities.dataframe_functions import impute


CHANNEL_NAMES = ["ch1", "ch2", "ch3", "ch4"]
DEFAULT_LABEL_ORDER = ["down", "left", "right", "up"]


@dataclass(frozen=True)
class FoldSplit:
    fold: int
    train_ids: np.ndarray
    test_ids: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate action separability with TSFresh features and classic ML")
    parser.add_argument(
        "--segmentation-dir",
        type=Path,
        default=Path("outputs/segmentation"),
        help="Directory containing manifest.json and *_segments.npy files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/classification_eval"),
        help="Directory for metrics, confusion matrices, and run config",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of contiguous blocked folds per class",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible models and permutation baseline",
    )
    parser.add_argument(
        "--feature-mode",
        type=str,
        default="single,fusion",
        help="Comma-separated subset of: single,fusion",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="lr,rf,knn,dt",
        help="Comma-separated subset of: lr,rf,knn,dt",
    )
    parser.add_argument(
        "--permutation-runs",
        type=int,
        default=20,
        help="Number of label-permutation repetitions for the baseline summary",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=0,
        help="Parallel jobs for TSFresh. Use 0 to let TSFresh decide.",
    )
    return parser.parse_args()


def validate_requested_items(raw_value: str, allowed: set[str], field_name: str) -> list[str]:
    items = [item.strip() for item in raw_value.split(",") if item.strip()]
    invalid = sorted(set(items) - allowed)
    if invalid:
        raise ValueError(f"Invalid {field_name}: {invalid}. Allowed: {sorted(allowed)}")
    if not items:
        raise ValueError(f"No {field_name} provided.")
    return items


def load_manifest(segmentation_dir: Path) -> dict[str, Any]:
    manifest_path = segmentation_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def resolve_label_order(manifest: dict[str, Any]) -> list[str]:
    manifest_labels = [str(item["direction"]) for item in manifest.get("files", [])]
    if not manifest_labels:
        raise ValueError("Manifest does not contain any segmented files.")

    ordered = [label for label in DEFAULT_LABEL_ORDER if label in manifest_labels]
    for label in manifest_labels:
        if label not in ordered:
            ordered.append(label)
    return ordered


def build_long_dataframe(segmentation_dir: Path, manifest: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    label_order = resolve_label_order(manifest)

    for label in label_order:
        file_item = next((item for item in manifest["files"] if str(item["direction"]) == label), None)
        if file_item is None:
            raise ValueError(f"Missing manifest entry for label {label}")

        segments_path = segmentation_dir / Path(str(file_item["segments_path"])).name
        if not segments_path.exists():
            raise FileNotFoundError(f"Missing segments file: {segments_path}")

        segments = np.load(segments_path)
        if segments.ndim != 3 or segments.shape[2] != len(CHANNEL_NAMES):
            raise ValueError(f"Unexpected segment shape for {label}: {segments.shape}")

        for segment_index in range(segments.shape[0]):
            segment_uid = f"{label}_{segment_index:03d}"
            sample_rows.append(
                {
                    "segment_uid": segment_uid,
                    "label": label,
                    "segment_index": int(segment_index),
                }
            )
            for channel_index, channel_name in enumerate(CHANNEL_NAMES):
                values = segments[segment_index, :, channel_index]
                for time_idx, value in enumerate(values):
                    rows.append(
                        {
                            "segment_uid": segment_uid,
                            "label": label,
                            "segment_index": int(segment_index),
                            "channel_name": channel_name,
                            "time_idx": int(time_idx),
                            "value": float(value),
                        }
                    )

    long_df = pd.DataFrame(rows)
    sample_df = pd.DataFrame(sample_rows).sort_values(["label", "segment_index"]).reset_index(drop=True)
    return long_df, sample_df, label_order


def build_fold_splits(sample_df: pd.DataFrame, label_order: list[str], n_splits: int) -> list[FoldSplit]:
    label_to_blocks: dict[str, list[np.ndarray]] = {}
    for label in label_order:
        label_ids = sample_df.loc[sample_df["label"] == label, "segment_uid"].to_numpy()
        if len(label_ids) < n_splits:
            raise ValueError(f"Label {label} has only {len(label_ids)} samples, cannot create {n_splits} folds.")
        label_to_blocks[label] = [np.asarray(block, dtype=object) for block in np.array_split(label_ids, n_splits)]

    splits: list[FoldSplit] = []
    for fold_idx in range(n_splits):
        test_parts = [label_to_blocks[label][fold_idx] for label in label_order]
        train_parts = [
            label_to_blocks[label][block_idx]
            for label in label_order
            for block_idx in range(n_splits)
            if block_idx != fold_idx
        ]
        test_ids = np.concatenate(test_parts)
        train_ids = np.concatenate(train_parts)
        splits.append(FoldSplit(fold=fold_idx, train_ids=train_ids, test_ids=test_ids))
    return splits


def extract_feature_tables(
    long_df: pd.DataFrame,
    sample_df: pd.DataFrame,
    requested_modes: list[str],
    n_jobs: int,
) -> dict[str, pd.DataFrame]:
    settings = EfficientFCParameters()
    sample_index = pd.Index(sample_df["segment_uid"], name="segment_uid")
    feature_tables: dict[str, pd.DataFrame] = {}

    if "fusion" in requested_modes:
        fusion = extract_features(
            long_df[["segment_uid", "channel_name", "time_idx", "value"]],
            column_id="segment_uid",
            column_kind="channel_name",
            column_sort="time_idx",
            column_value="value",
            default_fc_parameters=settings,
            disable_progressbar=True,
            n_jobs=n_jobs,
        )
        impute(fusion)
        feature_tables["fusion"] = fusion.reindex(sample_index)

    if "single" in requested_modes:
        for channel_name in CHANNEL_NAMES:
            single_df = long_df.loc[
                long_df["channel_name"] == channel_name, ["segment_uid", "time_idx", "value"]
            ]
            features = extract_features(
                single_df,
                column_id="segment_uid",
                column_sort="time_idx",
                column_value="value",
                default_fc_parameters=settings,
                disable_progressbar=True,
                n_jobs=n_jobs,
            )
            impute(features)
            feature_tables[channel_name] = features.reindex(sample_index)

    return feature_tables


def build_model_specs(random_state: int) -> dict[str, dict[str, Any]]:
    return {
        "lr": {
            "model": LogisticRegression(
                max_iter=5000,
                solver="lbfgs",
                random_state=random_state,
            ),
            "needs_scaling": True,
        },
        "rf": {
            "model": RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=1,
                random_state=random_state,
                n_jobs=-1,
            ),
            "needs_scaling": False,
        },
        "knn": {
            "model": KNeighborsClassifier(n_neighbors=5),
            "needs_scaling": True,
        },
        "dt": {
            "model": DecisionTreeClassifier(random_state=random_state),
            "needs_scaling": False,
        },
    }


def select_training_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train_features = X_train.copy()
    test_features = X_test.copy()

    non_constant_cols = train_features.columns[train_features.nunique(dropna=False) > 1]
    if len(non_constant_cols) == 0:
        raise ValueError("All extracted features are constant in the training split.")

    train_features = train_features.loc[:, non_constant_cols]
    test_features = test_features.loc[:, non_constant_cols]

    try:
        from tsfresh import select_features

        selected_train = select_features(train_features, y_train)
        selected_columns = list(selected_train.columns)
    except Exception:
        selected_columns = list(train_features.columns)

    if not selected_columns:
        selected_columns = list(train_features.columns)

    return (
        train_features.loc[:, selected_columns],
        test_features.loc[:, selected_columns],
        selected_columns,
    )


def build_estimator(model_name: str, specs: dict[str, dict[str, Any]]) -> Pipeline:
    spec = specs[model_name]
    steps: list[tuple[str, Any]] = [("variance", VarianceThreshold())]
    if spec["needs_scaling"]:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", clone(spec["model"])))
    return Pipeline(steps)


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def summarize_values(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=0))


def evaluate_feature_model(
    feature_set: str,
    features: pd.DataFrame,
    sample_df: pd.DataFrame,
    splits: list[FoldSplit],
    model_name: str,
    specs: dict[str, dict[str, Any]],
    label_order: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fold_rows: list[dict[str, Any]] = []
    confusion_total = np.zeros((len(label_order), len(label_order)), dtype=int)
    selected_feature_counts: list[int] = []

    labels = sample_df.set_index("segment_uid").loc[features.index, "label"]

    for split in splits:
        train_ids = [item for item in split.train_ids if item in features.index]
        test_ids = [item for item in split.test_ids if item in features.index]

        if not train_ids or not test_ids:
            raise ValueError(f"Fold {split.fold} has empty train/test ids for feature set {feature_set}.")

        X_train_raw = features.loc[train_ids]
        X_test_raw = features.loc[test_ids]
        y_train = labels.loc[train_ids]
        y_test = labels.loc[test_ids]

        X_train, X_test, selected_columns = select_training_features(X_train_raw, y_train, X_test_raw)
        estimator = build_estimator(model_name, specs)
        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_test)

        metrics = compute_metrics(y_test, y_pred)
        confusion = confusion_matrix(y_test, y_pred, labels=label_order)
        confusion_total += confusion
        selected_feature_counts.append(len(selected_columns))

        fold_rows.append(
            {
                "feature_set": feature_set,
                "model": model_name,
                "fold": split.fold,
                "balanced_accuracy": metrics["balanced_accuracy"],
                "macro_f1": metrics["macro_f1"],
                "accuracy": metrics["accuracy"],
                "n_train": int(len(train_ids)),
                "n_test": int(len(test_ids)),
                "selected_feature_count": int(len(selected_columns)),
                "status": "ok",
                "error": "",
            }
        )

    summary: dict[str, Any] = {
        "feature_set": feature_set,
        "model": model_name,
        "status": "ok",
        "completed_folds": len(fold_rows),
        "expected_folds": len(splits),
        "selected_feature_count_mean": float(np.mean(selected_feature_counts)) if selected_feature_counts else None,
        "selected_feature_count_std": float(np.std(selected_feature_counts)) if selected_feature_counts else None,
        "confusion_matrix": confusion_total,
    }
    for metric_name in ["balanced_accuracy", "macro_f1", "accuracy"]:
        mean_value, std_value = summarize_values([row[metric_name] for row in fold_rows])
        summary[f"{metric_name}_mean"] = mean_value
        summary[f"{metric_name}_std"] = std_value
    return fold_rows, summary


def evaluate_permutation_baseline(
    features: pd.DataFrame,
    sample_df: pd.DataFrame,
    splits: list[FoldSplit],
    model_name: str,
    specs: dict[str, dict[str, Any]],
    random_state: int,
    runs: int,
) -> dict[str, float | int | None]:
    if runs <= 0:
        return {
            "n_permutations": 0,
            "perm_balanced_accuracy_mean": None,
            "perm_balanced_accuracy_std": None,
            "perm_macro_f1_mean": None,
            "perm_macro_f1_std": None,
            "perm_accuracy_mean": None,
            "perm_accuracy_std": None,
        }

    rng = np.random.default_rng(random_state)
    labels = sample_df.set_index("segment_uid").loc[features.index, "label"]
    per_run_scores = {"balanced_accuracy": [], "macro_f1": [], "accuracy": []}

    for _ in range(runs):
        shuffled = pd.Series(rng.permutation(labels.to_numpy()), index=labels.index)
        run_scores = {"balanced_accuracy": [], "macro_f1": [], "accuracy": []}
        for split in splits:
            train_ids = [item for item in split.train_ids if item in features.index]
            test_ids = [item for item in split.test_ids if item in features.index]
            X_train_raw = features.loc[train_ids]
            X_test_raw = features.loc[test_ids]
            y_train = shuffled.loc[train_ids]
            y_test = shuffled.loc[test_ids]

            X_train, X_test, _ = select_training_features(X_train_raw, y_train, X_test_raw)
            estimator = build_estimator(model_name, specs)
            estimator.fit(X_train, y_train)
            y_pred = estimator.predict(X_test)
            metrics = compute_metrics(y_test, y_pred)
            for metric_name, value in metrics.items():
                run_scores[metric_name].append(value)

        for metric_name, values in run_scores.items():
            per_run_scores[metric_name].append(float(np.mean(values)))

    result: dict[str, float | int | None] = {"n_permutations": int(runs)}
    for metric_name, values in per_run_scores.items():
        mean_value, std_value = summarize_values(values)
        result[f"perm_{metric_name}_mean"] = mean_value
        result[f"perm_{metric_name}_std"] = std_value
    return result


def plot_confusion_matrix(
    confusion: np.ndarray,
    labels: list[str],
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(confusion, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    max_value = confusion.max() if confusion.size else 0
    threshold = max_value / 2.0 if max_value else 0.0
    for row_idx in range(confusion.shape[0]):
        for col_idx in range(confusion.shape[1]):
            value = int(confusion[row_idx, col_idx])
            color = "white" if value > threshold else "black"
            ax.text(col_idx, row_idx, str(value), ha="center", va="center", color=color)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_comparison_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    successful = summary_df.loc[summary_df["status"] == "ok"].copy()
    if successful.empty:
        return pd.DataFrame(
            columns=[
                "model",
                "fusion_balanced_accuracy_mean",
                "best_single_channel",
                "best_single_balanced_accuracy_mean",
                "fusion_minus_best_single",
            ]
        )

    rows: list[dict[str, Any]] = []
    for model_name, model_rows in successful.groupby("model"):
        fusion_rows = model_rows.loc[model_rows["feature_set"] == "fusion"]
        single_rows = model_rows.loc[model_rows["feature_set"].isin(CHANNEL_NAMES)]

        fusion_value = None if fusion_rows.empty else float(fusion_rows.iloc[0]["balanced_accuracy_mean"])
        best_single_row = None if single_rows.empty else single_rows.sort_values(
            "balanced_accuracy_mean", ascending=False
        ).iloc[0]
        best_single_value = None if best_single_row is None else float(best_single_row["balanced_accuracy_mean"])
        delta = None
        if fusion_value is not None and best_single_value is not None:
            delta = float(fusion_value - best_single_value)

        rows.append(
            {
                "model": model_name,
                "fusion_balanced_accuracy_mean": fusion_value,
                "best_single_channel": None if best_single_row is None else str(best_single_row["feature_set"]),
                "best_single_balanced_accuracy_mean": best_single_value,
                "fusion_minus_best_single": delta,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    requested_modes = validate_requested_items(args.feature_mode, {"single", "fusion"}, "feature modes")
    requested_models = validate_requested_items(args.models, {"lr", "rf", "knn", "dt"}, "models")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(args.segmentation_dir)
    long_df, sample_df, label_order = build_long_dataframe(args.segmentation_dir, manifest)
    splits = build_fold_splits(sample_df, label_order, args.n_splits)
    feature_tables = extract_feature_tables(long_df, sample_df, requested_modes, args.n_jobs)
    model_specs = build_model_specs(args.random_state)

    fold_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []

    for feature_set, features in feature_tables.items():
        for model_name in requested_models:
            try:
                feature_fold_rows, summary = evaluate_feature_model(
                    feature_set=feature_set,
                    features=features,
                    sample_df=sample_df,
                    splits=splits,
                    model_name=model_name,
                    specs=model_specs,
                    label_order=label_order,
                )
                fold_rows.extend(feature_fold_rows)

                baseline = evaluate_permutation_baseline(
                    features=features,
                    sample_df=sample_df,
                    splits=splits,
                    model_name=model_name,
                    specs=model_specs,
                    random_state=args.random_state,
                    runs=args.permutation_runs,
                )
                summary.update(baseline)
                confusion = summary.pop("confusion_matrix")
                plot_confusion_matrix(
                    confusion=confusion,
                    labels=label_order,
                    title=f"{feature_set} + {model_name}",
                    output_path=output_dir / f"confusion_matrix_{feature_set}_{model_name}.png",
                )
                summary_rows.append(summary)
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                error_rows.append(
                    {
                        "feature_set": feature_set,
                        "model": model_name,
                        "error": message,
                        "traceback": traceback.format_exc(),
                    }
                )
                summary_rows.append(
                    {
                        "feature_set": feature_set,
                        "model": model_name,
                        "status": "failed",
                        "completed_folds": 0,
                        "expected_folds": len(splits),
                        "balanced_accuracy_mean": None,
                        "balanced_accuracy_std": None,
                        "macro_f1_mean": None,
                        "macro_f1_std": None,
                        "accuracy_mean": None,
                        "accuracy_std": None,
                        "selected_feature_count_mean": None,
                        "selected_feature_count_std": None,
                        "n_permutations": args.permutation_runs,
                        "perm_balanced_accuracy_mean": None,
                        "perm_balanced_accuracy_std": None,
                        "perm_macro_f1_mean": None,
                        "perm_macro_f1_std": None,
                        "perm_accuracy_mean": None,
                        "perm_accuracy_std": None,
                        "error": message,
                    }
                )

    fold_df = pd.DataFrame(fold_rows)
    if not fold_df.empty:
        fold_df = fold_df.sort_values(["feature_set", "model", "fold"]).reset_index(drop=True)
    summary_df = pd.DataFrame(summary_rows).sort_values(["feature_set", "model"]).reset_index(drop=True)
    comparison_df = build_comparison_table(summary_df)

    fold_df.to_csv(output_dir / "fold_metrics.csv", index=False)
    summary_df.to_csv(output_dir / "metrics_summary.csv", index=False)
    comparison_df.to_csv(output_dir / "single_channel_vs_fusion.csv", index=False)
    pd.DataFrame(error_rows).to_csv(output_dir / "errors.csv", index=False)

    run_config = {
        "segmentation_dir": str(args.segmentation_dir),
        "output_dir": str(output_dir),
        "n_splits": int(args.n_splits),
        "random_state": int(args.random_state),
        "feature_mode": requested_modes,
        "models": requested_models,
        "permutation_runs": int(args.permutation_runs),
        "n_jobs": int(args.n_jobs),
        "label_order": label_order,
        "random_baseline_accuracy": 0.25,
        "assumptions": [
            "Current evaluation validates within-session segment-level separability only.",
            "Segmentation inputs come from *_segments.npy; legacy channel_samples artifacts are ignored.",
            "Feature selection and scaling are fit inside each training fold only.",
        ],
        "samples": {
            "total_segments": int(len(sample_df)),
            "per_label": sample_df.groupby("label").size().to_dict(),
        },
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
