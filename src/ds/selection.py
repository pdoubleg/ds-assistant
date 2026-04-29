"""Feature screening and ranking helpers for the standalone ds package."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .config import (
    BatchedFeatureScreenConfig,
    BatchedFeatureScreenResult,
    CorrelationAnalysisResult,
    FeatureRankingResult,
    FeatureScreenConfig,
    FeatureScreenResult,
    FeatureSubsetRankingResult,
    SplitConfig,
    TrainConfig,
)
from .io import cleanup_memory, read_parquet_fragment
from .modeling import (
    _is_numeric_dtype,
    build_train_valid_frames,
    fit_lightgbm_binary,
    infer_categorical_columns,
    prepare_lightgbm_train_valid_frames,
)
from .metrics import fast_auc_score


def _resolve_candidate_columns(
    dataframe: pd.DataFrame,
    *,
    target_column: str | None,
    id_columns: list[str] | None,
    feature_columns: list[str] | None = None,
) -> list[str]:
    """Resolve candidate feature columns in dataframe order.

    Args:
        dataframe: Source dataframe.
        target_column: Optional target column excluded from candidates.
        id_columns: Optional identifier columns excluded from candidates.
        feature_columns: Optional explicit feature subset.

    Returns:
        Ordered candidate feature column names.
    """

    if feature_columns is not None:
        missing_columns = [
            column for column in feature_columns if column not in dataframe.columns
        ]
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}.")
        return [column for column in feature_columns if column in dataframe.columns]

    excluded = set(id_columns or [])
    if target_column is not None:
        excluded.add(target_column)
    return [str(column) for column in dataframe.columns if str(column) not in excluded]


def _build_selected_dataframe(
    dataframe: pd.DataFrame,
    *,
    target_column: str | None,
    id_columns: list[str] | None,
    selected_features: list[str],
) -> pd.DataFrame:
    """Build a reduced dataframe from selected features and preserved columns.

    Args:
        dataframe: Source dataframe.
        target_column: Optional target column retained in the output.
        id_columns: Optional identifier columns retained in the output.
        selected_features: Selected feature columns.

    Returns:
        Reduced dataframe copy.
    """

    selected_columns: list[str] = []
    if target_column is not None and target_column in dataframe.columns:
        selected_columns.append(target_column)
    for column in list(id_columns or []):
        if column in dataframe.columns and column not in selected_columns:
            selected_columns.append(column)
    for column in selected_features:
        if column in dataframe.columns and column not in selected_columns:
            selected_columns.append(column)
    return dataframe[selected_columns].copy()


def _unique_in_order(columns: list[str]) -> list[str]:
    """Return columns with duplicates removed while preserving order.

    Args:
        columns: Input column names.

    Returns:
        De-duplicated column names.
    """

    seen: set[str] = set()
    unique_columns: list[str] = []
    for column in columns:
        if column in seen:
            continue
        seen.add(column)
        unique_columns.append(column)
    return unique_columns


def _split_feature_batches(
    feature_columns: list[str], batch_size: int
) -> list[list[str]]:
    """Split feature columns into deterministic order-preserving batches.

    Args:
        feature_columns: Candidate feature columns.
        batch_size: Maximum number of columns in each batch.

    Returns:
        Ordered feature batches.
    """

    if batch_size <= 0:
        raise ValueError("`batch_size` must be greater than zero.")
    return [
        feature_columns[index : index + batch_size]
        for index in range(0, len(feature_columns), batch_size)
    ]


def _projection_columns(
    *,
    target_column: str,
    id_columns: list[str] | None,
    feature_columns: list[str],
) -> list[str]:
    """Build the projected columns required for one narrow load.

    Args:
        target_column: Target column that must be included.
        id_columns: Optional identifier columns.
        feature_columns: Feature columns for the current batch.

    Returns:
        Ordered projected columns with duplicates removed.
    """

    return _unique_in_order([target_column, *list(id_columns or []), *feature_columns])


def _persist_finding_rows(
    rows: list[dict[str, Any]],
    *,
    output_dir: Path | None,
    batch_index: int,
    file_format: str,
) -> Path | None:
    """Persist batch finding rows if an output directory was configured.

    Args:
        rows: Finding rows to persist.
        output_dir: Optional destination directory.
        batch_index: One-based batch index.
        file_format: Either ``"json"`` or ``"csv"``.

    Returns:
        Path to the persisted file, or ``None`` when persistence is disabled.
    """

    if output_dir is None:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    if file_format == "json":
        path = output_dir / f"batch_{batch_index:04d}_findings.json"
        path.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
        return path
    if file_format == "csv":
        path = output_dir / f"batch_{batch_index:04d}_findings.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        return path
    raise ValueError("`persisted_file_format` must be either 'json' or 'csv'.")


def _choose_correlated_feature_drops(
    *,
    candidate_columns: list[str],
    flagged_pairs: list[dict[str, Any]],
    feature_scores: dict[str, float],
    missing_rates: dict[str, float],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Apply deterministic drop decisions to correlated feature pairs.

    Args:
        candidate_columns: Ordered candidate numeric columns.
        flagged_pairs: Correlated pairs sorted or unsorted.
        feature_scores: Univariate target score by feature.
        missing_rates: Missing rate by feature.

    Returns:
        Tuple of pair diagnostics and dropped feature columns.
    """

    feature_order = {column: index for index, column in enumerate(candidate_columns)}
    flagged_pairs.sort(
        key=lambda row: (
            -float(row["abs_correlation"]),
            feature_order[str(row["left_feature"])],
            feature_order[str(row["right_feature"])],
        )
    )

    pair_rows: list[dict[str, Any]] = []
    drop_columns: list[str] = []
    dropped_set: set[str] = set()

    for row in flagged_pairs:
        left_column = str(row["left_feature"])
        right_column = str(row["right_feature"])

        if left_column in dropped_set or right_column in dropped_set:
            row["decision"] = "skipped_already_dropped"
            pair_rows.append(row)
            continue

        left_score = feature_scores.get(left_column)
        right_score = feature_scores.get(right_column)
        drop_column = right_column
        keep_column = left_column
        drop_reason = "later_column_order"

        if left_score is not None and right_score is not None:
            if left_score < right_score:
                drop_column = left_column
                keep_column = right_column
                drop_reason = "lower_target_score"
            elif right_score < left_score:
                drop_column = right_column
                keep_column = left_column
                drop_reason = "lower_target_score"
            elif missing_rates[left_column] > missing_rates[right_column]:
                drop_column = left_column
                keep_column = right_column
                drop_reason = "higher_missing_rate"
            elif missing_rates[right_column] > missing_rates[left_column]:
                drop_column = right_column
                keep_column = left_column
                drop_reason = "higher_missing_rate"
        elif missing_rates[left_column] > missing_rates[right_column]:
            drop_column = left_column
            keep_column = right_column
            drop_reason = "higher_missing_rate"
        elif missing_rates[right_column] > missing_rates[left_column]:
            drop_column = right_column
            keep_column = left_column
            drop_reason = "higher_missing_rate"

        row["left_score"] = left_score
        row["right_score"] = right_score
        row["kept_feature"] = keep_column
        row["dropped_feature"] = drop_column
        row["drop_reason"] = drop_reason
        row["decision"] = "dropped"
        pair_rows.append(row)
        drop_columns.append(drop_column)
        dropped_set.add(drop_column)

    return pair_rows, drop_columns


def _blockwise_numeric_correlation_pairs(
    *,
    load_columns: Callable[[list[str]], pd.DataFrame],
    target_column: str,
    id_columns: list[str] | None,
    numeric_columns: list[str],
    threshold: float,
    correlation_batch_size: int,
    cleanup_batch_frames: bool,
) -> list[dict[str, Any]]:
    """Find high-correlation numeric pairs from projected column blocks.

    Args:
        load_columns: Callback that loads the requested columns.
        target_column: Target column included in projections for loader
            consistency.
        id_columns: Optional identifier columns included in projections.
        numeric_columns: Ordered numeric columns to compare.
        threshold: Absolute Pearson correlation threshold.
        correlation_batch_size: Maximum columns per numeric block.
        cleanup_batch_frames: Whether to run explicit cleanup on loaded blocks.

    Returns:
        Correlated feature-pair rows.
    """

    if correlation_batch_size <= 0:
        raise ValueError("`correlation_batch_size` must be greater than zero.")

    flagged_pairs: list[dict[str, Any]] = []
    numeric_blocks = _split_feature_batches(numeric_columns, correlation_batch_size)
    for left_block_index, left_columns in enumerate(numeric_blocks):
        left_frame = load_columns(
            _projection_columns(
                target_column=target_column,
                id_columns=id_columns,
                feature_columns=left_columns,
            )
        )
        left_numeric = left_frame[left_columns].apply(pd.to_numeric, errors="coerce")

        for right_block_index in range(left_block_index, len(numeric_blocks)):
            right_columns = numeric_blocks[right_block_index]
            if right_block_index == left_block_index:
                combined_numeric = left_numeric
            else:
                right_frame = load_columns(
                    _projection_columns(
                        target_column=target_column,
                        id_columns=id_columns,
                        feature_columns=right_columns,
                    )
                )
                right_numeric = right_frame[right_columns].apply(
                    pd.to_numeric, errors="coerce"
                )
                combined_numeric = pd.concat([left_numeric, right_numeric], axis=1)

            corr = combined_numeric.corr(method="pearson").abs().fillna(0.0)
            for left_offset, left_column in enumerate(left_columns):
                right_start = (
                    left_offset + 1 if right_block_index == left_block_index else 0
                )
                for right_column in right_columns[right_start:]:
                    corr_value = float(corr.loc[left_column, right_column])
                    if corr_value >= threshold:
                        flagged_pairs.append(
                            {
                                "left_feature": left_column,
                                "right_feature": right_column,
                                "abs_correlation": corr_value,
                            }
                        )

            if right_block_index != left_block_index and cleanup_batch_frames:
                cleanup_memory(right_frame, aggressive=True)

        if cleanup_batch_frames:
            cleanup_memory(left_frame, aggressive=True)

    return flagged_pairs


def univariate_numeric_score(feature: pd.Series, target: pd.Series) -> float:
    """Return a numeric screening score based on absolute AUC lift.

    Args:
        feature: Candidate numeric feature.
        target: Binary target series.

    Returns:
        Absolute AUC lift above random performance.
    """

    valid = feature.notna() & target.notna()
    if valid.sum() < 100:
        return 0.0
    feature_values = feature[valid].astype(float).values
    target_values = target[valid].astype(int).values
    if pd.Series(feature_values).std(ddof=0) == 0:
        return 0.0
    try:
        auc = fast_auc_score(target_values, feature_values)
        return float(abs(auc - 0.5))
    except Exception:
        return 0.0


def univariate_categorical_score(
    feature: pd.Series,
    target: pd.Series,
    *,
    min_count: int = 25,
) -> float:
    """Return an OOF target-encoded screening score for a categorical feature.

    Args:
        feature: Candidate categorical feature.
        target: Binary target series.
        min_count: Minimum category count before a category meaningfully departs
            from the global mean.

    Returns:
        Absolute AUC lift above random performance using out-of-fold target
        encoding.
    """

    valid = feature.notna() & target.notna()
    if valid.sum() < 100:
        return 0.0

    feature_values = feature[valid].astype(str).reset_index(drop=True)
    target_values = (
        pd.to_numeric(target[valid], errors="coerce").astype(int).reset_index(drop=True)
    )
    class_counts = target_values.value_counts()
    if len(class_counts) < 2 or int(class_counts.min()) < 2:
        return 0.0

    overall_rate = float(target_values.mean())
    if overall_rate <= 0.0 or overall_rate >= 1.0:
        return 0.0

    n_splits = min(5, int(class_counts.min()))
    if n_splits < 2:
        return 0.0

    encoded = np.full(len(feature_values), overall_rate, dtype=float)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    smoothing = float(max(min_count, 1))

    for train_idx, valid_idx in splitter.split(feature_values, target_values):
        fold_feature_train = feature_values.iloc[train_idx]
        fold_target_train = target_values.iloc[train_idx]
        fold_feature_valid = feature_values.iloc[valid_idx]

        category_stats = (
            pd.DataFrame({"feature": fold_feature_train, "target": fold_target_train})
            .groupby("feature", observed=True)
            .agg(target_sum=("target", "sum"), count=("target", "size"))
        )
        category_mean = category_stats["target_sum"] / category_stats["count"]
        category_weight = category_stats["count"] / (
            category_stats["count"] + smoothing
        )
        smoothed_encoding = overall_rate * (1.0 - category_weight) + category_mean * (
            category_weight
        )
        encoded[valid_idx] = fold_feature_valid.map(smoothed_encoding).fillna(
            overall_rate
        )

    if np.nanstd(encoded) == 0.0:
        return 0.0

    auc = fast_auc_score(target_values.to_numpy(), encoded)
    return float(abs(auc - 0.5))


def screen_features(
    dataframe: pd.DataFrame,
    *,
    target_column: str,
    id_columns: list[str] | None = None,
    feature_columns: list[str] | None = None,
    config: FeatureScreenConfig | None = None,
) -> FeatureScreenResult:
    """Screen candidate features with descriptive and univariate filters.

    Args:
        dataframe: Source dataframe.
        target_column: Binary target column used for screening.
        id_columns: Optional identifier columns excluded from the candidate set.
        feature_columns: Optional explicit feature subset.
        config: Screening configuration override.

    Returns:
        Structured feature screening result.
    """

    if target_column not in dataframe.columns:
        raise ValueError(f"Target column {target_column!r} was not found.")

    config = config or FeatureScreenConfig()
    candidate_columns = _resolve_candidate_columns(
        dataframe,
        target_column=target_column,
        id_columns=id_columns,
        feature_columns=feature_columns,
    )
    target = (
        pd.to_numeric(dataframe[target_column], errors="coerce").fillna(0).astype(int)
    )
    findings: list[dict[str, Any]] = []
    warnings: list[str] = []

    for column in candidate_columns:
        series = dataframe[column]
        non_null_count = int(series.notna().sum())
        missing_rate = float(series.isna().mean())
        unique_count = int(series.nunique(dropna=True))
        dominant_rate = (
            float(series.value_counts(dropna=False, normalize=True).iloc[0])
            if not series.empty
            else 0.0
        )

        keep = True
        reasons: list[str] = []
        if non_null_count < config.min_non_null:
            keep = False
            reasons.append("too_few_non_null")
        if missing_rate > config.max_missing_frac:
            keep = False
            reasons.append("too_missing")
        if unique_count <= 1:
            keep = False
            reasons.append("zero_variance")
        if dominant_rate >= config.near_constant_thresh:
            keep = False
            reasons.append("near_constant")

        score = 0.0
        if keep:
            score = (
                univariate_numeric_score(series, target)
                if _is_numeric_dtype(series)
                else univariate_categorical_score(series, target)
            )

        findings.append(
            {
                "feature": column,
                "dtype": str(series.dtype),
                "missing_rate": missing_rate,
                "unique_count": unique_count,
                "dominant_rate": dominant_rate,
                "score": float(score),
                "kept": keep,
                "reasons": reasons,
            }
        )

    kept_rows = [row for row in findings if row["kept"]]
    kept_rows.sort(key=lambda row: row["score"], reverse=True)
    selected_features = [
        str(row["feature"]) for row in kept_rows[: config.top_k_univariate]
    ]
    selected_feature_set = set(selected_features)
    dropped_features = [
        str(row["feature"])
        for row in findings
        if str(row["feature"]) not in selected_feature_set
    ]
    for row in findings:
        feature_name = str(row["feature"])
        row["kept"] = feature_name in selected_feature_set
        if feature_name in dropped_features and "top_k_filtered" not in row["reasons"]:
            if not row["reasons"]:
                row["reasons"].append("top_k_filtered")
            elif row["score"] > 0.0:
                row["reasons"].append("top_k_filtered")
    categorical_columns = infer_categorical_columns(dataframe, selected_features)
    filtered_df = _build_selected_dataframe(
        dataframe,
        target_column=target_column,
        id_columns=id_columns,
        selected_features=selected_features,
    )
    return FeatureScreenResult(
        filtered_df=filtered_df,
        selected_columns=selected_features,
        dropped_columns=dropped_features,
        categorical_columns=categorical_columns,
        findings=findings,
        metrics={
            "candidate_feature_count": len(candidate_columns),
            "selected_feature_count": len(selected_features),
            "dropped_feature_count": len(candidate_columns) - len(selected_features),
        },
        warnings=warnings,
    )


def screen_feature_batches(
    *,
    feature_columns: list[str],
    target_column: str,
    load_columns: Callable[[list[str]], pd.DataFrame],
    id_columns: list[str] | None = None,
    config: BatchedFeatureScreenConfig | None = None,
) -> BatchedFeatureScreenResult:
    """Screen and de-correlate features from projected column batches.

    Args:
        feature_columns: Ordered candidate feature columns.
        target_column: Binary target column used for screening.
        load_columns: Callback that returns a dataframe for requested columns.
        id_columns: Optional identifier columns loaded with each screening batch.
        config: Batched screening configuration.

    Returns:
        Batched screening result with persisted intermediate paths and final
        selected feature names.

    Example:
        >>> frame = pd.DataFrame({"target": [0, 1], "x": [0.1, 0.9]})
        >>> result = screen_feature_batches(
        ...     feature_columns=["x"],
        ...     target_column="target",
        ...     load_columns=lambda columns: frame[columns].copy(),
        ...     config=BatchedFeatureScreenConfig(
        ...         batch_size=1,
        ...         screen_config=FeatureScreenConfig(min_non_null=1),
        ...     ),
        ... )
        >>> result.selected_columns
        ['x']
    """

    config = config or BatchedFeatureScreenConfig()
    if not feature_columns:
        raise ValueError("At least one feature column is required.")
    if config.batch_size <= 0:
        raise ValueError("`batch_size` must be greater than zero.")

    ordered_features = _unique_in_order([str(column) for column in feature_columns])
    feature_order = {column: index for index, column in enumerate(ordered_features)}
    batches = _split_feature_batches(ordered_features, config.batch_size)
    finding_rows: list[dict[str, Any]] = []
    finding_paths: list[Path] = []
    batch_rows: list[dict[str, Any]] = []
    warnings: list[str] = []

    for batch_index, batch_features in enumerate(batches, start=1):
        projected_columns = _projection_columns(
            target_column=target_column,
            id_columns=id_columns,
            feature_columns=batch_features,
        )
        batch_frame = load_columns(projected_columns)
        missing_columns = [
            column for column in projected_columns if column not in batch_frame.columns
        ]
        if missing_columns:
            raise ValueError(
                f"Loaded batch is missing columns: {', '.join(missing_columns)}."
            )

        # Keep all columns that pass descriptive filters within the batch; global
        # top-k filtering happens after every batch has contributed a score.
        batch_screen_config = replace(
            config.screen_config,
            top_k_univariate=max(
                config.screen_config.top_k_univariate, len(batch_features)
            ),
        )
        screen_result = screen_features(
            batch_frame,
            target_column=target_column,
            id_columns=id_columns,
            feature_columns=batch_features,
            config=batch_screen_config,
        )

        numeric_flags = {
            column: _is_numeric_dtype(batch_frame[column])
            for column in batch_features
            if column in batch_frame.columns
        }
        batch_findings: list[dict[str, Any]] = []
        for row in screen_result.findings:
            feature_name = str(row["feature"])
            enriched_row = {
                **row,
                "feature": feature_name,
                "batch_index": batch_index,
                "feature_index": feature_order[feature_name],
                "is_numeric": bool(numeric_flags.get(feature_name, False)),
            }
            batch_findings.append(enriched_row)
            finding_rows.append(enriched_row)

        path = _persist_finding_rows(
            batch_findings,
            output_dir=config.output_dir,
            batch_index=batch_index,
            file_format=config.persisted_file_format,
        )
        if path is not None:
            finding_paths.append(path)

        batch_rows.append(
            {
                "batch_index": batch_index,
                "requested_columns": projected_columns,
                "input_feature_count": len(batch_features),
                "selected_feature_count": len(screen_result.selected_columns),
                "dropped_feature_count": len(screen_result.dropped_columns),
                "finding_path": str(path) if path is not None else None,
            }
        )
        warnings.extend(screen_result.warnings)
        if config.cleanup_batch_frames:
            cleanup_memory(batch_frame, aggressive=True)

    preselected_rows = [row for row in finding_rows if bool(row["kept"])]
    preselected_rows.sort(
        key=lambda row: (-float(row["score"]), int(row["feature_index"]))
    )
    top_k = config.top_k_univariate or config.screen_config.top_k_univariate
    globally_selected = [str(row["feature"]) for row in preselected_rows[:top_k]]
    globally_selected_set = set(globally_selected)

    for row in finding_rows:
        feature_name = str(row["feature"])
        if feature_name in globally_selected_set:
            continue
        row["kept"] = False
        if not row["reasons"] or float(row["score"]) > 0.0:
            if "top_k_filtered" not in row["reasons"]:
                row["reasons"].append("top_k_filtered")

    numeric_columns = [
        feature
        for feature in globally_selected
        if any(
            str(row["feature"]) == feature and bool(row["is_numeric"])
            for row in finding_rows
        )
    ]
    feature_scores = {str(row["feature"]): float(row["score"]) for row in finding_rows}
    missing_rates = {
        str(row["feature"]): float(row["missing_rate"]) for row in finding_rows
    }

    correlation_pair_rows: list[dict[str, Any]] = []
    correlation_drop_columns: list[str] = []
    if len(numeric_columns) >= 2:
        flagged_pairs = _blockwise_numeric_correlation_pairs(
            load_columns=load_columns,
            target_column=target_column,
            id_columns=id_columns,
            numeric_columns=numeric_columns,
            threshold=config.correlation_threshold,
            correlation_batch_size=config.correlation_batch_size,
            cleanup_batch_frames=config.cleanup_batch_frames,
        )
        correlation_pair_rows, correlation_drop_columns = (
            _choose_correlated_feature_drops(
                candidate_columns=numeric_columns,
                flagged_pairs=flagged_pairs,
                feature_scores=feature_scores,
                missing_rates=missing_rates,
            )
        )
    else:
        warnings.append(
            "Skipped numeric correlation pruning because fewer than two numeric features survived screening."
        )

    correlation_drop_set = set(correlation_drop_columns)
    final_selected_columns = [
        feature for feature in globally_selected if feature not in correlation_drop_set
    ]
    final_selected_set = set(final_selected_columns)
    dropped_columns = [
        feature for feature in ordered_features if feature not in final_selected_set
    ]
    categorical_columns = [
        feature
        for feature in final_selected_columns
        if not any(
            str(row["feature"]) == feature and bool(row["is_numeric"])
            for row in finding_rows
        )
    ]

    for row in finding_rows:
        feature_name = str(row["feature"])
        row["kept"] = feature_name in final_selected_set
        if (
            feature_name in correlation_drop_set
            and "correlated_numeric" not in row["reasons"]
        ):
            row["reasons"].append("correlated_numeric")

    return BatchedFeatureScreenResult(
        selected_columns=final_selected_columns,
        dropped_columns=dropped_columns,
        categorical_columns=categorical_columns,
        batch_rows=batch_rows,
        finding_rows=finding_rows,
        finding_paths=finding_paths,
        correlation_pair_rows=correlation_pair_rows,
        metrics={
            "candidate_feature_count": len(ordered_features),
            "batch_count": len(batches),
            "pre_correlation_selected_feature_count": len(globally_selected),
            "selected_feature_count": len(final_selected_columns),
            "dropped_feature_count": len(dropped_columns),
            "numeric_correlation_feature_count": len(numeric_columns),
            "correlated_pair_count": len(correlation_pair_rows),
        },
        warnings=warnings,
    )


def screen_parquet_feature_batches(
    uri: str | Path,
    *,
    target_column: str,
    feature_columns: list[str],
    id_columns: list[str] | None = None,
    config: BatchedFeatureScreenConfig | None = None,
    read_kwargs: dict[str, Any] | None = None,
) -> BatchedFeatureScreenResult:
    """Run batched feature screening from a parquet dataset.

    Args:
        uri: Local parquet path or remote URI such as ``s3://bucket/dataset``.
        target_column: Binary target column used for screening.
        feature_columns: Ordered candidate feature columns.
        id_columns: Optional identifier columns loaded with each batch.
        config: Batched screening configuration.
        read_kwargs: Additional keyword arguments forwarded to
            ``read_parquet_fragment``.

    Returns:
        Batched feature screening result.

    Example:
        >>> result = screen_parquet_feature_batches(
        ...     "train.parquet",
        ...     target_column="target",
        ...     feature_columns=["age", "balance"],
        ... )
    """

    config = config or BatchedFeatureScreenConfig()
    read_options = dict(read_kwargs or {})
    read_options.setdefault("random_seed", config.random_seed)
    read_options.pop("columns", None)

    def _load_columns(columns: list[str]) -> pd.DataFrame:
        loaded = read_parquet_fragment(uri, columns=columns, **read_options)
        if isinstance(loaded, tuple):
            return loaded[0]
        return loaded

    return screen_feature_batches(
        feature_columns=feature_columns,
        target_column=target_column,
        load_columns=_load_columns,
        id_columns=id_columns,
        config=config,
    )


def analyze_feature_correlation(
    dataframe: pd.DataFrame,
    *,
    target_column: str | None = None,
    id_columns: list[str] | None = None,
    feature_columns: list[str] | None = None,
    threshold: float = 0.95,
    max_numeric_features: int = 300,
) -> CorrelationAnalysisResult:
    """Find highly correlated numeric features and suggest deterministic drops.

    Args:
        dataframe: Source dataframe.
        target_column: Optional target column excluded from the candidate set.
        id_columns: Optional identifier columns excluded from the candidate set.
        feature_columns: Optional explicit feature subset.
        threshold: Absolute correlation threshold used to flag pairs.
        max_numeric_features: Maximum numeric columns considered in one pass.

    Returns:
        Structured correlation analysis result.
    """

    candidate_columns = _resolve_candidate_columns(
        dataframe,
        target_column=target_column,
        id_columns=id_columns,
        feature_columns=feature_columns,
    )
    numeric_columns = [
        column for column in candidate_columns if _is_numeric_dtype(dataframe[column])
    ]
    warnings: list[str] = []
    if len(numeric_columns) > max_numeric_features:
        warnings.append(
            "Too many numeric features for one correlation pass; used the first "
            f"{max_numeric_features} numeric columns in dataframe order."
        )
        numeric_columns = numeric_columns[:max_numeric_features]
    if len(numeric_columns) < 2:
        raise ValueError("At least two numeric feature columns are required.")

    numeric_frame = dataframe[numeric_columns].apply(pd.to_numeric, errors="coerce")
    corr = numeric_frame.corr(method="pearson").abs().fillna(0.0)
    missing_rates = {
        column: float(dataframe[column].isna().mean()) for column in numeric_columns
    }
    feature_scores: dict[str, float] = {}
    if target_column is not None and target_column in dataframe.columns:
        target = (
            pd.to_numeric(dataframe[target_column], errors="coerce")
            .fillna(0)
            .astype(int)
        )
        # Use the same lightweight univariate score as the screening pass so
        # correlated-pair pruning prefers the feature with stronger target signal.
        feature_scores = {
            column: float(univariate_numeric_score(dataframe[column], target))
            for column in numeric_columns
        }

    pair_rows: list[dict[str, Any]] = []
    drop_columns: list[str] = []
    dropped_set: set[str] = set()
    flagged_pairs: list[dict[str, Any]] = []

    for left_index, left_column in enumerate(numeric_columns):
        for right_column in numeric_columns[left_index + 1 :]:
            corr_value = float(corr.loc[left_column, right_column])
            if corr_value < threshold:
                continue
            flagged_pairs.append(
                {
                    "left_feature": left_column,
                    "right_feature": right_column,
                    "abs_correlation": corr_value,
                }
            )

    flagged_pairs.sort(key=lambda row: row["abs_correlation"], reverse=True)
    for row in flagged_pairs:
        left_column = str(row["left_feature"])
        right_column = str(row["right_feature"])

        if left_column in dropped_set or right_column in dropped_set:
            row["decision"] = "skipped_already_dropped"
            pair_rows.append(row)
            continue

        left_score = feature_scores.get(left_column)
        right_score = feature_scores.get(right_column)
        drop_column = right_column
        keep_column = left_column
        drop_reason = "later_column_order"

        if left_score is not None and right_score is not None:
            if left_score < right_score:
                drop_column = left_column
                keep_column = right_column
                drop_reason = "lower_target_score"
            elif right_score < left_score:
                drop_column = right_column
                keep_column = left_column
                drop_reason = "lower_target_score"
            elif missing_rates[left_column] > missing_rates[right_column]:
                drop_column = left_column
                keep_column = right_column
                drop_reason = "higher_missing_rate"
            elif missing_rates[right_column] > missing_rates[left_column]:
                drop_column = right_column
                keep_column = left_column
                drop_reason = "higher_missing_rate"
        else:
            if missing_rates[left_column] > missing_rates[right_column]:
                drop_column = left_column
                keep_column = right_column
                drop_reason = "higher_missing_rate"
            elif missing_rates[right_column] > missing_rates[left_column]:
                drop_column = right_column
                keep_column = left_column
                drop_reason = "higher_missing_rate"

        row["left_score"] = left_score
        row["right_score"] = right_score
        row["kept_feature"] = keep_column
        row["dropped_feature"] = drop_column
        row["drop_reason"] = drop_reason
        row["decision"] = "dropped"
        pair_rows.append(row)
        drop_columns.append(drop_column)
        dropped_set.add(drop_column)

    selected_features = [
        column for column in candidate_columns if column not in dropped_set
    ]
    filtered_df = _build_selected_dataframe(
        dataframe,
        target_column=target_column,
        id_columns=id_columns,
        selected_features=selected_features,
    )
    return CorrelationAnalysisResult(
        filtered_df=filtered_df,
        selected_columns=selected_features,
        dropped_columns=drop_columns,
        pair_rows=pair_rows,
        warnings=warnings,
    )


def _resolve_random_feature_name(columns: pd.Index) -> str:
    """Return a synthetic feature name that does not collide with existing columns.

    Args:
        columns: Existing dataframe columns.

    Returns:
        Unique column name reserved for random-baseline feature ranking.
    """

    base_name = "__random_feature_baseline__"
    if base_name not in columns:
        return base_name

    suffix = 1
    while f"{base_name}_{suffix}" in columns:
        suffix += 1
    return f"{base_name}_{suffix}"


def _add_random_feature(
    dataframe: pd.DataFrame,
    *,
    feature_name: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Return a dataframe copy with one random normal feature appended.

    Args:
        dataframe: Source dataframe.
        feature_name: Synthetic feature column name.
        rng: Random generator used for deterministic baseline values.

    Returns:
        Dataframe copy containing the synthetic random feature.
    """

    augmented = dataframe.copy()
    augmented[feature_name] = rng.normal(loc=0.0, scale=1.0, size=len(augmented))
    return augmented


def _coerce_binary_shap_matrix(
    shap_values: Any,
    *,
    feature_count: int,
) -> np.ndarray:
    """Normalize SHAP output shapes to a two-dimensional feature matrix.

    Args:
        shap_values: Raw values returned by ``shap.TreeExplainer.shap_values``.
        feature_count: Number of feature columns expected in the matrix.

    Returns:
        Two-dimensional SHAP value array with one column per feature.
    """

    if isinstance(shap_values, list):
        shap_values = shap_values[-1]

    shap_matrix = np.asarray(shap_values)
    if shap_matrix.ndim == 3:
        if shap_matrix.shape[1] == feature_count:
            shap_matrix = shap_matrix[:, :, -1]
        elif shap_matrix.shape[2] == feature_count:
            shap_matrix = shap_matrix[-1, :, :]
        else:
            raise ValueError(
                "Unexpected SHAP value shape; could not identify the feature axis."
            )

    if shap_matrix.ndim != 2 or shap_matrix.shape[1] != feature_count:
        raise ValueError(
            "Unexpected SHAP value shape; expected one column per feature."
        )
    return shap_matrix


def rank_features_by_lightgbm(
    dataframe: pd.DataFrame,
    *,
    target_column: str,
    feature_columns: list[str] | None = None,
    validation_df: pd.DataFrame | None = None,
    keep_top_k: int = 100,
    split_config: SplitConfig | None = None,
    train_config: TrainConfig | None = None,
    top_p: float = 0.05,
) -> FeatureRankingResult:
    """Rank candidate features using a lightweight LightGBM model.

    Args:
        dataframe: Source dataframe.
        target_column: Target column used for modeling.
        feature_columns: Optional explicit feature subset.
        validation_df: Optional held-out validation dataframe.
        keep_top_k: Maximum number of top-ranked features to keep.
        split_config: Split configuration used when ``validation_df`` is omitted.
        train_config: Training configuration override.
        top_p: Fraction retained for PPV-style validation metrics.

    Returns:
        Structured LightGBM ranking result.
    """

    if target_column not in dataframe.columns:
        raise ValueError(f"Target column {target_column!r} was not found.")

    feature_columns = feature_columns or [
        str(column) for column in dataframe.columns if str(column) != target_column
    ]
    split = build_train_valid_frames(
        dataframe,
        target_column=target_column,
        feature_columns=feature_columns,
        validation_df=validation_df,
        split_config=split_config,
    )
    categorical_columns = infer_categorical_columns(dataframe, feature_columns)
    result = fit_lightgbm_binary(
        split.train_df,
        split.valid_df,
        target_column=target_column,
        feature_columns=feature_columns,
        categorical_columns=categorical_columns,
        train_config=train_config,
        top_p=top_p,
    )
    importance_rows = [
        {"feature": feature, "gain": gain}
        for feature, gain in sorted(
            result.feature_importance_gain.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:keep_top_k]
    ]
    return FeatureRankingResult(
        selected_columns=[row["feature"] for row in importance_rows],
        categorical_columns=result.categorical_columns,
        importance_rows=importance_rows,
        evaluation_summary=result.evaluation_summary,
    )


def rank_features_by_shap(
    dataframe: pd.DataFrame,
    *,
    target_column: str,
    feature_columns: list[str] | None = None,
    validation_df: pd.DataFrame | None = None,
    keep_top_k: int | None = None,
    filter_below_random: bool = False,
    split_config: SplitConfig | None = None,
    train_config: TrainConfig | None = None,
    top_p: float = 0.05,
) -> FeatureRankingResult:
    """Rank candidate features using LightGBM SHAP values on validation rows.

    Args:
        dataframe: Source dataframe.
        target_column: Target column used for modeling.
        feature_columns: Optional explicit feature subset.
        validation_df: Optional held-out validation dataframe.
        keep_top_k: Optional maximum number of top-ranked features to keep.
        filter_below_random: Whether to add a random feature and drop features
            with lower mean absolute SHAP value than that random baseline.
        split_config: Split configuration used when ``validation_df`` is omitted.
        train_config: Training configuration override.
        top_p: Fraction retained for PPV-style validation metrics.

    Returns:
        Structured SHAP ranking result.

    Example:
        >>> ranking = rank_features_by_shap(
        ...     frame,
        ...     target_column="target",
        ...     keep_top_k=25,
        ...     filter_below_random=True,
        ... )
    """

    if target_column not in dataframe.columns:
        raise ValueError(f"Target column {target_column!r} was not found.")
    if keep_top_k is not None and keep_top_k <= 0:
        raise ValueError("`keep_top_k` must be greater than zero when provided.")

    feature_columns = feature_columns or [
        str(column) for column in dataframe.columns if str(column) != target_column
    ]
    ranking_dataframe = dataframe
    ranking_validation_df = validation_df
    ranking_feature_columns = list(feature_columns)
    random_feature_name: str | None = None

    if filter_below_random:
        all_input_columns = pd.Index(
            [
                *dataframe.columns,
                *(validation_df.columns if validation_df is not None else []),
            ]
        )
        random_feature_name = _resolve_random_feature_name(all_input_columns)
        seed = (
            split_config.random_seed
            if split_config is not None
            else SplitConfig().random_seed
        )
        rng = np.random.default_rng(seed)
        ranking_dataframe = _add_random_feature(
            dataframe,
            feature_name=random_feature_name,
            rng=rng,
        )
        if validation_df is not None:
            ranking_validation_df = _add_random_feature(
                validation_df,
                feature_name=random_feature_name,
                rng=rng,
            )
        ranking_feature_columns.append(random_feature_name)

    split = build_train_valid_frames(
        ranking_dataframe,
        target_column=target_column,
        feature_columns=ranking_feature_columns,
        validation_df=ranking_validation_df,
        split_config=split_config,
    )
    categorical_columns = infer_categorical_columns(
        ranking_dataframe,
        ranking_feature_columns,
    )
    result = fit_lightgbm_binary(
        split.train_df,
        split.valid_df,
        target_column=target_column,
        feature_columns=ranking_feature_columns,
        categorical_columns=categorical_columns,
        train_config=train_config,
        top_p=top_p,
    )
    _, _, X_valid, _, _ = prepare_lightgbm_train_valid_frames(
        split.train_df,
        split.valid_df,
        target_column=target_column,
        feature_columns=ranking_feature_columns,
        categorical_columns=categorical_columns,
    )

    import shap

    explainer = shap.TreeExplainer(result.booster)
    raw_shap_values = explainer.shap_values(X_valid)
    shap_matrix = _coerce_binary_shap_matrix(
        raw_shap_values,
        feature_count=len(ranking_feature_columns),
    )
    mean_abs_shap = {
        feature: float(value)
        for feature, value in zip(
            ranking_feature_columns,
            np.abs(shap_matrix).mean(axis=0),
            strict=True,
        )
    }

    random_threshold = (
        mean_abs_shap[random_feature_name] if random_feature_name is not None else None
    )
    importance_rows = []
    for feature, value in sorted(
        mean_abs_shap.items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        if feature == random_feature_name:
            continue
        passed_random_filter = (
            True if random_threshold is None else value >= random_threshold
        )
        importance_rows.append(
            {
                "feature": feature,
                "mean_abs_shap": value,
                "passed_random_filter": passed_random_filter,
            }
        )

    if filter_below_random:
        importance_rows = [
            row for row in importance_rows if bool(row["passed_random_filter"])
        ]
    if keep_top_k is not None:
        importance_rows = importance_rows[:keep_top_k]

    evaluation_summary = {
        **result.evaluation_summary,
        "importance_method": "shap_mean_abs",
        "filter_below_random": bool(filter_below_random),
        "random_feature": random_feature_name,
        "random_feature_mean_abs_shap": random_threshold,
    }
    selected_columns = [str(row["feature"]) for row in importance_rows]
    selected_column_set = set(selected_columns)
    return FeatureRankingResult(
        selected_columns=selected_columns,
        categorical_columns=[
            column
            for column in result.categorical_columns
            if column in selected_column_set
        ],
        importance_rows=importance_rows,
        evaluation_summary=evaluation_summary,
    )


def rank_feature_subsets(
    dataframe: pd.DataFrame,
    *,
    target_column: str,
    feature_subsets: list[list[str]],
    validation_df: pd.DataFrame | None = None,
    keep_top_k_per_subset: int = 25,
    split_config: SplitConfig | None = None,
    train_config: TrainConfig | None = None,
    top_p: float = 0.05,
) -> FeatureSubsetRankingResult:
    """Preselect features from wide tables using repeated subset-level LightGBM passes.

    Use this when one full-width importance run would be too expensive or noisy.
    Each subset is evaluated independently with ``rank_features_by_lightgbm(...)``,
    the top features from that subset are retained, and the winners are unioned
    into one reduced dataframe for later global ranking or modeling.

    Args:
        dataframe: Source dataframe.
        target_column: Binary target column used for subset-level ranking.
        feature_subsets: Ordered feature subsets to evaluate independently.
        validation_df: Optional held-out validation dataframe.
        keep_top_k_per_subset: Maximum retained features from each subset-level
            ranking pass.
        split_config: Split configuration used when ``validation_df`` is omitted.
        train_config: Training configuration override.
        top_p: Fraction retained for PPV-style validation metrics.

    Returns:
        Structured batched preselection result containing the union of per-subset
        winners.
    """

    if target_column not in dataframe.columns:
        raise ValueError(f"Target column {target_column!r} was not found.")

    subset_rows: list[dict[str, Any]] = []
    selected_union: list[str] = []
    warnings: list[str] = []

    for subset_index, raw_subset in enumerate(feature_subsets, start=1):
        subset_columns = [
            column
            for column in raw_subset
            if column in dataframe.columns and column != target_column
        ]
        if not subset_columns:
            warnings.append(f"Skipped empty feature subset at index {subset_index}.")
            continue

        ranking = rank_features_by_lightgbm(
            dataframe,
            target_column=target_column,
            feature_columns=subset_columns,
            validation_df=validation_df,
            keep_top_k=keep_top_k_per_subset,
            split_config=split_config,
            train_config=train_config,
            top_p=top_p,
        )
        for column in ranking.selected_columns:
            if column not in selected_union:
                selected_union.append(column)
        subset_rows.append(
            {
                "subset_index": subset_index,
                "input_feature_count": len(subset_columns),
                "selected_columns": ranking.selected_columns,
                **ranking.evaluation_summary,
            }
        )

    filtered_df = _build_selected_dataframe(
        dataframe,
        target_column=target_column,
        id_columns=None,
        selected_features=selected_union,
    )
    return FeatureSubsetRankingResult(
        filtered_df=filtered_df,
        selected_columns=selected_union,
        categorical_columns=infer_categorical_columns(dataframe, selected_union),
        subset_rows=subset_rows,
        warnings=warnings,
    )


__all__ = [
    "analyze_feature_correlation",
    "rank_feature_subsets",
    "rank_features_by_lightgbm",
    "rank_features_by_shap",
    "screen_feature_batches",
    "screen_features",
    "screen_parquet_feature_batches",
    "univariate_categorical_score",
    "univariate_numeric_score",
]
