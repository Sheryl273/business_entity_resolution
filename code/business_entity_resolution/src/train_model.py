"""
Owner: Person C — Krishna Gupta (Matching Model + Prediction)

Trains a binary match/no-match LightGBM classifier on Person B's feature
matrix (labeled via train_ground_truth.tsv) and tunes the decision threshold
for Macro F0.5 — the competition's official metric.

Pipeline:
  load config → load features → validate schema → load ground truth →
  construct binary labels → validate split / leakage → prepare X/y →
  (optional) hard-negative mining → train LightGBM → validate probabilities →
  threshold sweep on Macro F0.5 → save model + metadata + reports

Usage (see --help for all options):
  python src/train_model.py \\
    --train-features  feature_matrices/train_features.parquet \\
    --val-features    feature_matrices/val_features.parquet \\
    --train-gt        ../../dataset/train/train_ground_truth.tsv \\
    --val-gt          ../../dataset/train/val_ground_truth.tsv \\
    --config          config/model.yaml \\
    --output-model    models/matcher.txt \\
    --metadata-output models/matcher_metadata.json \\
    --reports-dir     reports/
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl
import yaml

# ─── Make sure we can import score_f05 from the same src/ directory ─────────
_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from score_f05 import macro_f05_score, load_id_list_tsv  # noqa: E402

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("train_model")

# ─── Column name constants (structural, not feature-column names) ─────────────
S1_COL = "source1_entity_id"
CAND_COL = "candidate_entity_id"
MATCH_COL = "matched_entity_ids"
LABEL_COL = "is_match"
PROB_COL = "_prob"


# ══════════════════════════════════════════════════════════════════════════════
#  Configuration loading
# ══════════════════════════════════════════════════════════════════════════════

def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML config and return as nested dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open() as fh:
        cfg = yaml.safe_load(fh)
    log.info("Loaded config from %s", path)
    return cfg


# ══════════════════════════════════════════════════════════════════════════════
#  Feature matrix loading + schema validation
# ══════════════════════════════════════════════════════════════════════════════

def _load_feature_file(path: str | Path, split_name: str) -> pl.DataFrame:
    """Load a feature matrix from Parquet or TSV/CSV, auto-detecting format."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Feature file for '{split_name}' not found: {path}")
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        df = pl.read_parquet(path)
    elif suffix in {".csv"}:
        df = pl.read_csv(path)
    elif suffix in {".tsv"}:
        df = pl.read_csv(path, separator="\t")
    else:
        # Try parquet first, fall back to TSV
        try:
            df = pl.read_parquet(path)
        except Exception:
            df = pl.read_csv(path, separator="\t")
    log.info("[%s] Loaded feature matrix: %d rows × %d cols from %s",
             split_name, len(df), len(df.columns), path)
    return df


def detect_feature_columns(df: pl.DataFrame, id_cols: list[str]) -> list[str]:
    """
    Infer feature columns = all numeric columns that are NOT identifier columns.
    This makes the code independent of any hard-coded feature name list.
    """
    non_id = [c for c in df.columns if c not in id_cols]
    numeric = [
        c for c in non_id
        if df[c].dtype in (
            pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64,
            pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
        )
    ]
    if not numeric:
        raise ValueError(
            "No numeric feature columns found after excluding ID columns. "
            f"ID columns: {id_cols}. All columns: {df.columns}"
        )
    log.info("Auto-detected %d feature columns: %s", len(numeric), numeric)
    return numeric


def validate_schema(
    df: pl.DataFrame,
    required_id_cols: list[str],
    feature_cols: list[str] | None,
    split_name: str,
) -> None:
    """Validate that required columns are present and have sensible dtypes."""
    missing_ids = [c for c in required_id_cols if c not in df.columns]
    if missing_ids:
        raise ValueError(
            f"[{split_name}] Required identifier columns missing: {missing_ids}. "
            f"Present columns: {df.columns}"
        )
    if feature_cols:
        missing_feats = [c for c in feature_cols if c not in df.columns]
        if missing_feats:
            raise ValueError(
                f"[{split_name}] Feature columns missing (expected from training metadata): "
                f"{missing_feats}. Present columns: {df.columns}"
            )
    # Check for duplicate (S1, candidate) pairs — vectorized
    n_total = len(df)
    n_unique = df.select(required_id_cols).n_unique()
    if n_total != n_unique:
        log.warning(
            "[%s] Found %d duplicate (source1, candidate) pairs — keeping first occurrence",
            split_name, n_total - n_unique,
        )


def validate_train_val_compatibility(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    feature_cols: list[str],
) -> None:
    """Assert that train and val have the same feature columns (same schema)."""
    val_set = set(val_df.columns)
    feats_in_train = set(feature_cols)
    feats_missing_val = feats_in_train - val_set
    if feats_missing_val:
        raise ValueError(
            f"Validation feature matrix is missing columns present in training: "
            f"{feats_missing_val}"
        )
    # Check dtype compatibility
    mismatched = []
    for col in feature_cols:
        if col in val_df.columns:
            t_dtype = train_df[col].dtype
            v_dtype = val_df[col].dtype
            # Allow float32/float64 mismatch (cast later); flag string vs numeric
            if t_dtype != v_dtype:
                mismatched.append((col, t_dtype, v_dtype))
    if mismatched:
        log.warning(
            "Dtype mismatches between train and val (will cast): %s", mismatched
        )


# ══════════════════════════════════════════════════════════════════════════════
#  Ground truth loading + label construction
# ══════════════════════════════════════════════════════════════════════════════

def load_ground_truth(path: str | Path) -> dict[str, set[str]]:
    """
    Parse train_ground_truth.tsv (source1_entity_id, matched_entity_ids)
    where matched_entity_ids is comma-separated (or empty for singletons).
    Returns {source1_entity_id: set_of_matched_ids}.

    Fully vectorized via Polars — no Python row-by-row loop.
    Handles: nulls, whitespace, duplicates within a list, duplicate rows.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {path}")

    df = pl.read_csv(path, separator="\t", null_values=["", "NULL", "null", "NA"])
    log.info("Loaded ground truth: %d rows from %s", len(df), path)

    required = [S1_COL, MATCH_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Ground truth TSV missing required columns: {missing}. "
            f"Found: {df.columns}"
        )

    # Cast S1_COL to string and strip whitespace
    df = df.with_columns(pl.col(S1_COL).cast(pl.Utf8).str.strip_chars())

    # Drop rows with empty/null S1 IDs
    df = df.filter(pl.col(S1_COL).is_not_null() & (pl.col(S1_COL) != ""))

    # Build result dict using vectorized Polars operations
    # For each S1 entity, collect all matched IDs as a set (union across duplicate rows)
    gt: dict[str, set[str]] = {}

    # Separate: singletons (null or empty matched_entity_ids)
    singleton_mask = pl.col(MATCH_COL).is_null() | (pl.col(MATCH_COL).cast(pl.Utf8).str.strip_chars() == "")
    singletons = df.filter(singleton_mask).select(S1_COL)[S1_COL].to_list()
    for s1_id in singletons:
        if s1_id not in gt:
            gt[s1_id] = set()

    # Non-singletons: split and explode comma-separated IDs
    non_singletons = df.filter(~singleton_mask).with_columns(
        pl.col(MATCH_COL).cast(pl.Utf8).str.strip_chars()
    )
    if len(non_singletons) > 0:
        exploded = (
            non_singletons
            .with_columns(
                pl.col(MATCH_COL)
                .str.split(",")
                .alias("_ids_list")
            )
            .explode("_ids_list", empty_as_null=True)
            .with_columns(pl.col("_ids_list").str.strip_chars().alias("_cand_id"))
            .filter(pl.col("_cand_id") != "")
            .select([S1_COL, "_cand_id"])
        )
        for row in exploded.iter_rows():
            s1_id, cand_id = row[0], row[1]
            if s1_id in gt:
                gt[s1_id].add(cand_id)
            else:
                gt[s1_id] = {cand_id}

    total_matches = sum(len(v) for v in gt.values())
    singletons_count = sum(1 for v in gt.values() if not v)
    log.info(
        "Ground truth: %d S1 entities, %d total matches, %d singletons",
        len(gt), total_matches, singletons_count,
    )
    return gt


def build_labels(
    feature_df: pl.DataFrame,
    ground_truth: dict[str, set[str]],
    split_name: str = "train",
) -> pl.DataFrame:
    """
    Vectorized label construction via Polars join.

    Each (source1_entity_id, candidate_entity_id) pair receives:
      is_match = 1  if candidate_entity_id ∈ ground_truth[source1_entity_id]
      is_match = 0  otherwise

    Strategy:
    1. Explode ground truth dict → flat polars DataFrame of positive pairs.
    2. Left-join onto feature_df on (S1_COL, CAND_COL).
    3. Fill non-matching rows with 0.

    This is a vectorized join — no Python row-by-row loop.
    """
    # Build a flat positive-pairs DataFrame from the ground truth dict
    pos_rows = [
        (s1_id, cand_id)
        for s1_id, cand_ids in ground_truth.items()
        for cand_id in cand_ids
    ]
    if pos_rows:
        pos_df = pl.DataFrame(
            {S1_COL: [r[0] for r in pos_rows], CAND_COL: [r[1] for r in pos_rows]},
            schema={S1_COL: pl.Utf8, CAND_COL: pl.Utf8},
        ).with_columns(pl.lit(1).cast(pl.Int8).alias(LABEL_COL))
        # Deduplicate positive pairs (safety measure)
        pos_df = pos_df.unique(subset=[S1_COL, CAND_COL])
    else:
        pos_df = pl.DataFrame(
            schema={S1_COL: pl.Utf8, CAND_COL: pl.Utf8, LABEL_COL: pl.Int8}
        )

    # Ensure ID cols in feature_df are strings
    df = feature_df.with_columns([
        pl.col(S1_COL).cast(pl.Utf8),
        pl.col(CAND_COL).cast(pl.Utf8),
    ])

    # Remove any pre-existing label column to avoid collisions
    if LABEL_COL in df.columns:
        df = df.drop(LABEL_COL)

    # Left join: every candidate pair gets a label; non-matches get null → 0
    labeled = df.join(pos_df, on=[S1_COL, CAND_COL], how="left").with_columns(
        pl.col(LABEL_COL).fill_null(0).cast(pl.Int8)
    )

    pos_count = int(labeled[LABEL_COL].sum())
    neg_count = len(labeled) - pos_count
    pos_ratio = pos_count / len(labeled) if len(labeled) > 0 else 0.0
    log.info(
        "[%s] Labels: %d rows | positives=%d | negatives=%d | pos_ratio=%.4f",
        split_name, len(labeled), pos_count, neg_count, pos_ratio,
    )
    return labeled


# ══════════════════════════════════════════════════════════════════════════════
#  Leakage check
# ══════════════════════════════════════════════════════════════════════════════

def check_leakage(
    train_labeled: pl.DataFrame,
    val_labeled: pl.DataFrame,
) -> None:
    """
    Detect Source-1 entity ID overlap between train and val splits.
    Entity-resolution splits must be disjoint by Source-1 entity.
    """
    train_s1 = set(train_labeled[S1_COL].cast(pl.Utf8).to_list())
    val_s1 = set(val_labeled[S1_COL].cast(pl.Utf8).to_list())
    overlap = train_s1 & val_s1
    if overlap:
        raise ValueError(
            f"LEAKAGE DETECTED: {len(overlap)} Source-1 entities appear in BOTH "
            f"train and validation splits. This invalidates the evaluation. "
            f"Sample overlapping IDs: {sorted(overlap)[:10]}"
        )
    log.info(
        "Leakage check passed: train has %d unique S1 entities, "
        "val has %d unique S1 entities, overlap = 0",
        len(train_s1), len(val_s1),
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Class imbalance
# ══════════════════════════════════════════════════════════════════════════════

def compute_scale_pos_weight(y: np.ndarray) -> float:
    """
    Compute scale_pos_weight = neg_count / pos_count.
    Used when class_weight != 'balanced' in config.
    """
    pos = int(y.sum())
    neg = len(y) - pos
    if pos == 0:
        raise ValueError("Training set has zero positive examples — cannot train.")
    spw = neg / pos
    log.info("Class distribution: pos=%d neg=%d, scale_pos_weight=%.2f", pos, neg, spw)
    return spw


# ══════════════════════════════════════════════════════════════════════════════
#  Hard-negative mining (optional Phase 2)
# ══════════════════════════════════════════════════════════════════════════════

def mine_hard_negatives(
    labeled_df: pl.DataFrame,
    feature_cols: list[str],
    model: lgb.Booster,
    cfg_hn: dict[str, Any],
) -> pl.DataFrame:
    """
    Re-score all training negatives with the current model, keep only the
    top-k hardest negatives per Source-1 entity, and return a resampled
    training set.

    This is a Phase 2 operation — only called when hard_negative_mining.enabled.

    Bug fix vs original: negatives are sorted within each S1 group BEFORE
    calling group_by().head(), ensuring we always get the k highest-prob negatives
    (not arbitrary k rows per group).
    """
    log.info("Starting hard-negative mining …")
    top_k = int(cfg_hn.get("top_k_per_entity", 5))
    ratio = float(cfg_hn.get("ratio", 1.0))

    X = labeled_df.select(feature_cols).to_numpy().astype(np.float32)
    probs = model.predict(X)

    df_scored = labeled_df.with_columns(
        pl.Series(PROB_COL, probs, dtype=pl.Float32)
    )

    positives = df_scored.filter(pl.col(LABEL_COL) == 1)
    negatives = df_scored.filter(pl.col(LABEL_COL) == 0)

    # FIX: Sort within each S1 group first, THEN take top-k.
    # group_by().head() does NOT guarantee any ordering — must sort first.
    hard_negs = (
        negatives
        .sort([S1_COL, PROB_COL], descending=[False, True])
        .group_by(S1_COL)
        .head(top_k)
    )

    n_pos = len(positives)
    n_neg_target = max(1, int(n_pos * ratio))
    if len(hard_negs) > n_neg_target:
        hard_negs = hard_negs.sample(n=n_neg_target, shuffle=True, seed=42)

    combined = pl.concat([positives, hard_negs]).drop(PROB_COL).sample(
        fraction=1.0, shuffle=True, seed=42
    )
    log.info(
        "Hard-negative mining: kept %d positives + %d hard negatives (%d total)",
        len(positives), len(hard_negs), len(combined),
    )
    return combined


# ══════════════════════════════════════════════════════════════════════════════
#  Model training
# ══════════════════════════════════════════════════════════════════════════════

def _build_lgb_params(cfg: dict[str, Any], seed: int) -> dict[str, Any]:
    """Translate YAML config → LightGBM parameter dict."""
    mc = cfg["model"]
    params = {
        "objective": mc.get("objective", "binary"),
        "metric": mc.get("metric", "binary_logloss"),
        "boosting_type": mc.get("boosting_type", "gbdt"),
        "num_leaves": int(mc.get("num_leaves", 63)),
        "max_depth": int(mc.get("max_depth", -1)),
        "learning_rate": float(mc.get("learning_rate", 0.05)),
        "min_child_samples": int(mc.get("min_child_samples", 50)),
        "subsample": float(mc.get("subsample", 0.8)),
        "subsample_freq": int(mc.get("subsample_freq", 1)),
        "colsample_bytree": float(mc.get("colsample_bytree", 0.8)),
        "reg_alpha": float(mc.get("reg_alpha", 0.1)),
        "reg_lambda": float(mc.get("reg_lambda", 1.0)),
        "n_jobs": int(mc.get("n_jobs", -1)),
        "verbose": int(mc.get("verbose", -1)),
        "seed": seed,
        "deterministic": True,
    }
    return params


def train_model(
    train_labeled: pl.DataFrame,
    val_labeled: pl.DataFrame,
    feature_cols: list[str],
    cfg: dict[str, Any],
    seed: int,
) -> lgb.Booster:
    """
    Train LightGBM with early stopping on validation log-loss.
    Returns the fitted Booster (best iteration is already selected).
    """
    X_train = train_labeled.select(feature_cols).to_numpy().astype(np.float32)
    y_train = train_labeled[LABEL_COL].to_numpy().astype(np.int32)
    X_val = val_labeled.select(feature_cols).to_numpy().astype(np.float32)
    y_val = val_labeled[LABEL_COL].to_numpy().astype(np.int32)

    params = _build_lgb_params(cfg, seed)

    # Handle class weighting
    class_weight_cfg = cfg["model"].get("class_weight", "balanced")
    if class_weight_cfg == "balanced":
        pos = int(y_train.sum())
        neg = len(y_train) - pos
        params["scale_pos_weight"] = neg / pos if pos > 0 else 1.0
        log.info("Using balanced class weighting: scale_pos_weight=%.4f", params["scale_pos_weight"])
    elif class_weight_cfg == "none" or class_weight_cfg is None:
        log.info("No class weighting applied")
    else:
        # Numeric value supplied directly
        try:
            params["scale_pos_weight"] = float(class_weight_cfg)
            log.info("Using explicit scale_pos_weight=%.4f", params["scale_pos_weight"])
        except (TypeError, ValueError):
            log.warning("Unrecognized class_weight '%s', ignoring", class_weight_cfg)

    n_estimators = int(cfg["model"].get("n_estimators", 3000))
    es_cfg = cfg.get("early_stopping", {})
    es_rounds = int(es_cfg.get("rounds", 100))

    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols, free_raw_data=False)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain, free_raw_data=False)

    callbacks = [
        lgb.early_stopping(stopping_rounds=es_rounds, verbose=False),
        lgb.log_evaluation(period=100),
    ]

    log.info(
        "Training LightGBM: n_estimators=%d, early_stopping=%d rounds, "
        "num_leaves=%d, lr=%.4f",
        n_estimators, es_rounds, params["num_leaves"], params["learning_rate"],
    )
    t0 = time.time()
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=n_estimators,
        valid_sets=[dval],
        callbacks=callbacks,
    )
    elapsed = time.time() - t0
    log.info(
        "Training complete: best_iteration=%d, elapsed=%.1fs",
        model.best_iteration, elapsed,
    )
    return model


# ══════════════════════════════════════════════════════════════════════════════
#  Threshold tuning using Macro F0.5
# ══════════════════════════════════════════════════════════════════════════════

def _df_to_predictions_dict(
    df: pl.DataFrame, prob_col: str, threshold: float
) -> dict[str, set[str]]:
    """
    For a given threshold, build {source1_entity_id: set(candidate_entity_id)}
    from the scored DataFrame.
    All Source-1 entities are included (empty set if nothing clears threshold).

    Fully vectorized — no Python row-by-row loop.
    """
    # All S1 entities present in the scored df
    all_s1 = df[S1_COL].cast(pl.Utf8).unique().to_list()
    preds: dict[str, set[str]] = {s1: set() for s1 in all_s1}

    # Filter and group with Polars — vectorized
    above = df.filter(pl.col(prob_col) >= threshold)
    if len(above) > 0:
        grouped = (
            above
            .select([S1_COL, CAND_COL])
            .group_by(S1_COL)
            .agg(pl.col(CAND_COL).cast(pl.Utf8).unique())
        )
        for row in grouped.iter_rows(named=True):
            preds[str(row[S1_COL])] = set(row[CAND_COL])
    return preds


def tune_threshold(
    model: lgb.Booster,
    val_labeled: pl.DataFrame,
    ground_truth: dict[str, set[str]],
    feature_cols: list[str],
    cfg: dict[str, Any],
) -> tuple[float, float, dict[str, float]]:
    """
    Sweep candidate thresholds, score each using Macro F0.5 on the validation set.
    Returns (best_threshold, best_f05, {threshold: f05, ...}).

    ground_truth must be the VALIDATION ground truth only (no train leakage).
    """
    ts_cfg = cfg.get("threshold_search", {})
    t_min = float(ts_cfg.get("min", 0.05))
    t_max = float(ts_cfg.get("max", 0.95))
    t_step = float(ts_cfg.get("step", 0.01))

    X_val = val_labeled.select(feature_cols).to_numpy().astype(np.float32)
    probs = model.predict(X_val)

    val_scored = val_labeled.with_columns(
        pl.Series(PROB_COL, probs, dtype=pl.Float64)
    )

    # Build val-only ground truth: include S1 entities in val features but not in gt
    # (treat as empty / singleton). This must ONLY use val data — no train leakage.
    val_s1_ids = set(val_labeled[S1_COL].cast(pl.Utf8).to_list())
    val_gt: dict[str, set[str]] = {}
    for s1 in val_s1_ids:
        val_gt[s1] = ground_truth.get(s1, set())

    thresholds = np.arange(t_min, t_max + t_step / 2, t_step)
    best_thresh = float(thresholds[0])
    best_f05 = -1.0
    results: dict[str, float] = {}

    log.info(
        "Sweeping %d thresholds [%.2f → %.2f step %.2f] …",
        len(thresholds), t_min, t_max, t_step,
    )
    for thresh in thresholds:
        preds = _df_to_predictions_dict(val_scored, PROB_COL, float(thresh))
        f05 = macro_f05_score(preds, val_gt)
        results[f"{thresh:.4f}"] = f05
        if f05 > best_f05:
            best_f05 = f05
            best_thresh = float(thresh)

    log.info(
        "Threshold search complete: best_threshold=%.4f, best_macro_f05=%.6f",
        best_thresh, best_f05,
    )
    return best_thresh, best_f05, results


# ══════════════════════════════════════════════════════════════════════════════
#  Precision / Recall / F1 at chosen threshold (pair-level)
# ══════════════════════════════════════════════════════════════════════════════

def _compute_pr_at_threshold(
    val_labeled: pl.DataFrame,
    model: lgb.Booster,
    feature_cols: list[str],
    threshold: float,
) -> tuple[float, float, float]:
    """
    Compute pair-level precision, recall, and F1 at the selected threshold.
    Returns (precision, recall, f1).
    """
    X = val_labeled.select(feature_cols).to_numpy().astype(np.float32)
    probs = model.predict(X)
    y_true = val_labeled[LABEL_COL].to_numpy()
    y_pred = (probs >= threshold).astype(np.int32)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0

    log.info(
        "Pair-level at threshold=%.4f: TP=%d FP=%d FN=%d TN=%d | "
        "precision=%.4f recall=%.4f F1=%.4f accuracy=%.4f",
        threshold, tp, fp, fn, tn, precision, recall, f1, accuracy,
    )
    return precision, recall, f1


# ══════════════════════════════════════════════════════════════════════════════
#  Feature importance
# ══════════════════════════════════════════════════════════════════════════════

def extract_feature_importance(
    model: lgb.Booster, feature_cols: list[str]
) -> list[dict[str, Any]]:
    """Extract gain + split importance from trained model."""
    gain_imp = model.feature_importance(importance_type="gain")
    split_imp = model.feature_importance(importance_type="split")
    total_gain = gain_imp.sum() or 1.0
    total_split = split_imp.sum() or 1.0
    rows = [
        {
            "feature": col,
            "gain_importance": float(g),
            "gain_importance_normalized": float(g / total_gain),
            "split_importance": int(s),
            "split_importance_normalized": float(s / total_split),
        }
        for col, g, s in sorted(
            zip(feature_cols, gain_imp, split_imp),
            key=lambda x: -x[1],
        )
    ]
    return rows


# ══════════════════════════════════════════════════════════════════════════════
#  Save artifacts
# ══════════════════════════════════════════════════════════════════════════════

def save_model_and_metadata(
    model: lgb.Booster,
    feature_cols: list[str],
    threshold: float,
    train_stats: dict[str, Any],
    val_stats: dict[str, Any],
    cfg: dict[str, Any],
    seed: int,
    model_path: str | Path,
    metadata_path: str | Path,
) -> None:
    """Persist model (LightGBM text format) and JSON metadata."""
    model_path = Path(model_path)
    metadata_path = Path(metadata_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    model.save_model(str(model_path))
    log.info("Model saved to %s", model_path)

    metadata = {
        "model_type": "lightgbm",
        "training_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "random_seed": seed,
        "feature_columns": feature_cols,
        "selected_threshold": threshold,
        "best_iteration": int(model.best_iteration),
        "train": train_stats,
        "val": val_stats,
        "model_config": cfg["model"],
        "early_stopping_config": cfg.get("early_stopping", {}),
        "threshold_search_config": cfg.get("threshold_search", {}),
        "library_versions": {
            "lightgbm": lgb.__version__,
            "polars": pl.__version__,
            "numpy": np.__version__,
        },
    }
    with metadata_path.open("w") as fh:
        json.dump(metadata, fh, indent=2)
    log.info("Metadata saved to %s", metadata_path)


def save_reports(
    feature_importance: list[dict[str, Any]],
    val_stats: dict[str, Any],
    threshold_results: dict[str, float],
    reports_dir: str | Path,
    experiment_label: str = "baseline",
) -> None:
    """Save model_metrics.json, feature_importance.csv, experiments.json."""
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Feature importance CSV
    fi_path = reports_dir / "feature_importance.csv"
    fi_lines = ["feature,gain_importance,gain_importance_normalized,split_importance,split_importance_normalized"]
    for row in feature_importance:
        fi_lines.append(
            f"{row['feature']},{row['gain_importance']:.6f},"
            f"{row['gain_importance_normalized']:.6f},"
            f"{row['split_importance']},{row['split_importance_normalized']:.6f}"
        )
    fi_path.write_text("\n".join(fi_lines))
    log.info("Feature importance saved to %s", fi_path)

    # Model metrics JSON
    metrics_path = reports_dir / "model_metrics.json"
    with metrics_path.open("w") as fh:
        json.dump(val_stats, fh, indent=2)
    log.info("Model metrics saved to %s", metrics_path)

    # Experiments JSON (append-style; allows comparing runs)
    experiments_path = reports_dir / "experiments.json"
    existing: list[dict] = []
    if experiments_path.exists():
        try:
            with experiments_path.open() as fh:
                existing = json.load(fh)
        except (json.JSONDecodeError, ValueError):
            existing = []
    experiment_entry = {
        "label": experiment_label,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        **val_stats,
        "threshold_curve_sample": {
            k: v for i, (k, v) in enumerate(threshold_results.items()) if i % 10 == 0
        },
    }
    existing.append(experiment_entry)
    with experiments_path.open("w") as fh:
        json.dump(existing, fh, indent=2)
    log.info("Experiment record saved to %s", experiments_path)


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="train_model.py",
        description="Person C — Train LightGBM matcher and tune Macro F0.5 threshold.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--train-features", required=True, metavar="PATH",
        help="Path to training feature matrix (Parquet or TSV/CSV).",
    )
    parser.add_argument(
        "--val-features", required=True, metavar="PATH",
        help="Path to validation feature matrix (same schema as train).",
    )
    parser.add_argument(
        "--train-gt", required=True, metavar="PATH",
        help="Path to train_ground_truth.tsv (source1_entity_id, matched_entity_ids).",
    )
    parser.add_argument(
        "--val-gt", required=True, metavar="PATH",
        help="Path to validation ground truth TSV (same format as train-gt).",
    )
    parser.add_argument(
        "--config", default="config/model.yaml", metavar="PATH",
        help="Path to model.yaml configuration file.",
    )
    parser.add_argument(
        "--output-model", default="models/matcher.txt", metavar="PATH",
        help="Where to save the trained LightGBM model.",
    )
    parser.add_argument(
        "--metadata-output", default="models/matcher_metadata.json", metavar="PATH",
        help="Where to save model metadata JSON.",
    )
    parser.add_argument(
        "--reports-dir", default="reports/", metavar="PATH",
        help="Directory for reports (feature importance, metrics, experiments).",
    )
    parser.add_argument(
        "--seed", type=int, default=None, metavar="INT",
        help="Random seed (overrides config). Default: read from config.",
    )
    parser.add_argument(
        "--hard-negative-mining", action="store_true",
        help="Enable hard-negative mining after first training pass (Phase 2).",
    )
    parser.add_argument(
        "--experiment-label", default="baseline", metavar="STR",
        help="Label for this experiment in experiments.json.",
    )
    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
#  Main entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()

    # ── 1. Load config ────────────────────────────────────────────────────────
    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 42))
    np.random.seed(seed)
    log.info("Random seed: %d", seed)

    # ── 2. Load feature matrices ──────────────────────────────────────────────
    train_df = _load_feature_file(args.train_features, "train")
    val_df = _load_feature_file(args.val_features, "val")

    required_id_cols = [S1_COL, CAND_COL]

    # ── 3. Validate schema ────────────────────────────────────────────────────
    validate_schema(train_df, required_id_cols, feature_cols=None, split_name="train")
    feature_cols = detect_feature_columns(train_df, required_id_cols)

    validate_schema(val_df, required_id_cols, feature_cols=feature_cols, split_name="val")
    validate_train_val_compatibility(train_df, val_df, feature_cols)

    # ── 4. Load ground truth ──────────────────────────────────────────────────
    train_gt = load_ground_truth(args.train_gt)
    val_gt = load_ground_truth(args.val_gt)

    # ── 5. Construct labels ───────────────────────────────────────────────────
    train_labeled = build_labels(train_df, train_gt, split_name="train")
    val_labeled = build_labels(val_df, val_gt, split_name="val")

    # ── 6. Validate split / leakage ───────────────────────────────────────────
    check_leakage(train_labeled, val_labeled)

    # ── 7. Stats ──────────────────────────────────────────────────────────────
    train_pos = int(train_labeled[LABEL_COL].sum())
    train_neg = len(train_labeled) - train_pos
    val_pos = int(val_labeled[LABEL_COL].sum())
    val_neg = len(val_labeled) - val_pos
    train_s1_count = train_labeled[S1_COL].n_unique()
    val_s1_count = val_labeled[S1_COL].n_unique()

    log.info("─── Dataset statistics ───────────────────────────────────")
    log.info("Train: %d rows | %d unique S1 | pos=%d neg=%d ratio=%.4f",
             len(train_labeled), train_s1_count, train_pos, train_neg,
             train_pos / len(train_labeled) if len(train_labeled) > 0 else 0)
    log.info("Val:   %d rows | %d unique S1 | pos=%d neg=%d ratio=%.4f",
             len(val_labeled), val_s1_count, val_pos, val_neg,
             val_pos / len(val_labeled) if len(val_labeled) > 0 else 0)
    log.info("──────────────────────────────────────────────────────────")

    # ── 8. Train model ────────────────────────────────────────────────────────
    model = train_model(train_labeled, val_labeled, feature_cols, cfg, seed)

    # ── 8b. Optional hard-negative mining (Phase 2) ───────────────────────────
    hn_cfg = cfg.get("hard_negative_mining", {})
    enable_hn = args.hard_negative_mining or bool(hn_cfg.get("enabled", False))
    if enable_hn:
        log.info("Hard-negative mining enabled — starting Phase 2 …")
        train_hn = mine_hard_negatives(train_labeled, feature_cols, model, hn_cfg)
        model = train_model(train_hn, val_labeled, feature_cols, cfg, seed)

    # ── 9. Threshold tuning ───────────────────────────────────────────────────
    best_thresh, best_f05, threshold_results = tune_threshold(
        model, val_labeled, val_gt, feature_cols, cfg
    )

    # ── 10. Precision / recall / F1 at best threshold ─────────────────────────
    precision, recall, f1 = _compute_pr_at_threshold(
        val_labeled, model, feature_cols, best_thresh
    )
    log.info(
        "Validation at threshold=%.4f: Macro_F0.5=%.6f | precision=%.6f | recall=%.6f | F1=%.6f",
        best_thresh, best_f05, precision, recall, f1,
    )

    # ── 11. Build stats dicts ─────────────────────────────────────────────────
    train_stats = {
        "rows": len(train_labeled),
        "unique_s1_entities": train_s1_count,
        "positive_count": train_pos,
        "negative_count": train_neg,
        "positive_ratio": round(train_pos / len(train_labeled), 6) if len(train_labeled) > 0 else 0.0,
    }
    val_stats = {
        "rows": len(val_labeled),
        "unique_s1_entities": val_s1_count,
        "positive_count": val_pos,
        "negative_count": val_neg,
        "positive_ratio": round(val_pos / len(val_labeled), 6) if len(val_labeled) > 0 else 0.0,
        "threshold": best_thresh,
        "macro_f05": round(best_f05, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "best_lgbm_iteration": int(model.best_iteration),
        "hard_negative_mining": enable_hn,
        "experiment_label": args.experiment_label,
    }

    # ── 12. Save model + metadata ─────────────────────────────────────────────
    save_model_and_metadata(
        model, feature_cols, best_thresh,
        train_stats, val_stats, cfg, seed,
        args.output_model, args.metadata_output,
    )

    # ── 13. Save reports ──────────────────────────────────────────────────────
    fi = extract_feature_importance(model, feature_cols)
    save_reports(fi, val_stats, threshold_results, args.reports_dir, args.experiment_label)

    log.info("═══ Training complete ══════════════════════════════════════")
    log.info("  Model        : %s", args.output_model)
    log.info("  Metadata     : %s", args.metadata_output)
    log.info("  Reports      : %s", args.reports_dir)
    log.info("  Threshold    : %.4f", best_thresh)
    log.info("  Macro F0.5   : %.6f", best_f05)
    log.info("  Precision    : %.6f", precision)
    log.info("  Recall       : %.6f", recall)
    log.info("  F1           : %.6f", f1)
    log.info("══════════════════════════════════════════════════════════════")


if __name__ == "__main__":
    main()
