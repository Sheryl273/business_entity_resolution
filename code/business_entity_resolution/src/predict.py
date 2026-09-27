"""
Owner: Person C — Krishna Gupta (Matching Model + Prediction)

Runs inference over the test feature matrix and produces
output/matching_results.tsv — the ONLY file scored on the leaderboard.

Hard constraints from the problem statement (violating any causes rejection):
  - Every Source 1 test entity must have exactly one row
    (empty matched_entity_ids if no matches)
  - matched_entity_ids: S2-/S3- IDs only, that exist in the test set
  - No duplicate IDs within a list, no duplicate source1_entity_id rows
  - Every matched ID must also appear in output/candidate_pairs.tsv

Usage (see --help for all options):
  python src/predict.py \\
    --model    models/matcher.txt \\
    --metadata models/matcher_metadata.json \\
    --features feature_matrices/test_features.parquet \\
    --source1  ../../dataset/test/test_source1.tsv \\
    --output   ../../output/matching_results.tsv

Optional safety guard (validates predicted IDs are in candidate pairs):
    --candidate-pairs ../../output/candidate_pairs.tsv
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl

# ─── Structural column name constants ─────────────────────────────────────────
S1_COL = "source1_entity_id"
CAND_COL = "candidate_entity_id"
MATCH_COL = "matched_entity_ids"
PROB_COL = "_prob"

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("predict")


# ══════════════════════════════════════════════════════════════════════════════
#  Metadata loading
# ══════════════════════════════════════════════════════════════════════════════

def load_metadata(path: str | Path) -> dict[str, Any]:
    """Load JSON metadata produced by train_model.py."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")
    with path.open() as fh:
        meta = json.load(fh)
    log.info("Loaded metadata from %s", path)
    _required_meta_keys = ["feature_columns", "selected_threshold"]
    missing = [k for k in _required_meta_keys if k not in meta]
    if missing:
        raise ValueError(
            f"Metadata is missing required keys: {missing}. "
            "Re-run training to regenerate metadata."
        )
    return meta


# ══════════════════════════════════════════════════════════════════════════════
#  Model loading
# ══════════════════════════════════════════════════════════════════════════════

def load_model(model_path: str | Path) -> lgb.Booster:
    """Load a LightGBM model from the text format saved by train_model.py."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    model = lgb.Booster(model_file=str(path))
    log.info("Loaded model from %s", path)
    return model


# ══════════════════════════════════════════════════════════════════════════════
#  Feature file loading + schema validation
# ══════════════════════════════════════════════════════════════════════════════

def load_feature_file(path: str | Path) -> pl.DataFrame:
    """Load test feature matrix, auto-detecting format by extension."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Test feature file not found: {path}")
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        df = pl.read_parquet(path)
    elif suffix == ".csv":
        df = pl.read_csv(path)
    elif suffix == ".tsv":
        df = pl.read_csv(path, separator="\t")
    else:
        try:
            df = pl.read_parquet(path)
        except Exception:
            df = pl.read_csv(path, separator="\t")
    log.info("Loaded test features: %d rows × %d cols from %s", len(df), len(df.columns), path)
    return df


def validate_test_schema(
    df: pl.DataFrame,
    feature_cols: list[str],
    split_name: str = "test",
) -> pl.DataFrame:
    """
    Validate that the test feature matrix is compatible with training metadata.
    - Required ID columns must be present.
    - All training feature columns must be present.
    - Reorder columns to match training order exactly.
    - Cast float32/float64 mismatches gracefully.
    - Warn about unexpected extra columns.
    Returns the df with features in the correct order.
    """
    required_id_cols = [S1_COL, CAND_COL]
    missing_ids = [c for c in required_id_cols if c not in df.columns]
    if missing_ids:
        raise ValueError(
            f"[{split_name}] Required identifier columns missing: {missing_ids}. "
            f"Present columns: {df.columns}"
        )

    missing_feats = [c for c in feature_cols if c not in df.columns]
    if missing_feats:
        raise ValueError(
            f"[{split_name}] Feature columns from training metadata are missing: "
            f"{missing_feats}. Present columns: {df.columns}"
        )

    extra_feats = [
        c for c in df.columns
        if c not in required_id_cols and c not in feature_cols
    ]
    if extra_feats:
        log.warning(
            "[%s] Extra columns not in training feature list (will be ignored): %s",
            split_name, extra_feats,
        )

    # Check for NaN/null in feature columns
    for col in feature_cols:
        null_count = df[col].null_count()
        if null_count > 0:
            log.warning(
                "[%s] Feature '%s' has %d null values — LightGBM handles these natively",
                split_name, col, null_count,
            )

    # Duplicate (S1, candidate) pairs check
    dup_count = df.filter(pl.struct([S1_COL, CAND_COL]).is_duplicated()).shape[0]
    if dup_count > 0:
        log.warning(
            "[%s] %d duplicate (source1, candidate) rows found — keeping first",
            split_name, dup_count,
        )
        df = df.unique(subset=[S1_COL, CAND_COL], keep="first")

    # Select + reorder to exactly match training feature order
    df = df.select(required_id_cols + feature_cols)
    log.info("[%s] Schema validated: %d feature columns confirmed", split_name, len(feature_cols))
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  Source-1 entity loading
# ══════════════════════════════════════════════════════════════════════════════

def load_source1_ids(path: str | Path) -> list[str]:
    """
    Load all Source-1 entity IDs from the test source1 TSV.
    Every one of these must appear in the final output — even with empty matches.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Source-1 file not found: {path}")
    df = pl.read_csv(path, separator="\t", columns=[S1_COL])
    ids = df[S1_COL].cast(pl.Utf8).unique().to_list()
    log.info("Loaded %d unique Source-1 test entity IDs from %s", len(ids), path)
    return ids


def load_candidate_pair_ids(path: str | Path) -> set[str]:
    """
    Load all candidate entity IDs from candidate_pairs.tsv.
    Used for output validation — predicted IDs must be a subset.

    Fully vectorized via Polars split + explode — no Python row-by-row loop.
    """
    path = Path(path)
    if not path.exists():
        log.warning("candidate_pairs.tsv not found at %s — skipping candidate ID validation", path)
        return set()
    df = pl.read_csv(path, separator="\t", null_values=["NULL"])
    col = "candidate_entity_ids"
    if col not in df.columns:
        log.warning("Column '%s' not found in candidate_pairs.tsv — skipping validation", col)
        return set()

    # Vectorized: split comma-separated IDs, explode, strip, collect unique set
    cand_ids: set[str] = set(
        df
        .filter(pl.col(col).is_not_null() & (pl.col(col) != ""))
        .with_columns(pl.col(col).str.split(",").alias("_ids"))
        .explode("_ids", empty_as_null=True)
        .with_columns(pl.col("_ids").str.strip_chars())
        .filter(pl.col("_ids") != "")
        ["_ids"]
        .to_list()
    )
    log.info("Loaded %d unique candidate IDs from candidate_pairs.tsv", len(cand_ids))
    return cand_ids


# ══════════════════════════════════════════════════════════════════════════════
#  Core prediction
# ══════════════════════════════════════════════════════════════════════════════

def predict_matches(
    model: lgb.Booster,
    threshold: float,
    test_feature_df: pl.DataFrame,
    feature_cols: list[str],
    all_test_source1_ids: list[str],
) -> pl.DataFrame:
    """
    Score all candidate pairs, apply threshold, group by Source-1 entity.

    Rules:
    - Every Source-1 entity in all_test_source1_ids gets exactly one row.
    - If no candidates clear the threshold, matched_entity_ids = "" (empty string).
    - Multiple candidates above threshold → comma-joined, deduplicated, sorted
      (deterministic ordering).
    - Only candidate IDs from the feature matrix are ever predicted — no invention.
    """
    log.info(
        "Predicting: %d candidate pairs, threshold=%.4f",
        len(test_feature_df), threshold,
    )

    # Score
    X = test_feature_df.select(feature_cols).to_numpy().astype(np.float32)
    probs = model.predict(X)

    # Attach probabilities
    scored = test_feature_df.select([S1_COL, CAND_COL]).with_columns(
        pl.Series(PROB_COL, probs, dtype=pl.Float64)
    )

    # Filter above threshold
    above = scored.filter(pl.col(PROB_COL) >= threshold)
    log.info(
        "Pairs above threshold: %d / %d (%.2f%%)",
        len(above), len(scored),
        100 * len(above) / len(scored) if len(scored) > 0 else 0,
    )

    # Group by S1 → collect unique sorted candidate IDs
    if len(above) > 0:
        grouped = (
            above
            .group_by(S1_COL)
            .agg(pl.col(CAND_COL).unique().sort().alias(MATCH_COL))
            .with_columns(
                pl.col(MATCH_COL).list.join(",").alias(MATCH_COL)
            )
        )
    else:
        grouped = pl.DataFrame(
            {S1_COL: pl.Series([], dtype=pl.Utf8),
             MATCH_COL: pl.Series([], dtype=pl.Utf8)}
        )

    # Build a complete DataFrame for all Source-1 IDs (preserve singletons)
    all_s1_df = pl.DataFrame({S1_COL: all_test_source1_ids}).with_columns(
        pl.col(S1_COL).cast(pl.Utf8)
    )

    result = all_s1_df.join(grouped, on=S1_COL, how="left").with_columns(
        pl.col(MATCH_COL).fill_null("").alias(MATCH_COL)
    )

    n_with_matches = int((result[MATCH_COL] != "").sum())
    n_singletons = len(result) - n_with_matches
    log.info(
        "Output: %d S1 entities | with matches=%d | empty (singleton)=%d",
        len(result), n_with_matches, n_singletons,
    )
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  Output validation
# ══════════════════════════════════════════════════════════════════════════════

def validate_output(
    matches_df: pl.DataFrame,
    all_test_source1_ids: list[str],
    candidate_pair_ids: set[str] | None = None,
) -> None:
    """
    Validate the output DataFrame before writing.
    Fails loudly with a clear error message on any violation.
    """
    errors: list[str] = []

    # Required columns
    for col in [S1_COL, MATCH_COL]:
        if col not in matches_df.columns:
            errors.append(f"Missing required output column: '{col}'")

    if errors:
        raise ValueError("Output validation failed:\n" + "\n".join(errors))

    # No duplicate source1_entity_id rows
    total = len(matches_df)
    unique_s1 = matches_df[S1_COL].n_unique()
    if total != unique_s1:
        errors.append(
            f"Duplicate source1_entity_id rows: {total} rows but only {unique_s1} unique IDs"
        )

    # Every required Source-1 entity is present
    output_s1_set = set(matches_df[S1_COL].cast(pl.Utf8).to_list())
    required_s1_set = set(all_test_source1_ids)
    missing_s1 = required_s1_set - output_s1_set
    if missing_s1:
        errors.append(
            f"{len(missing_s1)} Source-1 test entities missing from output. "
            f"Sample: {sorted(missing_s1)[:10]}"
        )
    extra_s1 = output_s1_set - required_s1_set
    if extra_s1:
        errors.append(
            f"{len(extra_s1)} unexpected Source-1 IDs in output (not in test set). "
            f"Sample: {sorted(extra_s1)[:10]}"
        )

    # No NaN/None text in output
    nan_rows = matches_df.filter(
        pl.col(MATCH_COL).cast(pl.Utf8).is_in(["None", "nan", "NaN", "null"])
    )
    if len(nan_rows) > 0:
        errors.append(
            f"{len(nan_rows)} rows have literal 'None'/'nan' in matched_entity_ids"
        )

    # Validate predicted candidate IDs against the candidate pair set (vectorized)
    if candidate_pair_ids:
        non_empty = matches_df.filter(pl.col(MATCH_COL) != "")
        if len(non_empty) > 0:
            exploded_preds = (
                non_empty
                .with_columns(pl.col(MATCH_COL).str.split(",").alias("_ids"))
                .explode("_ids", empty_as_null=True)
                .with_columns(pl.col("_ids").str.strip_chars())
                .filter(pl.col("_ids") != "")
            )
            all_predicted = set(exploded_preds["_ids"].to_list())
            invalid_ids = all_predicted - candidate_pair_ids
            if invalid_ids:
                sample = sorted(invalid_ids)[:5]
                errors.append(
                    f"{len(invalid_ids)} predicted candidate IDs not found in candidate_pairs.tsv. "
                    f"Sample: {sample}"
                )

    # Duplicate IDs within a single matched_entity_ids list (vectorized check)
    non_empty = matches_df.filter(pl.col(MATCH_COL) != "")
    if len(non_empty) > 0:
        # Count unique IDs per row vs total IDs per row
        dup_check = (
            non_empty
            .with_columns(pl.col(MATCH_COL).str.split(",").alias("_ids"))
            .with_columns([
                pl.col("_ids").list.len().alias("_n_total"),
                pl.col("_ids").list.unique().list.len().alias("_n_unique"),
            ])
            .filter(pl.col("_n_total") != pl.col("_n_unique"))
        )
        if len(dup_check) > 0:
            sample_s1 = dup_check[S1_COL][0]
            errors.append(
                f"{len(dup_check)} rows have duplicate candidate IDs in matched_entity_ids. "
                f"First occurrence: source1_entity_id='{sample_s1}'"
            )

    if errors:
        raise ValueError(
            "Output validation FAILED — fix before submission:\n"
            + "\n".join(f"  • {e}" for e in errors)
        )

    log.info("Output validation passed ✓ (%d Source-1 entities)", len(matches_df))


# ══════════════════════════════════════════════════════════════════════════════
#  Writing the TSV
# ══════════════════════════════════════════════════════════════════════════════

def write_matching_results_tsv(matches_df: pl.DataFrame, out_path: str | Path) -> None:
    """
    Write output/matching_results.tsv in exact required format:
      source1_entity_id<TAB>matched_entity_ids
    - Tab-separated, no quoting (bare TSV)
    - matched_entity_ids is comma-joined (empty string for no matches)
    - Header row included
    - Deterministic ordering by source1_entity_id
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Sort deterministically
    result = matches_df.sort(S1_COL)

    # Write as TSV without quoting — Polars CSV writer with tab separator
    result.write_csv(str(out_path), separator="\t", quote_style="never")
    log.info("Wrote matching_results.tsv: %d rows → %s", len(result), out_path)


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="predict.py",
        description="Person C — Run inference and produce matching_results.tsv.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", required=True, metavar="PATH",
        help="Path to trained LightGBM model file (matcher.txt).",
    )
    parser.add_argument(
        "--metadata", required=True, metavar="PATH",
        help="Path to matcher_metadata.json (contains feature list + threshold).",
    )
    parser.add_argument(
        "--features", required=True, metavar="PATH",
        help="Path to test feature matrix (Parquet or TSV/CSV).",
    )
    parser.add_argument(
        "--source1", required=True, metavar="PATH",
        help="Path to test_source1.tsv (to enumerate all Source-1 test entities).",
    )
    parser.add_argument(
        "--candidate-pairs", default=None, metavar="PATH",
        help="(Optional) candidate_pairs.tsv for output validation.",
    )
    parser.add_argument(
        "--output", default="../../output/matching_results.tsv", metavar="PATH",
        help="Output path for matching_results.tsv.",
    )
    parser.add_argument(
        "--threshold", type=float, default=None, metavar="FLOAT",
        help="Decision threshold (overrides value stored in metadata).",
    )
    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
#  Main entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()

    # ── 1. Load model and metadata ────────────────────────────────────────────
    model = load_model(args.model)
    metadata = load_metadata(args.metadata)

    feature_cols: list[str] = metadata["feature_columns"]
    threshold: float = (
        args.threshold
        if args.threshold is not None
        else float(metadata["selected_threshold"])
    )
    log.info(
        "Feature columns from training metadata (%d): %s", len(feature_cols), feature_cols
    )
    log.info("Decision threshold: %.4f%s",
             threshold,
             " (from metadata)" if args.threshold is None else " (overridden by CLI)")

    # ── 2. Load test features ─────────────────────────────────────────────────
    test_df = load_feature_file(args.features)
    test_df = validate_test_schema(test_df, feature_cols)

    # ── 3. Load all Source-1 test entity IDs ─────────────────────────────────
    all_s1_ids = load_source1_ids(args.source1)

    # ── 4. Load candidate pair IDs (for validation) ───────────────────────────
    candidate_pair_ids: set[str] | None = None
    if args.candidate_pairs:
        candidate_pair_ids = load_candidate_pair_ids(args.candidate_pairs)

    # ── 5. Predict ────────────────────────────────────────────────────────────
    matches_df = predict_matches(
        model, threshold, test_df, feature_cols, all_s1_ids
    )

    # ── 6. Validate output ────────────────────────────────────────────────────
    validate_output(matches_df, all_s1_ids, candidate_pair_ids)

    # ── 7. Write output ───────────────────────────────────────────────────────
    write_matching_results_tsv(matches_df, args.output)

    log.info("═══ Prediction complete ═══════════════════════════════════")
    log.info("  Output       : %s", args.output)
    log.info("  S1 entities  : %d", len(matches_df))
    log.info("  With matches : %d", int((matches_df[MATCH_COL] != "").sum()))
    log.info("  Singletons   : %d", int((matches_df[MATCH_COL] == "").sum()))
    log.info("══════════════════════════════════════════════════════════════")


if __name__ == "__main__":
    main()
