"""
Owner: Person D (Evaluation / Validation / Packaging)
       load_id_list_tsv and macro_f05_score completed here for Person C's
       threshold tuning — Person D should review and finalise.

Local implementation of the challenge's exact scoring metric, used by everyone
(especially Person C for threshold tuning) to self-evaluate before spending a
leaderboard submission.

Formula (from problem statement):
  F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
Computed PER Source 1 entity, then macro-averaged across all entities.

Singleton rule (entity with no true matches):
  - predicting an empty list correctly -> score 1.0 for that entity
  - predicting ANY match for it -> score 0.0 for that entity
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import polars as pl

log = logging.getLogger(__name__)

S1_COL = "source1_entity_id"
MATCH_COL = "matched_entity_ids"


def _f05(precision: float, recall: float) -> float:
    if precision == 0 and recall == 0:
        return 0.0
    denom = 0.25 * precision + recall
    if denom == 0:
        return 0.0
    return (1.25 * precision * recall) / denom


def score_entity(predicted_ids: set[str], true_ids: set[str]) -> float:
    """
    Score a single Source 1 entity's prediction against ground truth.

    Worked example from problem statement:
      predicted = {S2-00047, S2-00193, S3-00812}
      true      = {S2-00047, S3-00812}
      -> precision = 2/3, recall = 1.0, F_0.5 ≈ 0.714
    """
    if not true_ids:
        # singleton: correct iff prediction is also empty
        return 1.0 if not predicted_ids else 0.0

    if not predicted_ids:
        return 0.0  # recall = 0 -> F_0.5 = 0

    tp = len(predicted_ids & true_ids)
    precision = tp / len(predicted_ids) if predicted_ids else 0.0
    recall = tp / len(true_ids) if true_ids else 0.0
    return _f05(precision, recall)


def macro_f05_score(
    predictions: dict[str, set[str]],
    ground_truth: dict[str, set[str]],
) -> float:
    """
    Macro-average F_0.5 across all Source 1 entities in ground_truth.

    - Every entity in ground_truth is scored, even if missing from predictions
      (treat missing prediction as empty set → 0.0 for non-singleton, 1.0 for singleton).
    - If ground_truth is empty, returns 0.0.
    """
    scores = []
    for entity_id, true_ids in ground_truth.items():
        predicted_ids = predictions.get(entity_id, set())
        scores.append(score_entity(predicted_ids, true_ids))
    return sum(scores) / len(scores) if scores else 0.0


def load_id_list_tsv(
    path: str | Path,
    id_col: str = S1_COL,
    list_col: str = MATCH_COL,
) -> dict[str, set[str]]:
    """
    Parse a TSV shaped like matching_results.tsv / train_ground_truth.tsv
    into {source1_entity_id: set(other_ids)}.
    Empty string in list_col -> empty set (singleton).
    Handles: nulls, whitespace, duplicate rows (merged via union).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"TSV file not found: {path}")

    df = pl.read_csv(path, separator="\t", null_values=["", "NULL", "null", "NA"])
    if id_col not in df.columns:
        raise ValueError(
            f"Column '{id_col}' not found in {path}. Found: {df.columns}"
        )
    if list_col not in df.columns:
        raise ValueError(
            f"Column '{list_col}' not found in {path}. Found: {df.columns}"
        )

    result: dict[str, set[str]] = {}
    for row in df.iter_rows(named=True):
        s1_id = row[id_col]
        if s1_id is None:
            continue
        s1_id = str(s1_id).strip()
        if not s1_id:
            continue

        raw = row[list_col]
        if raw is None or str(raw).strip() == "":
            ids: set[str] = set()
        else:
            ids = {tok.strip() for tok in str(raw).split(",") if tok.strip()}

        if s1_id in result:
            result[s1_id] |= ids
        else:
            result[s1_id] = ids

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  CLI — score predictions vs ground truth
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="score_f05.py",
        description="Compute Macro F0.5 between a predictions TSV and ground truth TSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--predictions", required=True, metavar="PATH",
        help="Path to matching_results.tsv (source1_entity_id, matched_entity_ids).",
    )
    parser.add_argument(
        "--ground-truth", required=True, metavar="PATH",
        help="Path to val_ground_truth.tsv or train_ground_truth.tsv.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    preds = load_id_list_tsv(args.predictions, id_col=S1_COL, list_col=MATCH_COL)
    gt = load_id_list_tsv(args.ground_truth, id_col=S1_COL, list_col=MATCH_COL)

    score = macro_f05_score(preds, gt)
    print(f"Macro F0.5: {score:.6f}")
    print(f"Entities scored: {len(gt)}")
    n_singleton_gt = sum(1 for v in gt.values() if not v)
    n_singleton_pred = sum(1 for v in preds.values() if not v)
    print(f"Singletons in ground truth: {n_singleton_gt}")
    print(f"Empty predictions (singletons): {n_singleton_pred}")


if __name__ == "__main__":
    main()
