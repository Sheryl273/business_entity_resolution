"""
Owner: Person D (Evaluation / Validation / Packaging)

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
from pathlib import Path


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

    TODO(Person D): confirm this matches the problem statement's worked
    example exactly:
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


def macro_f05_score(predictions: dict[str, set[str]], ground_truth: dict[str, set[str]]) -> float:
    """
    Macro-average F_0.5 across all Source 1 entities in ground_truth.

    TODO(Person D):
      - Load predictions from a matching_results.tsv-shaped dict
      - Load ground_truth from train_ground_truth.tsv-shaped dict
      - Every entity in ground_truth must be scored, even if missing from
        predictions (treat as empty prediction for that entity)
    """
    scores = []
    for entity_id, true_ids in ground_truth.items():
        predicted_ids = predictions.get(entity_id, set())
        scores.append(score_entity(predicted_ids, true_ids))
    return sum(scores) / len(scores) if scores else 0.0


def load_id_list_tsv(path: str | Path, id_col: str, list_col: str) -> dict[str, set[str]]:
    """
    TODO(Person D): parse a TSV shaped like matching_results.tsv /
    train_ground_truth.tsv into {source1_entity_id: set(other_ids)}.
    Empty string -> empty set.
    """
    raise NotImplementedError


if __name__ == "__main__":
    # TODO(Person D): wire up CLI args, e.g.:
    #   python score_f05.py --predictions output/matching_results.tsv \
    #                        --ground-truth dataset/train/val_ground_truth.tsv
    pass
