"""
Owner: Person C (Matching Model)

Runs inference over the test feature matrix and produces
output/matching_results.tsv — the ONLY file scored on the leaderboard.

Hard constraints from the problem statement (violating any of these causes
rejection, not just a low score):
  - Every Source 1 test entity must have exactly one row (empty string if no matches)
  - matched_entity_ids: S2-/S3- ids only, that exist in the test set
  - No duplicate ids within a list, no duplicate source1_entity_id rows
  - Every matched id must also appear in output/candidate_pairs.tsv
"""

from __future__ import annotations
from pathlib import Path
import polars as pl


def predict_matches(
    model,
    threshold: float,
    test_feature_df: pl.DataFrame,
    all_test_source1_ids: list[str],
) -> pl.DataFrame:
    """
    TODO(Person C):
      - Score all candidate pairs with `model`
      - Keep pairs with score >= threshold
      - Group by source1_entity_id into deduplicated matched_entity_ids lists
      - CRITICAL: include every id in all_test_source1_ids, even with an
        empty matched_entity_ids, so every Source 1 test entity has a row
    """
    raise NotImplementedError


def write_matching_results_tsv(matches_df: pl.DataFrame, out_path: str | Path) -> None:
    """
    Write output/matching_results.tsv in the exact required format:
      source1_entity_id<TAB>matched_entity_ids (comma-joined, no quoting, no header quirks)
    """
    raise NotImplementedError


if __name__ == "__main__":
    # TODO(Person C): wire up CLI args, e.g.:
    #   python predict.py --model models/matcher.txt --threshold 0.62 \
    #                      --features feature_matrices/test_features.parquet \
    #                      --source1 dataset/test/test_source1.tsv \
    #                      --out output/matching_results.tsv
    pass
