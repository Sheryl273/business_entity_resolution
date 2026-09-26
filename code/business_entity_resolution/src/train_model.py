"""
Owner: Person C (Matching Model)

Trains a match/no-match classifier on Person B's feature matrix, labeled via
train_ground_truth.tsv, and tunes the decision threshold for F_0.5 (NOT F1 —
precision is weighted 2x over recall in this challenge).

Model choice: prefer LightGBM/XGBoost on engineered features — fast at this
scale (candidate pairs can be tens of millions) and trivially MIT/Apache-2.0
license-compliant. If any embedding/LLM component is used instead or in
addition, confirm license + <=8B params BEFORE relying on it — this is a hard
submission constraint.
"""

from __future__ import annotations
from pathlib import Path
import polars as pl

from features import FEATURE_COLUMNS
from score_f05 import macro_f05_score


def build_labels(feature_df: pl.DataFrame, ground_truth_path: str | Path) -> pl.DataFrame:
    """
    TODO(Person C): join feature_df to train_ground_truth.tsv, producing a
    binary `is_match` column: 1 if candidate_entity_id is in the entity's
    matched_entity_ids, else 0.
    """
    raise NotImplementedError


def train(feature_df: pl.DataFrame, label_col: str = "is_match"):
    """
    TODO(Person C): train LightGBM/XGBoost classifier on FEATURE_COLUMNS.
    Return the fitted model.
    """
    raise NotImplementedError


def tune_threshold(model, val_feature_df: pl.DataFrame, val_ground_truth_path: str | Path) -> float:
    """
    TODO(Person C): sweep candidate thresholds (e.g. 0.05 to 0.95 step 0.01),
    score each with macro_f05_score() from score_f05.py (singletons included),
    and return the threshold that maximizes macro F_0.5 — not F1, not accuracy.
    """
    raise NotImplementedError


if __name__ == "__main__":
    # TODO(Person C): wire up CLI args, e.g.:
    #   python train_model.py --features feature_matrices/train_features.parquet \
    #                          --ground-truth dataset/train/train_ground_truth.tsv \
    #                          --out models/matcher.txt
    pass
