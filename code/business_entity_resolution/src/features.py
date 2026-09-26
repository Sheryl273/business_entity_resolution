"""
Owner: Person B (Feature Engineering)

Consumes output/candidate_pairs.tsv (Person A) + the three source TSVs, and
produces a feature matrix: one row per (source1_entity_id, candidate_entity_id)
pair, with similarity features consumed directly by Person C's model.

Candidate pair counts after blocking can still be in the tens of millions —
use vectorized/batched computation (rapidfuzz batch APIs, polars expressions),
never a plain Python for-loop over pairs.
"""

from __future__ import annotations
from pathlib import Path
import polars as pl

from normalize import normalize_name, normalize_address


FEATURE_COLUMNS = [
    "name_jaccard",
    "name_levenshtein_ratio",
    "name_tfidf_cosine",
    "name_token_overlap",
    "address_token_overlap",
    "address_locality_match",
    # TODO(Person B): finalize full feature list from EDA + experiments,
    # keep this list in sync with train_model.py's expected columns
]


def load_candidate_pairs(path: str | Path) -> pl.DataFrame:
    """
    Load candidate_pairs.tsv and explode into one row per
    (source1_entity_id, candidate_entity_id) pair (currently comma-joined per row).
    """
    raise NotImplementedError


def compute_name_features(name_a: pl.Series, name_b: pl.Series) -> pl.DataFrame:
    """
    TODO(Person B): vectorized Jaccard, Levenshtein ratio, TF-IDF cosine,
    token overlap between two name Series. Must not assume Latin-only script —
    Devanagari names appear in real Source 2 India records.
    """
    raise NotImplementedError


def compute_address_features(addr_a: pl.Series, addr_b: pl.Series, country: pl.Series) -> pl.DataFrame:
    """
    TODO(Person B): address similarity features. Must degrade gracefully when
    business_address is empty (confirmed present in real data) and must not
    hardcode logic to only {US, India} — France appears in test with no
    training examples.
    """
    raise NotImplementedError


def build_feature_matrix(
    candidate_pairs_path: str | Path,
    source1_path: str | Path,
    source2_path: str | Path,
    source3_path: str | Path,
    out_path: str | Path,
) -> None:
    """
    End-to-end: join candidate pairs to raw fields, compute all features in
    FEATURE_COLUMNS, write out a feature matrix file (parquet recommended over
    CSV for size/speed — coordinate the format with Person C).
    """
    raise NotImplementedError


if __name__ == "__main__":
    # TODO(Person B): wire up CLI args, e.g.:
    #   python features.py --candidates output/candidate_pairs.tsv \
    #                       --source1 dataset/test/test_source1.tsv \
    #                       --source2 dataset/test/test_source2.tsv \
    #                       --source3 dataset/test/test_source3.tsv \
    #                       --out feature_matrices/test_features.parquet
    pass
