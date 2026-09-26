"""
Owner: Person A (Blocking / Candidate Generation)

Produces output/candidate_pairs.tsv — the LAST-stage candidate set fed to the
matching model. This is now separately graded (smaller candidate sets per
Source 1 entity rank higher), so treat reduction ratio as a real objective,
not just recall.

Real scale (confirmed from actual files):
  train_source1  ~2.2M rows | test_source1  ~1.7M rows
  train_source2  ~5.0M rows | test_source2   ~4.9M rows
  train_source3  ~5.3M rows | test_source3   ~5.1M rows

Naive O(n*m) comparison is not feasible. Use an inverted index / MinHash LSH
and ALWAYS partition by country first — never generate candidates across
different country labels.

Input: dataset/train/*.tsv or dataset/test/*.tsv (paths, not hardcoded — accept
       as CLI args or function params so Person D can call this in the
       end-to-end pipeline runner)
Output: output/candidate_pairs.tsv
  columns: source1_entity_id, candidate_entity_ids
  candidate_entity_ids: comma-separated S2-/S3- ids, empty allowed, no dupes
  MUST include every source1_entity_id present in the input source1 file,
  even with an empty candidate list.
"""

from __future__ import annotations
from pathlib import Path
import polars as pl

from normalize import normalize_name, normalize_address


def load_source(path: str | Path) -> pl.DataFrame:
    """Load a source TSV with explicit tab separator (see problem statement warning)."""
    return pl.read_csv(path, separator="\t")


def build_blocking_keys(df: pl.DataFrame) -> pl.DataFrame:
    """
    TODO(Person A):
      - Add a normalized_name column via normalize_name()
      - Add blocking key column(s): e.g. sorted name-token n-grams, phonetic
        code (Soundex/Metaphone) on first token, address locality token
      - Keep `country` as-is for partitioning — do not filter/one-hot it
    """
    raise NotImplementedError


def generate_candidates(
    source1_df: pl.DataFrame,
    source2_df: pl.DataFrame,
    source3_df: pl.DataFrame,
    top_k: int = 20,
) -> pl.DataFrame:
    """
    TODO(Person A): country-partitioned inverted-index / MinHash LSH candidate
    generation. For each source1 entity, return up to top_k plausible S2-/S3-
    candidates.

    Must guarantee:
      - every source1_entity_id appears exactly once in the output (empty list ok)
      - no cross-country candidates
      - no duplicate candidate ids per row
    """
    raise NotImplementedError


def write_candidate_pairs_tsv(candidates_df: pl.DataFrame, out_path: str | Path) -> None:
    """
    Write output/candidate_pairs.tsv in the exact required format:
      source1_entity_id<TAB>candidate_entity_ids (comma-joined, no quoting)
    """
    raise NotImplementedError


if __name__ == "__main__":
    # TODO(Person A): wire up CLI args for train/test mode, e.g.:
    #   python blocking.py --source1 dataset/test/test_source1.tsv \
    #                       --source2 dataset/test/test_source2.tsv \
    #                       --source3 dataset/test/test_source3.tsv \
    #                       --out output/candidate_pairs.tsv
    pass
