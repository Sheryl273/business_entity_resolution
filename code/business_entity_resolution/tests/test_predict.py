"""
Tests for Person C — predict.py
Owner: Krishna Gupta (Person C)

Tests use small synthetic fixtures — no Amazon dataset required.
"""

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))

from predict import (
    validate_output,
    predict_matches,
    write_matching_results_tsv,
    validate_test_schema,
    load_source1_ids,
)

S1_COL = "source1_entity_id"
CAND_COL = "candidate_entity_id"
MATCH_COL = "matched_entity_ids"


def make_mock_model(constant_prob: float = 0.8):
    """Return a mock model whose predict() always returns a fixed probability."""
    class MockModel:
        def predict(self, X):
            return np.full(len(X), constant_prob)
    return MockModel()


def make_feature_df(s1_ids, cand_ids, n_feats=2, seed=0):
    rng = np.random.RandomState(seed)
    n = len(s1_ids)
    data = {S1_COL: s1_ids, CAND_COL: cand_ids}
    for i in range(n_feats):
        data[f"feat_{i}"] = rng.uniform(0, 1, n).tolist()
    return pl.DataFrame(data)


# ══════════════════════════════════════════════════════════════════════════════
#  1. Prediction formatting tests
# ══════════════════════════════════════════════════════════════════════════════

class TestPredictMatches:
    feature_cols = ["feat_0", "feat_1"]

    def test_all_source1_ids_preserved(self):
        """Every S1 entity must appear in the output — even singletons."""
        df = make_feature_df(["S1-001", "S1-002"], ["S2-A", "S2-B"])
        all_s1 = ["S1-001", "S1-002", "S1-003"]  # S1-003 has no candidates
        model = make_mock_model(0.0)  # below threshold
        result = predict_matches(model, 0.5, df, self.feature_cols, all_s1)
        output_s1 = set(result[S1_COL].to_list())
        assert output_s1 == {"S1-001", "S1-002", "S1-003"}

    def test_empty_match_string_for_singletons(self):
        """S1 with no candidates above threshold gets empty matched_entity_ids."""
        df = make_feature_df(["S1-001"], ["S2-A"])
        model = make_mock_model(0.1)  # always below 0.5
        result = predict_matches(model, 0.5, df, self.feature_cols, ["S1-001"])
        assert result.filter(pl.col(S1_COL) == "S1-001")[MATCH_COL][0] == ""

    def test_multiple_candidate_ids_joined(self):
        """Multi-match: A,B not just A."""
        df = make_feature_df(
            ["S1-001", "S1-001", "S1-001"],
            ["S2-A", "S2-B", "S2-C"],
        )
        model = make_mock_model(0.9)  # all above any threshold
        result = predict_matches(model, 0.5, df, self.feature_cols, ["S1-001"])
        matched = result.filter(pl.col(S1_COL) == "S1-001")[MATCH_COL][0]
        ids = set(matched.split(","))
        assert ids == {"S2-A", "S2-B", "S2-C"}

    def test_partial_threshold(self):
        """Only candidates strictly above threshold are included."""
        df = make_feature_df(
            ["S1-001", "S1-001", "S1-001"],
            ["S2-A", "S2-B", "S2-C"],
        )
        # MockModel always returns constant — we need custom probs
        class VariableModel:
            def predict(self, X):
                return np.array([0.9, 0.3, 0.8])  # only S2-A and S2-C pass

        result = predict_matches(
            VariableModel(), 0.5, df, self.feature_cols, ["S1-001"]
        )
        matched = result.filter(pl.col(S1_COL) == "S1-001")[MATCH_COL][0]
        ids = set(matched.split(","))
        assert ids == {"S2-A", "S2-C"}

    def test_no_predictions_gives_empty_string(self):
        """When model predicts zero above threshold → matched_entity_ids = ''"""
        df = make_feature_df(["S1-001"], ["S2-A"])
        model = make_mock_model(0.0)
        result = predict_matches(model, 0.5, df, self.feature_cols, ["S1-001"])
        assert result[MATCH_COL][0] == ""

    def test_deterministic_ordering(self):
        """Running predict twice with same input gives identical output."""
        df = make_feature_df(
            ["S1-001", "S1-001"],
            ["S2-ZZZ", "S2-AAA"],
        )
        model = make_mock_model(0.9)
        r1 = predict_matches(model, 0.5, df, self.feature_cols, ["S1-001"])
        r2 = predict_matches(model, 0.5, df, self.feature_cols, ["S1-001"])
        assert r1[MATCH_COL][0] == r2[MATCH_COL][0]

    def test_no_duplicate_candidate_ids_in_list(self):
        """Duplicate candidate IDs must be deduplicated."""
        df = pl.DataFrame({
            S1_COL: ["S1-001", "S1-001"],
            CAND_COL: ["S2-A", "S2-A"],  # Duplicate candidate
            "feat_0": [0.9, 0.9],
            "feat_1": [0.1, 0.1],
        })
        model = make_mock_model(0.8)
        result = predict_matches(model, 0.5, df, self.feature_cols, ["S1-001"])
        matched = result[MATCH_COL][0]
        ids = [x for x in matched.split(",") if x]
        assert len(ids) == len(set(ids)), "Duplicate candidate IDs found in output"


# ══════════════════════════════════════════════════════════════════════════════
#  2. Output validation tests
# ══════════════════════════════════════════════════════════════════════════════

class TestValidateOutput:
    def _make_output_df(self, s1_ids, matched):
        return pl.DataFrame({S1_COL: s1_ids, MATCH_COL: matched})

    def test_valid_output_passes(self):
        df = self._make_output_df(["S1-001", "S1-002"], ["S2-A", ""])
        validate_output(df, ["S1-001", "S1-002"], candidate_pair_ids=None)

    def test_missing_source1_raises(self):
        df = self._make_output_df(["S1-001"], ["S2-A"])
        with pytest.raises(ValueError, match="missing from output"):
            validate_output(df, ["S1-001", "S1-002"], candidate_pair_ids=None)

    def test_duplicate_source1_raises(self):
        df = self._make_output_df(["S1-001", "S1-001"], ["S2-A", "S2-B"])
        with pytest.raises(ValueError, match="Duplicate source1_entity_id"):
            validate_output(df, ["S1-001"], candidate_pair_ids=None)

    def test_nan_text_in_matched_ids_raises(self):
        df = self._make_output_df(["S1-001"], ["None"])
        with pytest.raises(ValueError, match="literal 'None'"):
            validate_output(df, ["S1-001"], candidate_pair_ids=None)

    def test_candidate_not_in_candidate_pairs_raises(self):
        df = self._make_output_df(["S1-001"], ["S2-UNKNOWN"])
        with pytest.raises(ValueError, match="not found in candidate_pairs.tsv"):
            validate_output(df, ["S1-001"], candidate_pair_ids={"S2-KNOWN"})

    def test_empty_candidate_pair_ids_skips_validation(self):
        """Empty candidate_pair_ids set → skip that check."""
        df = self._make_output_df(["S1-001"], [""])
        validate_output(df, ["S1-001"], candidate_pair_ids=set())

    def test_empty_match_is_valid(self):
        """Empty string is valid for singletons."""
        df = self._make_output_df(["S1-001", "S1-002"], ["", ""])
        validate_output(df, ["S1-001", "S1-002"], candidate_pair_ids=None)


# ══════════════════════════════════════════════════════════════════════════════
#  3. Schema validation tests
# ══════════════════════════════════════════════════════════════════════════════

class TestValidateTestSchema:
    def test_missing_feature_raises(self):
        df = pl.DataFrame({
            S1_COL: ["S1-001"],
            CAND_COL: ["S2-001"],
            "feat_0": [0.5],
        })
        with pytest.raises(ValueError, match="Feature columns from training metadata"):
            validate_test_schema(df, ["feat_0", "feat_MISSING"])

    def test_missing_id_col_raises(self):
        df = pl.DataFrame({
            "wrong_id": ["S1-001"],
            CAND_COL: ["S2-001"],
            "feat_0": [0.5],
        })
        with pytest.raises(ValueError, match="identifier columns missing"):
            validate_test_schema(df, ["feat_0"])

    def test_column_reordering(self):
        """Schema validator should reorder columns to match training order."""
        df = pl.DataFrame({
            "feat_1": [0.5],
            "feat_0": [0.3],
            S1_COL: ["S1-001"],
            CAND_COL: ["S2-001"],
        })
        result = validate_test_schema(df, ["feat_0", "feat_1"])
        # ID cols first, then features in training order
        assert result.columns == [S1_COL, CAND_COL, "feat_0", "feat_1"]


# ══════════════════════════════════════════════════════════════════════════════
#  4. write_matching_results_tsv tests
# ══════════════════════════════════════════════════════════════════════════════

class TestWriteMatchingResultsTsv:
    def test_writes_tab_separated(self, tmp_path):
        df = pl.DataFrame({
            S1_COL: ["S1-001", "S1-002"],
            MATCH_COL: ["S2-A,S2-B", ""],
        })
        out = tmp_path / "results.tsv"
        write_matching_results_tsv(df, out)
        content = out.read_text()
        assert "\t" in content
        assert "S1-001\tS2-A,S2-B" in content or "S1-001\t" in content

    def test_all_entities_in_output(self, tmp_path):
        df = pl.DataFrame({
            S1_COL: ["S1-001", "S1-002", "S1-003"],
            MATCH_COL: ["S2-A", "", "S2-B,S2-C"],
        })
        out = tmp_path / "results.tsv"
        write_matching_results_tsv(df, out)
        written = pl.read_csv(str(out), separator="\t", null_values=[])
        assert set(written[S1_COL].to_list()) == {"S1-001", "S1-002", "S1-003"}

    def test_sorted_output(self, tmp_path):
        df = pl.DataFrame({
            S1_COL: ["S1-ZZZ", "S1-AAA", "S1-MMM"],
            MATCH_COL: ["S2-1", "S2-2", "S2-3"],
        })
        out = tmp_path / "results.tsv"
        write_matching_results_tsv(df, out)
        written = pl.read_csv(str(out), separator="\t", null_values=[])
        ids = written[S1_COL].to_list()
        assert ids == sorted(ids)

    def test_load_source1_ids(self, tmp_path):
        f = tmp_path / "test_source1.tsv"
        f.write_text("source1_entity_id\tname\nS1-001\tAcme Corp\nS1-002\tBeta Inc\n")
        ids = load_source1_ids(f)
        assert set(ids) == {"S1-001", "S1-002"}
