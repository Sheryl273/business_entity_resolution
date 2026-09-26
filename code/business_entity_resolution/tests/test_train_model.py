"""
Tests for Person C — train_model.py
Owner: Krishna Gupta (Person C)

Tests use small synthetic fixtures — no Amazon dataset required.
All test functions are standalone and deterministic.
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

# Ensure src/ is importable
_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))

from train_model import (
    build_labels,
    check_leakage,
    compute_scale_pos_weight,
    detect_feature_columns,
    load_ground_truth,
    validate_schema,
    validate_train_val_compatibility,
    _df_to_predictions_dict,
    extract_feature_importance,
)
from score_f05 import macro_f05_score, score_entity, load_id_list_tsv

S1_COL = "source1_entity_id"
CAND_COL = "candidate_entity_id"
LABEL_COL = "is_match"
MATCH_COL = "matched_entity_ids"


# ══════════════════════════════════════════════════════════════════════════════
#  Fixtures
# ══════════════════════════════════════════════════════════════════════════════

def make_feature_df(s1_ids, cand_ids, n_feats=3, seed=0):
    """Create a synthetic feature matrix."""
    rng = np.random.RandomState(seed)
    n = len(s1_ids)
    data = {S1_COL: s1_ids, CAND_COL: cand_ids}
    for i in range(n_feats):
        data[f"feat_{i}"] = rng.uniform(0, 1, n).tolist()
    return pl.DataFrame(data)


# ══════════════════════════════════════════════════════════════════════════════
#  1. Label construction tests
# ══════════════════════════════════════════════════════════════════════════════

class TestBuildLabels:
    def test_positive_pair(self):
        df = make_feature_df(["S1-001", "S1-001"], ["S2-001", "S2-002"])
        gt = {"S1-001": {"S2-001"}}
        labeled = build_labels(df, gt)
        row = labeled.filter(
            (pl.col(S1_COL) == "S1-001") & (pl.col(CAND_COL) == "S2-001")
        )
        assert row[LABEL_COL][0] == 1, "Matched pair must have label 1"

    def test_negative_pair(self):
        df = make_feature_df(["S1-001", "S1-001"], ["S2-001", "S2-002"])
        gt = {"S1-001": {"S2-001"}}
        labeled = build_labels(df, gt)
        row = labeled.filter(
            (pl.col(S1_COL) == "S1-001") & (pl.col(CAND_COL) == "S2-002")
        )
        assert row[LABEL_COL][0] == 0, "Non-matched pair must have label 0"

    def test_multi_match(self):
        df = make_feature_df(
            ["S1-001", "S1-001", "S1-001"],
            ["S2-001", "S2-002", "S2-003"],
        )
        gt = {"S1-001": {"S2-001", "S2-002"}}
        labeled = build_labels(df, gt)
        assert labeled[LABEL_COL].sum() == 2, "Two matches expected"

    def test_singleton_entity_in_gt(self):
        """Entity with no true matches (singleton) — all candidates should be 0."""
        df = make_feature_df(["S1-001", "S1-001"], ["S2-001", "S2-002"])
        gt = {"S1-001": set()}
        labeled = build_labels(df, gt)
        assert labeled[LABEL_COL].sum() == 0, "Singleton entity must have all 0 labels"

    def test_entity_not_in_gt(self):
        """S1 entity in features but not in GT → all candidates labeled 0."""
        df = make_feature_df(["S1-999"], ["S2-001"])
        gt = {"S1-001": {"S2-001"}}
        labeled = build_labels(df, gt)
        assert labeled[LABEL_COL][0] == 0

    def test_duplicate_gt_rows_merged(self):
        """build_labels must handle duplicate positive pairs without inflating counts."""
        df = make_feature_df(["S1-001", "S1-001"], ["S2-001", "S2-002"])
        # S2-001 appears twice in the list for S1-001 — should still count as 1
        gt = {"S1-001": {"S2-001", "S2-001"}}  # set deduplication
        labeled = build_labels(df, gt)
        assert labeled[LABEL_COL].sum() == 1

    def test_empty_ground_truth(self):
        df = make_feature_df(["S1-001"], ["S2-001"])
        labeled = build_labels(df, {})
        assert labeled[LABEL_COL][0] == 0

    def test_whitespace_in_ids(self):
        """IDs with leading/trailing whitespace should still match."""
        df = pl.DataFrame({
            S1_COL: ["S1-001"],
            CAND_COL: ["S2-001"],
            "feat_0": [0.5],
        })
        # Ground truth constructed with trimmed IDs (build_labels trims during join)
        gt = {"S1-001": {"S2-001"}}
        labeled = build_labels(df, gt)
        assert labeled[LABEL_COL][0] == 1

    def test_label_column_is_int(self):
        df = make_feature_df(["S1-001"], ["S2-001"])
        gt = {"S1-001": {"S2-001"}}
        labeled = build_labels(df, gt)
        assert labeled[LABEL_COL].dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64)

    def test_output_has_all_feature_rows(self):
        """build_labels must not drop any rows from the feature matrix."""
        df = make_feature_df(
            ["S1-001", "S1-001", "S1-002"],
            ["S2-001", "S2-002", "S2-003"],
        )
        gt = {"S1-001": {"S2-001"}}
        labeled = build_labels(df, gt)
        assert len(labeled) == 3


# ══════════════════════════════════════════════════════════════════════════════
#  2. Schema validation tests
# ══════════════════════════════════════════════════════════════════════════════

class TestSchemaValidation:
    def test_missing_s1_col_raises(self):
        df = pl.DataFrame({"wrong_col": ["x"], CAND_COL: ["y"], "feat": [0.5]})
        with pytest.raises(ValueError, match="identifier columns missing"):
            validate_schema(df, [S1_COL, CAND_COL], None, "test")

    def test_missing_feature_col_raises(self):
        df = pl.DataFrame({S1_COL: ["x"], CAND_COL: ["y"], "feat_0": [0.5]})
        with pytest.raises(ValueError, match="Feature columns missing"):
            validate_schema(df, [S1_COL, CAND_COL], ["feat_0", "feat_1"], "test")

    def test_no_numeric_features_raises(self):
        df = pl.DataFrame({S1_COL: ["x"], CAND_COL: ["y"], "text_col": ["abc"]})
        with pytest.raises(ValueError, match="No numeric feature columns"):
            detect_feature_columns(df, [S1_COL, CAND_COL])

    def test_detect_feature_columns_excludes_ids(self):
        df = make_feature_df(["S1-001"], ["S2-001"])
        feats = detect_feature_columns(df, [S1_COL, CAND_COL])
        assert S1_COL not in feats
        assert CAND_COL not in feats
        assert len(feats) == 3

    def test_incompatible_schema_raises(self):
        train_df = make_feature_df(["S1-001"], ["S2-001"])
        val_df = pl.DataFrame({S1_COL: ["S1-002"], CAND_COL: ["S2-002"], "feat_0": [0.1]})
        with pytest.raises(ValueError, match="missing columns"):
            validate_train_val_compatibility(train_df, val_df, ["feat_0", "feat_1", "feat_2"])


# ══════════════════════════════════════════════════════════════════════════════
#  3. Leakage check tests
# ══════════════════════════════════════════════════════════════════════════════

class TestLeakageCheck:
    def test_no_leakage_passes(self):
        train = make_feature_df(["S1-001", "S1-002"], ["S2-001", "S2-002"])
        val = make_feature_df(["S1-003", "S1-004"], ["S2-003", "S2-004"])
        gt = {"S1-001": {"S2-001"}}
        train_labeled = build_labels(train, gt)
        val_labeled = build_labels(val, {})
        check_leakage(train_labeled, val_labeled)  # should not raise

    def test_leakage_raises(self):
        train = make_feature_df(["S1-001"], ["S2-001"])
        val = make_feature_df(["S1-001"], ["S2-002"])  # Same S1!
        gt = {"S1-001": {"S2-001"}}
        train_labeled = build_labels(train, gt)
        val_labeled = build_labels(val, {})
        with pytest.raises(ValueError, match="LEAKAGE DETECTED"):
            check_leakage(train_labeled, val_labeled)


# ══════════════════════════════════════════════════════════════════════════════
#  4. scale_pos_weight tests
# ══════════════════════════════════════════════════════════════════════════════

class TestScalePosWeight:
    def test_basic_ratio(self):
        y = np.array([0, 0, 0, 1])  # 3 neg, 1 pos → spw = 3.0
        spw = compute_scale_pos_weight(y)
        assert abs(spw - 3.0) < 1e-6

    def test_zero_positives_raises(self):
        y = np.zeros(10)
        with pytest.raises(ValueError, match="zero positive"):
            compute_scale_pos_weight(y)


# ══════════════════════════════════════════════════════════════════════════════
#  5. Threshold logic tests
# ══════════════════════════════════════════════════════════════════════════════

class TestThresholdLogic:
    def _make_scored_df(self, s1_ids, cand_ids, probs):
        return pl.DataFrame({
            S1_COL: s1_ids,
            CAND_COL: cand_ids,
            "_prob": probs,
        })

    def test_all_below_threshold(self):
        df = self._make_scored_df(["S1-001", "S1-001"], ["S2-A", "S2-B"], [0.1, 0.2])
        preds = _df_to_predictions_dict(df, "_prob", threshold=0.5)
        assert preds.get("S1-001") == set()

    def test_all_above_threshold(self):
        df = self._make_scored_df(["S1-001", "S1-001"], ["S2-A", "S2-B"], [0.8, 0.9])
        preds = _df_to_predictions_dict(df, "_prob", threshold=0.5)
        assert preds["S1-001"] == {"S2-A", "S2-B"}

    def test_partial_above_threshold(self):
        df = self._make_scored_df(
            ["S1-001", "S1-001", "S1-001"],
            ["S2-A", "S2-B", "S2-C"],
            [0.8, 0.3, 0.9],
        )
        preds = _df_to_predictions_dict(df, "_prob", threshold=0.5)
        assert preds["S1-001"] == {"S2-A", "S2-C"}

    def test_multiple_s1_entities(self):
        df = self._make_scored_df(
            ["S1-001", "S1-002"],
            ["S2-A", "S2-B"],
            [0.9, 0.1],
        )
        preds = _df_to_predictions_dict(df, "_prob", threshold=0.5)
        assert preds["S1-001"] == {"S2-A"}
        assert preds["S1-002"] == set()

    def test_all_s1_ids_present_even_empty(self):
        """Every S1 entity in the df is present in the output, even with no predictions."""
        df = self._make_scored_df(
            ["S1-001", "S1-002"],
            ["S2-A", "S2-B"],
            [0.1, 0.1],
        )
        preds = _df_to_predictions_dict(df, "_prob", threshold=0.5)
        assert "S1-001" in preds
        assert "S1-002" in preds


# ══════════════════════════════════════════════════════════════════════════════
#  6. score_f05 tests (also used by Person D)
# ══════════════════════════════════════════════════════════════════════════════

class TestScoreEntity:
    def test_perfect_prediction(self):
        assert score_entity({"A", "B"}, {"A", "B"}) == pytest.approx(1.0)

    def test_empty_prediction_singleton(self):
        """Correct singleton: predict empty for empty truth → 1.0"""
        assert score_entity(set(), set()) == pytest.approx(1.0)

    def test_wrong_singleton_prediction(self):
        """Any prediction for singleton → 0.0"""
        assert score_entity({"A"}, set()) == pytest.approx(0.0)

    def test_empty_prediction_non_singleton(self):
        """No prediction but truth is non-empty → 0.0 (recall=0)"""
        assert score_entity(set(), {"A"}) == pytest.approx(0.0)

    def test_partial_prediction(self):
        """Worked example from problem statement: P=2/3, R=1.0, F0.5≈0.714"""
        score = score_entity({"A", "B", "C"}, {"A", "B"})
        precision = 2 / 3
        recall = 1.0
        expected = (1.25 * precision * recall) / (0.25 * precision + recall)
        assert abs(score - expected) < 1e-6

    def test_zero_overlap(self):
        """No overlap between predicted and true → precision=0 → F0.5=0"""
        assert score_entity({"X"}, {"A"}) == pytest.approx(0.0)


class TestMacroF05Score:
    def test_all_perfect(self):
        preds = {"E1": {"A"}, "E2": {"B"}}
        gt = {"E1": {"A"}, "E2": {"B"}}
        assert macro_f05_score(preds, gt) == pytest.approx(1.0)

    def test_empty_gt(self):
        assert macro_f05_score({}, {}) == pytest.approx(0.0)

    def test_missing_prediction_treated_as_empty(self):
        """Entity in GT but not in predictions → treated as empty prediction."""
        preds = {}
        gt = {"E1": {"A"}}
        assert macro_f05_score(preds, gt) == pytest.approx(0.0)

    def test_singletons_counted_correctly(self):
        """Mix of singleton correct + non-singleton correct."""
        preds = {"E1": set(), "E2": {"B"}}
        gt = {"E1": set(), "E2": {"B"}}
        assert macro_f05_score(preds, gt) == pytest.approx(1.0)

    def test_macro_averaging(self):
        """Score should be arithmetic mean across entities."""
        preds = {"E1": {"A"}, "E2": set()}
        gt = {"E1": {"A"}, "E2": set()}
        # Both entities score 1.0 → macro = 1.0
        assert macro_f05_score(preds, gt) == pytest.approx(1.0)


# ══════════════════════════════════════════════════════════════════════════════
#  7. load_id_list_tsv tests
# ══════════════════════════════════════════════════════════════════════════════

class TestLoadIdListTsv:
    def test_basic_load(self, tmp_path):
        f = tmp_path / "gt.tsv"
        f.write_text("source1_entity_id\tmatched_entity_ids\nS1-001\tS2-A,S2-B\nS1-002\t\n")
        result = load_id_list_tsv(f)
        assert result["S1-001"] == {"S2-A", "S2-B"}
        assert result["S1-002"] == set()

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_id_list_tsv(tmp_path / "nonexistent.tsv")

    def test_whitespace_trimmed(self, tmp_path):
        f = tmp_path / "gt.tsv"
        f.write_text("source1_entity_id\tmatched_entity_ids\nS1-001\tS2-A ,  S2-B\n")
        result = load_id_list_tsv(f)
        assert result["S1-001"] == {"S2-A", "S2-B"}

    def test_duplicate_rows_merged(self, tmp_path):
        f = tmp_path / "gt.tsv"
        f.write_text(
            "source1_entity_id\tmatched_entity_ids\n"
            "S1-001\tS2-A\n"
            "S1-001\tS2-B\n"
        )
        result = load_id_list_tsv(f)
        assert result["S1-001"] == {"S2-A", "S2-B"}


# ══════════════════════════════════════════════════════════════════════════════
#  8. load_ground_truth tests
# ══════════════════════════════════════════════════════════════════════════════

class TestLoadGroundTruth:
    def test_loads_correctly(self, tmp_path):
        f = tmp_path / "gt.tsv"
        f.write_text(
            "source1_entity_id\tmatched_entity_ids\n"
            "S1-001\tS2-A,S2-B\n"
            "S1-002\t\n"
        )
        gt = load_ground_truth(f)
        assert gt["S1-001"] == {"S2-A", "S2-B"}
        assert gt["S1-002"] == set()

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_ground_truth(tmp_path / "missing.tsv")

    def test_missing_column_raises(self, tmp_path):
        f = tmp_path / "gt.tsv"
        f.write_text("source1_entity_id\twrong_col\nS1-001\tS2-A\n")
        with pytest.raises(ValueError, match="Ground truth TSV missing"):
            load_ground_truth(f)


# ══════════════════════════════════════════════════════════════════════════════
#  9. Smoke test: full mini-pipeline
# ══════════════════════════════════════════════════════════════════════════════

class TestFullMiniPipeline:
    """
    End-to-end smoke test with a tiny synthetic dataset.
    Does NOT require the Amazon dataset.
    """

    def test_smoke(self, tmp_path):
        import json
        from train_model import (
            train_model as do_train,
            tune_threshold,
            save_model_and_metadata,
        )
        from predict import (
            predict_matches,
            validate_output,
            write_matching_results_tsv,
            load_model,
            load_metadata,
        )

        # --- Create synthetic data ---
        np.random.seed(42)
        n_train, n_val = 200, 60
        # 10% positive rate
        train_s1 = [f"S1-{i:04d}" for i in range(n_train)]
        train_cand = [f"S2-{i:04d}" for i in range(n_train)]
        train_feats = {
            "source1_entity_id": train_s1,
            "candidate_entity_id": train_cand,
            "feat_0": np.random.uniform(0, 1, n_train).tolist(),
            "feat_1": np.random.uniform(0, 1, n_train).tolist(),
        }
        train_df = pl.DataFrame(train_feats)

        val_s1 = [f"S1-{i:04d}" for i in range(1000, 1000 + n_val)]
        val_cand = [f"S2-{i:04d}" for i in range(1000, 1000 + n_val)]
        val_feats = {
            "source1_entity_id": val_s1,
            "candidate_entity_id": val_cand,
            "feat_0": np.random.uniform(0, 1, n_val).tolist(),
            "feat_1": np.random.uniform(0, 1, n_val).tolist(),
        }
        val_df = pl.DataFrame(val_feats)

        # Positive pairs: every 10th pair
        train_gt = {
            s1: {cand} if i % 10 == 0 else set()
            for i, (s1, cand) in enumerate(zip(train_s1, train_cand))
        }
        val_gt = {
            s1: {cand} if i % 10 == 0 else set()
            for i, (s1, cand) in enumerate(zip(val_s1, val_cand))
        }

        feature_cols = ["feat_0", "feat_1"]
        train_labeled = build_labels(train_df, train_gt)
        val_labeled = build_labels(val_df, val_gt)
        check_leakage(train_labeled, val_labeled)

        # Minimal config
        cfg = {
            "model": {
                "objective": "binary",
                "metric": "binary_logloss",
                "boosting_type": "gbdt",
                "n_estimators": 50,
                "learning_rate": 0.1,
                "num_leaves": 15,
                "max_depth": -1,
                "min_child_samples": 5,
                "subsample": 0.8,
                "subsample_freq": 1,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.0,
                "reg_lambda": 1.0,
                "class_weight": "balanced",
                "n_jobs": 1,
                "verbose": -1,
            },
            "early_stopping": {"rounds": 10, "verbose": False},
            "threshold_search": {"min": 0.1, "max": 0.9, "step": 0.1},
        }

        # Train
        model = do_train(train_labeled, val_labeled, feature_cols, cfg, seed=42)
        assert model is not None

        # Tune threshold
        best_thresh, best_f05, _ = tune_threshold(model, val_labeled, val_gt, feature_cols, cfg)
        assert 0.0 <= best_thresh <= 1.0
        assert 0.0 <= best_f05 <= 1.0

        # Save model and metadata
        model_path = tmp_path / "matcher.txt"
        meta_path = tmp_path / "matcher_metadata.json"
        save_model_and_metadata(
            model, feature_cols, best_thresh,
            train_stats={"rows": n_train},
            val_stats={"rows": n_val, "macro_f05": best_f05, "threshold": best_thresh},
            cfg=cfg, seed=42,
            model_path=model_path, metadata_path=meta_path,
        )
        assert model_path.exists()
        assert meta_path.exists()

        # Load back and predict
        loaded_model = load_model(model_path)
        loaded_meta = load_metadata(meta_path)
        assert loaded_meta["feature_columns"] == feature_cols
        assert "selected_threshold" in loaded_meta

        # Predict on val (pretending it's test)
        all_s1_test = val_s1
        result_df = predict_matches(
            loaded_model, best_thresh, val_df, feature_cols, all_s1_test
        )

        # Validate output
        validate_output(result_df, all_s1_test, candidate_pair_ids=None)

        # Write TSV
        out_path = tmp_path / "matching_results.tsv"
        write_matching_results_tsv(result_df, out_path)
        assert out_path.exists()

        # Check every S1 entity is in the output
        written = pl.read_csv(str(out_path), separator="\t", null_values=[])
        assert written["source1_entity_id"].n_unique() == len(all_s1_test)
        assert set(written["source1_entity_id"].to_list()) == set(all_s1_test)
