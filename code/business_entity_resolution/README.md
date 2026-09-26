# business_entity_resolution — pipeline

Self-contained pipeline code. This folder is what ships inside the final
submission zip under `code/business_entity_resolution/`.

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Reproduce end-to-end (fill in exact commands as each stage is implemented)

```bash
# 1. Blocking / candidate generation  (Person A)
python src/blocking.py \
  --source1 ../../dataset/test/test_source1.tsv \
  --source2 ../../dataset/test/test_source2.tsv \
  --source3 ../../dataset/test/test_source3.tsv \
  --out ../../output/candidate_pairs.tsv

# 2. Feature engineering  (Person B)
python src/features.py \
  --candidates ../../output/candidate_pairs.tsv \
  --source1 ../../dataset/test/test_source1.tsv \
  --source2 ../../dataset/test/test_source2.tsv \
  --source3 ../../dataset/test/test_source3.tsv \
  --out feature_matrices/test_features.parquet

# 3. Train model on training split  (Person C — run once, not per test run)
python src/train_model.py \
  --features feature_matrices/train_features.parquet \
  --ground-truth ../../dataset/train/train_ground_truth.tsv \
  --out models/matcher.txt

# 4. Predict on test set  (Person C)
python src/predict.py \
  --model models/matcher.txt --threshold 0.62 \
  --features feature_matrices/test_features.parquet \
  --source1 ../../dataset/test/test_source1.tsv \
  --out ../../output/matching_results.tsv

# 5. Validate + package  (Person D)
python src/validate_and_package.py --team-name <YourTeamName>
```

## Notes
- Update the `--threshold` value above once Person C finalizes it via
  `tune_threshold()` in `train_model.py`.
- `utils/validate_submission.py` is organizer-provided; place it at
  `../../utils/validate_submission.py` locally (not committed here since it's
  supplied separately in the challenge's `student_resource/` zip).
