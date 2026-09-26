# Contribution Guide — Work Division

Based directly on the problem statement's pipeline stages and confirmed against
the real dataset (train_source1 ≈ 2.2M rows, test_source1 ≈ 1.7M rows,
source2/source3 ≈ 5M rows each, US/India/France, Devanagari-script names present,
some empty address fields). Ordinary pandas loops will not scale here — use
`polars` or `duckdb` for anything touching a full file.

Every person's module has one **input interface** and one **output interface**.
Stick to these exactly so no one is blocked waiting on someone else's internals.

---

## Person A — Blocking / Candidate Generation
**Owns:** `src/normalize.py`, `src/blocking.py` → produces `output/candidate_pairs.tsv`

This is now separately graded (smaller candidate set per Source 1 entity ranks
higher), so treat it as a first-class deliverable, not throwaway.

**Responsibilities**
- [ ] EDA: name/address length distributions, country breakdown, missing-field rates, script mix (Latin vs Devanagari etc.)
- [ ] `normalize_name()` / `normalize_address()`: lowercase, strip legal suffixes (Corp/Corporation, Pvt/Private, Ltd/Limited), punctuation (`&` vs `and`), transliteration handling
- [ ] Blocking keys: token n-grams, phonetic codes (Soundex/Metaphone) on name tokens, address locality tokens — **always partition by country first**, never block across US/India/France
- [ ] Candidate generation at scale: inverted index or MinHash/LSH, not O(n×m) comparison
- [ ] Recall ceiling + reduction ratio measurement on the held-out validation split
- [ ] Final output: `output/candidate_pairs.tsv` — this must be the *last* filtering stage, i.e. exactly what gets fed to the matching model

**Interface out:** `candidate_pairs.tsv` with columns `source1_entity_id`, `candidate_entity_ids` (comma-separated S2-/S3- IDs, empty allowed, no dupes)

**Prompts to use:**
```
Write a Python normalization function for noisy business names that strips
legal suffixes (Corp/Corporation, Pvt/Private, Ltd/Limited), normalizes
punctuation (& vs and), and lowercases — using only stdlib/rapidfuzz, no
external APIs or lookups.

Write an address normalization function robust to missing components,
landmark references (e.g. "Near SBI ATM"), and reordered components across
US and Indian address formats, without hardcoding to only those two countries.

Implement country-partitioned candidate generation using MinHash LSH (or an
inverted index on normalized name/address tokens) over train_source1.tsv
(~2.2M rows) against train_source2.tsv and train_source3.tsv (~5M rows each),
using polars or duckdb for memory efficiency. Output candidate_pairs.tsv in
the exact required format.

Given train_ground_truth.tsv and my candidate_pairs.tsv, write a script that
computes blocking recall (fraction of true matches present in the candidate
set) and reduction ratio (fraction of all possible pairs eliminated).
```

---

## Person B — Feature Engineering
**Owns:** `src/features.py` → produces feature matrices for train/val/test

**Responsibilities**
- [ ] For every `(source1_entity_id, candidate_entity_id)` pair from Person A's output, compute similarity features
- [ ] Name features: Jaccard, Levenshtein/edit distance ratio, TF-IDF cosine, token overlap, abbreviation-aware match score
- [ ] Address features: component-wise similarity, partial/missing-field robustness, landmark handling
- [ ] Country-aware feature toggles — must degrade gracefully for France (unseen in training), never hardcoded to US/India only
- [ ] Vectorized computation (rapidfuzz batch mode / polars UDFs) — candidate pairs can be in the tens of millions after blocking
- [ ] Hand off clean, versioned feature files (train/val/test splits) to Person C

**Interface in:** `candidate_pairs.tsv` (Person A) + the three source TSVs
**Interface out:** feature matrix file(s), one row per `(source1_entity_id, candidate_entity_id)` pair, versioned by split

**Prompts to use:**
```
Write vectorized functions to compute Jaccard similarity, Levenshtein ratio,
and TF-IDF cosine similarity between two business name strings, applied
efficiently over a polars/pandas DataFrame of tens of millions of candidate
pairs (not a Python for-loop).

Write an address similarity feature set that tokenizes street/city/PIN when
present, handles missing components without crashing, and works across
US, Indian, and unseen (e.g. French) address formats without country-specific
hardcoding.

Given candidate_pairs.tsv and the three source TSVs, write a script that
joins in the raw fields and generates a feature matrix file for every
Source1-candidate pair, split consistently with Person D's train/val split.
```

---

## Person C — Matching Model
**Owns:** `src/train_model.py`, `src/predict.py` → produces `output/matching_results.tsv`

**Responsibilities**
- [ ] Train a classifier on Person B's feature matrix against `train_ground_truth.tsv` labels
- [ ] Prefer LightGBM/XGBoost on engineered features — fast at this scale and trivially MIT/Apache-2.0 compliant. If using any embedding/LLM component, confirm ≤8B params and MIT/Apache-2.0 license *before* committing to it
- [ ] Tune decision threshold specifically for **F_0.5** (precision weighted 2× over recall), not F1 — sweep thresholds on the validation split
- [ ] Handle multi-match entities (one Source1 → several valid matches) and singletons (predict empty list correctly — worth a full 1.0 each)
- [ ] Generate final `output/matching_results.tsv`, ensuring it's a strict subset of `output/candidate_pairs.tsv`

**Interface in:** feature matrices (Person B)
**Interface out:** `matching_results.tsv` with columns `source1_entity_id`, `matched_entity_ids`

**Prompts to use:**
```
Given a feature matrix and binary match labels derived from
train_ground_truth.tsv, train a LightGBM (or XGBoost) classifier and report
precision, recall, and F_0.5 on a held-out validation split.

Write a threshold-tuning script that sweeps decision thresholds and selects
the one maximizing macro-averaged F_0.5 per Source1 entity, correctly
scoring singleton entities as 1.0 (correct empty prediction) or 0.0 (any
false-positive match).

Write inference code that, for every Source1 test entity, filters candidates
above the tuned threshold, deduplicates matched IDs, and writes
matching_results.tsv in the exact required tab-separated format, guaranteeing
every matched ID also appears in candidate_pairs.tsv.
```

---

## Person D — Evaluation, Validation, Infra, Packaging
**Owns:** `src/score_f05.py`, `src/validate_and_package.py`, final submission zip, `Documentation_template.md`, README reproduction steps

**Responsibilities**
- [ ] Build the local macro-averaged F_0.5 scorer (per-entity, singleton logic included) — the team's self-check before every leaderboard upload
- [ ] Own the fixed train/val split used consistently by A/B/C so all recall/F_0.5 numbers are comparable
- [ ] Run `utils/validate_submission.py` (organizer-provided) before every submission
- [ ] Decide team infra: one shared strong machine/cloud instance vs. each member chunk-processing locally, given file sizes (500MB+ per source file)
- [ ] Assemble the final zip exactly per the required structure (`output/`, `code/business_entity_resolution/{src,README.md,requirements.txt}`, `Documentation_template.md`)
- [ ] Write `code/business_entity_resolution/README.md` with exact end-to-end reproduction steps
- [ ] Fill in `Documentation_template.md` from A/B/C's methodology as each finishes
- [ ] Sanity-check France (unseen-country) handling doesn't silently break anywhere in the pipeline
- [ ] Track candidate-set size over time with Person A, since it's now separately graded

**Interface in:** everything (candidate_pairs.tsv, matching_results.tsv, all source code)
**Interface out:** validated, zipped final submission package

**Prompts to use:**
```
Write a macro-averaged F_0.5 scorer that compares predicted vs ground-truth
matched_entity_ids per source1_entity_id, scoring singletons as 1.0 for a
correct empty prediction and 0.0 for any false-positive match, matching the
exact formula F_0.5 = (1.25 × P × R) / (0.25 × P + R).

Write an end-to-end pipeline runner script (data → blocking → features →
model → output) so the README's reproduction steps are guaranteed accurate,
and so it can be re-run cleanly before final submission.

Write a script that checks a batch of test entities including an unseen
country (France) are not dropped, mis-normalized, or silently mishandled
anywhere in the pipeline.

Write a packaging script that assembles output/, code/business_entity_resolution/,
and Documentation_template.md into <team_name>_submission.zip in the exact
required structure, and runs utils/validate_submission.py as a final check
before zipping.
```

---

## Shared rules for everyone

- No external data lookups, APIs, or geocoding services — normalization and
  matching use only the provided training data. Any violation is instant
  disqualification per the challenge rules.
- Final model (if any learned component beyond classical ML) must be
  MIT/Apache-2.0 licensed and ≤8B parameters.
- `matched_entity_ids` / `candidate_entity_ids`: S2-/S3- IDs only, no self-matches
  to Source 1, no IDs outside the test set, no duplicates within a list.
- Every Source 1 test entity needs exactly one row in both output files, even
  if empty (singleton).
- Run `utils/validate_submission.py` locally before every leaderboard upload —
  don't spend a submission attempt discovering a format bug.
