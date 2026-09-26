# Business Entity Resolution — Amazon ML Challenge (Unstop)

Team repo for the Business Entity Resolution challenge. Matches noisy business
records from Source 2 / Source 3 against the deduplicated Source 1 reference,
across US, India, and (test-only) France.

Pipeline: **normalize → block (candidate generation) → feature engineer → match/score → validate/package**

Final scored artifact: `output/matching_results.tsv`
Also required: `output/candidate_pairs.tsv` (now separately graded — smaller candidate
sets per Source 1 entity rank higher, so blocking quality matters beyond recall).

---

## 1. Data — NOT in this repo

The dataset is ~2.4 GB total (millions of rows per file) and is **git-ignored**.
Every member downloads it independently from the challenge portal and places it at:

```
dataset/train/train_source1.tsv
dataset/train/train_source2.tsv
dataset/train/train_source3.tsv
dataset/train/train_ground_truth.tsv
dataset/test/test_source1.tsv
dataset/test/test_source2.tsv
dataset/test/test_source3.tsv
utils/validate_submission.py   # from the organizer's student_resource/ zip
```

Never `git add` anything under `dataset/` or large files under `output/`. The
`.gitignore` already blocks this, but double-check before every push.

---

## 2. Setup

```bash
git clone <your-repo-url>.git
cd business_entity_resolution
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r code/business_entity_resolution/requirements.txt
```

Drop the dataset files into `dataset/train/` and `dataset/test/` as shown above.

---

## 3. Git workflow (4-person team)

**One-time (team lead):**
```bash
git init
git add .
git commit -m "Initial repo skeleton"
git branch -M main
git remote add origin <your-repo-url>.git
git push -u origin main
```

**Everyone else — first time:**
```bash
git clone <your-repo-url>.git
cd business_entity_resolution
```

**Each person works on their own branch, named after their module:**
```bash
git checkout -b feature/blocking        # Person A
git checkout -b feature/features        # Person B
git checkout -b feature/matching-model  # Person C
git checkout -b feature/eval-packaging  # Person D
```

**Daily loop:**
```bash
git add src/<your_file>.py
git commit -m "blocking: add country-partitioned LSH candidate generation"
git push -u origin feature/blocking
```
Open a Pull Request into `main` once your module's interface (see
`CONTRIBUTING.md`) is stable enough for teammates to build on. Don't wait
until it's "done" — the interfaces are what unblock everyone else.

**Before merging, always pull latest main first:**
```bash
git checkout main
git pull origin main
git checkout feature/blocking
git merge main
```

**Keeping large intermediate files (feature matrices, model artifacts) out of git:**
Add any file over ~10MB you generate locally to `.gitignore` immediately —
don't rely on remembering later.

---

## 4. Repo structure

```
business_entity_resolution/
├── README.md
├── CONTRIBUTING.md              # work division, owners, prompts
├── .gitignore
├── dataset/                     # git-ignored, populate locally
│   ├── train/
│   └── test/
├── utils/
│   └── validate_submission.py   # organizer-provided, drop in locally
├── output/                      # git-ignored except .gitkeep
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── Documentation_template.md    # fill in before final submission
└── code/business_entity_resolution/
    ├── README.md                # exact reproduction steps (fill in as pipeline solidifies)
    ├── requirements.txt
    └── src/
        ├── normalize.py         # Person A
        ├── blocking.py          # Person A
        ├── features.py          # Person B
        ├── train_model.py       # Person C
        ├── predict.py           # Person C
        ├── score_f05.py         # Person D
        └── validate_and_package.py  # Person D
```

See `CONTRIBUTING.md` for exact ownership, interfaces, and prompts to use
with an AI coding assistant for each piece.
