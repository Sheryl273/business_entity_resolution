"""
Owner: Person D (Evaluation / Validation / Packaging)

1. Runs the organizer-provided utils/validate_submission.py against our
   output/ files (drop that script into utils/ locally — it's provided by
   the challenge, not authored here).
2. Assembles the final <team_name>_submission.zip in the exact required
   structure once validation passes.

Required zip structure (from problem statement):
  <team_name>_submission.zip
  ├── output/
  │   ├── matching_results.tsv
  │   └── candidate_pairs.tsv
  ├── code/
  │   └── business_entity_resolution/
  │       ├── src/
  │       ├── README.md
  │       └── requirements.txt
  └── Documentation_template.md
"""

from __future__ import annotations
import subprocess
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]  # .../business_entity_resolution/


def run_validator(
    matching_path: str = "output/matching_results.tsv",
    candidate_path: str = "output/candidate_pairs.tsv",
    test_dir: str = "dataset/test",
) -> bool:
    """
    Runs utils/validate_submission.py (organizer-provided). Returns True on
    PASS (exit 0), prints issues and returns False otherwise.

    TODO(Person D): once utils/validate_submission.py is dropped in locally,
    confirm the exact CLI flags match the problem statement:
      python3 utils/validate_submission.py --matching <path> --candidate <path> --test-dir <dir>
    """
    result = subprocess.run(
        [
            sys.executable, str(REPO_ROOT / "utils" / "validate_submission.py"),
            "--matching", matching_path,
            "--candidate", candidate_path,
            "--test-dir", test_dir,
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        return False
    return True


def package_submission(team_name: str, out_zip_dir: str = ".") -> Path:
    """
    TODO(Person D): copy output/, code/business_entity_resolution/, and
    Documentation_template.md into a staging dir named <team_name>_submission/
    and zip it as <team_name>_submission.zip. Only call this after
    run_validator() returns True.
    """
    raise NotImplementedError


if __name__ == "__main__":
    # TODO(Person D): wire up CLI, e.g.:
    #   python validate_and_package.py --team-name AwesomeTeam
    pass
