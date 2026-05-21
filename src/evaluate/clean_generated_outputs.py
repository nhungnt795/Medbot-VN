import argparse
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
TESTSET_DIR = DATA_DIR / "testset"
OUTPUTS_DIR = PROJECT_DIR / "outputs"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
EVALUATION_DIR = OUTPUTS_DIR / "evaluation"

PROTECTED_DIRS = [
    DATA_DIR / "raw",
    DATA_DIR / "processed",
    PROJECT_DIR / "src",
]

TESTSET_PATTERNS = [
    "gold_testset_*.xlsx",
    "gold_testset_*.csv",
    "gold_testset_*_auto_checkpoint.jsonl",
    "gold_testset_*_rejected.jsonl",
    "retrieval_quality_*.xlsx",
    "retrieval_quality_*.csv",
    "retrieval_quality_*_summary.json",
]


def is_relative_to(path, parent):
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def is_protected(path):
    return any(is_relative_to(path, protected) for protected in PROTECTED_DIRS)


def collect_candidates():
    candidates = []

    if TESTSET_DIR.exists():
        for pattern in TESTSET_PATTERNS:
            candidates.extend(TESTSET_DIR.glob(pattern))

    for generated_dir in [PREDICTIONS_DIR, EVALUATION_DIR]:
        if generated_dir.exists():
            candidates.extend(path for path in generated_dir.rglob("*") if path.is_file())

    unique = []
    seen = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)

    return sorted(unique, key=lambda item: str(item).lower())


def main():
    parser = argparse.ArgumentParser(
        description="Delete generated testset, prediction, evaluation, chart, checkpoint, and rejected artifacts."
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    deleted = 0
    for path in collect_candidates():
        if is_protected(path):
            raise RuntimeError(f"Refusing to touch protected path: {path}")
        if not path.is_file():
            continue

        if args.dry_run:
            print("Would delete:", path)
        else:
            path.unlink()
            deleted += 1
            print("Deleted:", path)

    if args.dry_run:
        print("Dry run complete")
    else:
        print("Deleted files:", deleted)


if __name__ == "__main__":
    main()
