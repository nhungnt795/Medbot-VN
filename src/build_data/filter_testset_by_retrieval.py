import argparse
from pathlib import Path

import pandas as pd

from retrieval.config import TESTSET_DIR


def target_testset_path(target):
    return TESTSET_DIR / f"gold_testset_{target}.xlsx"


def target_testset_csv_path(target):
    return TESTSET_DIR / f"gold_testset_{target}.csv"


def retrieval_report_path(method, k, target):
    return TESTSET_DIR / f"retrieval_quality_{method}_k{k}_{target}.xlsx"


def filtered_output_paths(target):
    return (
        TESTSET_DIR / f"gold_testset_{target}_retrieval_filtered.xlsx",
        TESTSET_DIR / f"gold_testset_{target}_retrieval_filtered.csv"
    )


def read_table(path, sheet_name=None):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path, sheet_name=sheet_name if sheet_name is not None else 0)

    return pd.read_csv(path)


def normalize_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def is_truthy_hit(value):
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "1.0", "true", "yes"}


def should_keep_row(row, hit_col, keep_non_context, require_validation_passed):
    expected_behavior = normalize_text(row.get("expected_behavior", ""))
    source_generation = normalize_text(row.get("source_generation", ""))
    validation_status = normalize_text(row.get("validation_status", ""))

    if require_validation_passed and source_generation.startswith("llm") and validation_status and validation_status != "passed":
        return False, "validation_not_passed"

    if expected_behavior != "answer_from_context":
        if keep_non_context:
            return True, "non_context_kept"
        return False, "non_context_dropped"

    if is_truthy_hit(row.get(hit_col)):
        return True, "retrieval_hit"

    return False, "retrieval_miss"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=300)
    parser.add_argument("--method", default="mmr")
    parser.add_argument("--testset", default=None)
    parser.add_argument("--retrieval-report", default=None)
    parser.add_argument("--output-xlsx", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--hit-k", type=int, default=5)
    parser.add_argument("--require", choices=["any", "all"], default="any")
    parser.add_argument("--drop-non-context", action="store_true")
    parser.add_argument("--allow-unvalidated-llm", action="store_true")
    parser.add_argument("--overwrite-main", action="store_true")
    parser.add_argument("--max-rows", type=int, default=0)
    args = parser.parse_args()

    default_output_xlsx, default_output_csv = filtered_output_paths(args.target)
    testset_path = Path(args.testset) if args.testset else target_testset_path(args.target)
    retrieval_report = (
        Path(args.retrieval_report)
        if args.retrieval_report
        else retrieval_report_path(args.method, args.hit_k, args.target)
    )
    output_xlsx = Path(args.output_xlsx) if args.output_xlsx else default_output_xlsx
    output_csv = Path(args.output_csv) if args.output_csv else default_output_csv

    testset_df = read_table(testset_path)
    report_df = read_table(retrieval_report, sheet_name="details")

    if "question_id" not in testset_df.columns:
        raise ValueError("Testset is missing required column: question_id")
    if "question_id" not in report_df.columns:
        raise ValueError("Retrieval report is missing required column: question_id")

    hit_col = f"hit_{args.require}_at_{args.hit_k}"
    if hit_col not in report_df.columns:
        raise ValueError(f"Retrieval report is missing required column: {hit_col}")

    report_cols = [
        "question_id",
        "retrieval_evaluated",
        hit_col,
        f"matched_at_{args.hit_k}",
        "retrieved_sources_json"
    ]
    report_cols = [col for col in report_cols if col in report_df.columns]

    merged = testset_df.merge(
        report_df[report_cols],
        on="question_id",
        how="left",
        suffixes=("", "_retrieval")
    )

    decisions = merged.apply(
        lambda row: should_keep_row(
            row=row,
            hit_col=hit_col,
            keep_non_context=not args.drop_non_context,
            require_validation_passed=not args.allow_unvalidated_llm
        ),
        axis=1
    )
    merged["retrieval_filter_keep"] = [int(item[0]) for item in decisions]
    merged["retrieval_filter_reason"] = [item[1] for item in decisions]

    filtered = merged[merged["retrieval_filter_keep"] == 1].copy()
    if args.max_rows > 0:
        filtered = filtered.head(args.max_rows)

    output_xlsx.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    filtered.to_excel(output_xlsx, index=False)
    filtered.to_csv(output_csv, index=False, encoding="utf-8-sig")

    if args.overwrite_main:
        filtered.to_excel(target_testset_path(args.target), index=False)
        filtered.to_csv(target_testset_csv_path(args.target), index=False, encoding="utf-8-sig")

    print("Done filtering testset by retrieval")
    print("Input rows:", len(testset_df))
    print("Kept rows:", len(filtered))
    print("Dropped rows:", len(testset_df) - len(filtered))
    print("Reasons:")
    print(merged["retrieval_filter_reason"].value_counts())
    print("Saved:", output_xlsx)
    print("Saved:", output_csv)
    if args.overwrite_main:
        print("Overwritten:", target_testset_path(args.target))
        print("Overwritten:", target_testset_csv_path(args.target))


if __name__ == "__main__":
    main()
