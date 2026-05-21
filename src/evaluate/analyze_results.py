import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from retrieval.config import EVALUATION_DIR


COMMON_METRICS = [
    "answer_correctness",
    "answer_relevancy",
    "answer_completeness",
    "medical_safety"
]

RAG_METRICS = [
    "faithfulness",
    "context_precision",
    "context_recall"
]

DETAILED_SCORE_METRICS = [
    "answer_correctness",
    "answer_relevancy",
    "answer_completeness",
    "medical_safety",
    "faithfulness",
    "context_precision",
    "context_recall"
]

OTHER_METRICS = [
    "latency_seconds"
]


def flatten_columns(df):
    new_columns = []

    for col in df.columns:
        if isinstance(col, tuple):
            parts = [str(x) for x in col if str(x) and str(x) != "nan"]
            new_columns.append("_".join(parts).strip("_"))
        else:
            new_columns.append(str(col))

    df.columns = new_columns
    return df


def make_summary(df, group_cols, metrics):
    available_metrics = [m for m in metrics if m in df.columns]

    if not available_metrics:
        return pd.DataFrame()

    summary = (
        df.groupby(group_cols)[available_metrics]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    return flatten_columns(summary)


def make_long_summary(df, group_cols, metrics):
    available_metrics = [m for m in metrics if m in df.columns]
    rows = []

    for group_values, group_df in df.groupby(group_cols):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)

        group_data = dict(zip(group_cols, group_values))

        for metric in available_metrics:
            rows.append({
                **group_data,
                "metric": metric,
                "mean": group_df[metric].mean(),
                "std": group_df[metric].std(),
                "count": group_df[metric].count()
            })

    return pd.DataFrame(rows)


def make_pass_rate_summary(df, group_cols):
    required_cols = group_cols + ["pass_common"]

    if any(col not in df.columns for col in required_cols):
        return pd.DataFrame()

    summary = (
        df.groupby(group_cols, dropna=False)["pass_common"]
        .agg(["sum", "count", "mean"])
        .reset_index()
        .rename(columns={
            "sum": "num_pass",
            "count": "total",
            "mean": "pass_rate"
        })
    )

    if not summary.empty:
        summary["num_pass"] = summary["num_pass"].astype(int)
        summary["total"] = summary["total"].astype(int)

    return flatten_columns(summary)


def make_mean_metrics_summary(df, group_cols, metrics):
    if any(col not in df.columns for col in group_cols):
        return pd.DataFrame()

    available_metrics = [m for m in metrics if m in df.columns]

    if not available_metrics:
        return pd.DataFrame()

    summary = (
        df.groupby(group_cols, dropna=False)[available_metrics]
        .mean()
        .reset_index()
    )

    return flatten_columns(summary)


def save_bar_chart(df, x_col, y_cols, title, ylabel, output_path):
    if df.empty:
        return

    plot_df = df[[x_col] + y_cols].copy()
    plot_df = plot_df.set_index(x_col)

    ax = plot_df.plot(kind="bar", figsize=(12, 6))
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="best")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_grouped_pass_rate_chart(df, category_col, output_path, title):
    required_cols = ["config_name", category_col, "pass_rate"]

    if df.empty or any(col not in df.columns for col in required_cols):
        return

    plot_df = df[required_cols].copy()
    plot_df[category_col] = plot_df[category_col].fillna("missing").astype(str)
    plot_df["config_name"] = plot_df["config_name"].fillna("missing").astype(str)

    plot_df = plot_df.pivot(
        index=category_col,
        columns="config_name",
        values="pass_rate"
    ).sort_index()

    ax = plot_df.plot(kind="bar", figsize=(12, 6))
    ax.set_title(title)
    ax.set_ylabel("Pass Rate")
    ax.set_xlabel("")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="best")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_count_chart(df, category_col, output_path, title):
    if df.empty or category_col not in df.columns:
        return

    count_df = (
        df.groupby(category_col, dropna=False)
        .size()
        .reset_index(name="count")
    )

    count_df[category_col] = count_df[category_col].fillna("missing").astype(str)
    count_df = count_df.sort_values(category_col)
    plot_df = count_df.set_index(category_col)

    ax = plot_df.plot(kind="bar", figsize=(10, 5), legend=False)
    ax.set_title(title)
    ax.set_ylabel("Count")
    ax.set_xlabel("")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_latency_chart(df, output_path):
    if df.empty or "latency_seconds_mean" not in df.columns:
        return

    plot_df = df[["config_name", "latency_seconds_mean"]].copy()
    plot_df = plot_df.set_index("config_name")

    ax = plot_df.plot(kind="bar", figsize=(10, 5), legend=False)
    ax.set_title("Average Latency by Configuration")
    ax.set_ylabel("Seconds")
    ax.set_xlabel("")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_pass_rate_chart(df, output_path):
    if df.empty:
        return

    plot_df = df[["config_name", "pass_rate"]].copy()
    plot_df = plot_df.set_index("config_name")

    ax = plot_df.plot(kind="bar", figsize=(10, 5), legend=False)
    ax.set_title("Pass Rate by Configuration")
    ax.set_ylabel("Pass Rate")
    ax.set_xlabel("")
    ax.set_ylim(0, 1.05)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=None)
    args = parser.parse_args()

    input_name = f"all_scores_target{args.target}.xlsx" if args.target is not None else "all_scores.xlsx"
    input_path = EVALUATION_DIR / input_name

    if not input_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {input_path}")

    charts_dir = EVALUATION_DIR / (f"charts_target{args.target}" if args.target is not None else "charts")
    charts_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(input_path)

    print("Loaded:", input_path)
    print("Rows:", len(df))

    if "judge_status" in df.columns:
        df_valid = df[df["judge_status"] == "success"].copy()
    else:
        df_valid = df.copy()

    print("Valid scored rows:", len(df_valid))

    all_metric_cols = COMMON_METRICS + RAG_METRICS + OTHER_METRICS

    for col in all_metric_cols:
        if col in df_valid.columns:
            df_valid[col] = pd.to_numeric(df_valid[col], errors="coerce")

    if "config_name" not in df_valid.columns:
        raise ValueError("Missing column: config_name")

    if "mode" not in df_valid.columns:
        df_valid["mode"] = ""

    # 1. Summary chung theo config
    summary_common = make_summary(
        df_valid,
        ["config_name"],
        COMMON_METRICS + OTHER_METRICS
    )

    # 2. Summary dạng long để dễ vẽ hoặc đưa vào báo cáo
    summary_common_long = make_long_summary(
        df_valid,
        ["config_name"],
        COMMON_METRICS + OTHER_METRICS
    )

    # 3. Summary riêng cho RAG
    df_rag = df_valid[df_valid["mode"] == "rag"].copy()

    summary_rag = make_summary(
        df_rag,
        ["config_name"],
        RAG_METRICS
    )

    summary_rag_long = make_long_summary(
        df_rag,
        ["config_name"],
        RAG_METRICS
    )

    # 4. Pass rate
    df_valid["pass_common"] = (
        (df_valid["answer_correctness"] >= 0.7) &
        (df_valid["answer_relevancy"] >= 0.7) &
        (df_valid["medical_safety"] >= 0.8)
    )

    pass_rate = (
        df_valid.groupby("config_name")["pass_common"]
        .agg(["sum", "count", "mean"])
        .reset_index()
        .rename(columns={
            "sum": "num_pass",
            "count": "total",
            "mean": "pass_rate"
        })
    )

    # 5. Pass rate chi tiết theo question_type / difficulty / expected_behavior
    pass_rate_by_question_type = make_pass_rate_summary(
        df_valid,
        ["config_name", "question_type"]
    )

    pass_rate_by_difficulty = make_pass_rate_summary(
        df_valid,
        ["config_name", "difficulty"]
    )

    pass_rate_by_behavior = make_pass_rate_summary(
        df_valid,
        ["config_name", "expected_behavior"]
    )

    # 6. Mean score theo question_type cho các metric chính
    metrics_by_question_type = make_mean_metrics_summary(
        df_valid,
        ["config_name", "question_type"],
        DETAILED_SCORE_METRICS
    )

    # 7. Breakdown theo question_type
    if "question_type" in df_valid.columns:
        summary_by_question_type = make_summary(
            df_valid,
            ["config_name", "question_type"],
            COMMON_METRICS
        )
    else:
        summary_by_question_type = pd.DataFrame()

    # 8. Breakdown theo difficulty
    if "difficulty" in df_valid.columns:
        summary_by_difficulty = make_summary(
            df_valid,
            ["config_name", "difficulty"],
            COMMON_METRICS
        )
    else:
        summary_by_difficulty = pd.DataFrame()

    # 9. Failed judge rows nếu có
    if "judge_status" in df.columns:
        failed_rows = df[df["judge_status"] != "success"].copy()
    else:
        failed_rows = pd.DataFrame()

    # 10. Biểu đồ common metrics
    common_chart_cols = [
        "answer_correctness_mean",
        "answer_relevancy_mean",
        "answer_completeness_mean",
        "medical_safety_mean"
    ]

    common_chart_cols = [c for c in common_chart_cols if c in summary_common.columns]

    save_bar_chart(
        summary_common,
        x_col="config_name",
        y_cols=common_chart_cols,
        title="Common Metrics by Configuration",
        ylabel="Score",
        output_path=charts_dir / "common_metrics_by_config.png"
    )

    # 11. Biểu đồ RAG metrics
    rag_chart_cols = [
        "faithfulness_mean",
        "context_precision_mean",
        "context_recall_mean"
    ]

    rag_chart_cols = [c for c in rag_chart_cols if c in summary_rag.columns]

    save_bar_chart(
        summary_rag,
        x_col="config_name",
        y_cols=rag_chart_cols,
        title="RAG Metrics by Configuration",
        ylabel="Score",
        output_path=charts_dir / "rag_metrics_by_config.png"
    )

    # 12. Biểu đồ pass rate
    save_pass_rate_chart(
        pass_rate,
        output_path=charts_dir / "pass_rate_by_config.png"
    )

    save_grouped_pass_rate_chart(
        pass_rate_by_question_type,
        category_col="question_type",
        output_path=charts_dir / "pass_rate_by_question_type.png",
        title="Pass Rate by Question Type"
    )

    save_grouped_pass_rate_chart(
        pass_rate_by_difficulty,
        category_col="difficulty",
        output_path=charts_dir / "pass_rate_by_difficulty.png",
        title="Pass Rate by Difficulty"
    )

    save_grouped_pass_rate_chart(
        pass_rate_by_behavior,
        category_col="expected_behavior",
        output_path=charts_dir / "pass_rate_by_behavior.png",
        title="Pass Rate by Expected Behavior"
    )

    save_count_chart(
        df_valid,
        category_col="question_type",
        output_path=charts_dir / "question_type_count.png",
        title="Question Count by Question Type"
    )

    # 13. Biểu đồ latency
    save_latency_chart(
        summary_common,
        output_path=charts_dir / "latency_by_config.png"
    )

    # 14. Xuất Excel
    output_name = f"summary_by_config_target{args.target}.xlsx" if args.target is not None else "summary_by_config.xlsx"
    output_path = EVALUATION_DIR / output_name

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="all_scores_raw", index=False)
        df_valid.to_excel(writer, sheet_name="all_scores_valid", index=False)

        summary_common.to_excel(writer, sheet_name="summary_common", index=False)
        summary_common_long.to_excel(writer, sheet_name="summary_common_long", index=False)

        if not summary_rag.empty:
            summary_rag.to_excel(writer, sheet_name="summary_rag_only", index=False)
            summary_rag_long.to_excel(writer, sheet_name="summary_rag_long", index=False)

        pass_rate.to_excel(writer, sheet_name="pass_rate", index=False)

        if not pass_rate_by_question_type.empty:
            pass_rate_by_question_type.to_excel(writer, sheet_name="pass_rate_by_question_type", index=False)

        if not pass_rate_by_difficulty.empty:
            pass_rate_by_difficulty.to_excel(writer, sheet_name="pass_rate_by_difficulty", index=False)

        if not pass_rate_by_behavior.empty:
            pass_rate_by_behavior.to_excel(writer, sheet_name="pass_rate_by_behavior", index=False)

        if not metrics_by_question_type.empty:
            metrics_by_question_type.to_excel(writer, sheet_name="metrics_by_question_type", index=False)

        if not summary_by_question_type.empty:
            summary_by_question_type.to_excel(writer, sheet_name="by_question_type", index=False)

        if not summary_by_difficulty.empty:
            summary_by_difficulty.to_excel(writer, sheet_name="by_difficulty", index=False)

        if not failed_rows.empty:
            failed_rows.to_excel(writer, sheet_name="failed_judge_rows", index=False)

    print("Saved:", output_path)
    print("Saved charts to:", charts_dir)

    print("\n=== Summary common ===")
    print(summary_common)

    print("\n=== Pass rate ===")
    print(pass_rate)

    if not summary_rag.empty:
        print("\n=== RAG metrics ===")
        print(summary_rag)


if __name__ == "__main__":
    main()
