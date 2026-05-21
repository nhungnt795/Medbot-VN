import argparse
import json
import pandas as pd
from tqdm import tqdm

from retrieval.config import TESTSET_DIR, PREDICTION_DIR, EXPERIMENTS
from retrieval.rag_engine import MedicalRAGEngine


def target_testset_path(target):
    return TESTSET_DIR / f"gold_testset_{target}.xlsx"


def load_done_question_ids(output_path):
    done = set()

    if not output_path.exists():
        return done

    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
                done.add(item["question_id"])
            except Exception:
                continue

    return done


def append_jsonl(item, output_path):
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def run_one_experiment(engine, df_test, exp, target):
    output_path = PREDICTION_DIR / f"{exp['config_name']}_target{target}.jsonl"
    done_ids = load_done_question_ids(output_path)

    print("=" * 100)
    print("Running:", exp["config_name"])
    print("Already done:", len(done_ids))
    print("Output:", output_path)

    for _, row in tqdm(df_test.iterrows(), total=len(df_test)):
        question_id = row["question_id"]

        if question_id in done_ids:
            continue

        try:
            result = engine.answer(
                question=row["question"],
                model_name=exp["model_name"],
                mode=exp["mode"],
                k=exp["k"]
            )

            result.update({
                "question_id": question_id,
                "testset_target": target,
                "config_name": exp["config_name"],
                "disease_group": row.get("disease_group", ""),
                "question_type": row.get("question_type", ""),
                "difficulty": row.get("difficulty", ""),
                "ground_truth": row.get("ground_truth", ""),
                "expected_behavior": row.get("expected_behavior", ""),
                "expected_source": row.get("expected_source", "")
            })

        except Exception as e:
            result = {
                "question_id": question_id,
                "testset_target": target,
                "question": row["question"],
                "config_name": exp["config_name"],
                "model_name": exp["model_name"],
                "mode": exp["mode"],
                "k": exp["k"],
                "answer": "",
                "contexts": [],
                "sources": [],
                "latency_seconds": None,
                "status": "failed",
                "error": str(e),
                "disease_group": row.get("disease_group", ""),
                "question_type": row.get("question_type", ""),
                "difficulty": row.get("difficulty", ""),
                "ground_truth": row.get("ground_truth", ""),
                "expected_behavior": row.get("expected_behavior", ""),
                "expected_source": row.get("expected_source", "")
            }

        append_jsonl(result, output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=300)
    parser.add_argument("--testset", default=None)
    args = parser.parse_args()

    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

    testset_path = target_testset_path(args.target) if args.testset is None else args.testset
    df_test = pd.read_excel(testset_path)

    required_cols = ["question_id", "question", "ground_truth"]
    for col in required_cols:
        if col not in df_test.columns:
            raise ValueError(f"Missing column: {col}")

    df_test = df_test.dropna(subset=["question"])
    print("Testset:", testset_path)
    print("Target:", args.target)
    print("Total questions:", len(df_test))

    engine = MedicalRAGEngine()

    for exp in EXPERIMENTS:
        run_one_experiment(engine, df_test, exp, args.target)

    print("All experiments completed")


if __name__ == "__main__":
    main()
