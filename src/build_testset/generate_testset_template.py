import pandas as pd
from retrieval.config import TESTSET_XLSX_PATH, TESTSET_CSV_PATH


def build_template():
    rows = []

    plan = [
        ("in_scope_easy", 90, "easy", "answer_from_context"),
        ("in_scope_medium", 90, "medium", "answer_from_context"),
        ("in_scope_hard", 60, "hard", "answer_from_context"),
        ("comparison", 30, "medium", "answer_from_context"),
        ("out_of_scope", 20, "medium", "refuse_due_to_missing_context"),
        ("safety", 10, "hard", "safety_warning"),
    ]

    qid = 1

    for question_type, count, difficulty, expected_behavior in plan:
        for _ in range(count):
            rows.append({
                "question_id": f"Q{qid:03d}",
                "disease_group": "",
                "question_type": question_type,
                "difficulty": difficulty,
                "question": "",
                "ground_truth": "",
                "expected_behavior": expected_behavior,
                "expected_source": "",
                "note": ""
            })
            qid += 1

    df = pd.DataFrame(rows)
    df.to_excel(TESTSET_XLSX_PATH, index=False)
    df.to_csv(TESTSET_CSV_PATH, index=False, encoding="utf-8-sig")

    print("Created:", TESTSET_XLSX_PATH)
    print("Created:", TESTSET_CSV_PATH)
    print("Total questions:", len(df))


if __name__ == "__main__":
    build_template()
