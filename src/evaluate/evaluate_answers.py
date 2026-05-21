import argparse
import os
import re
import json
import time
import pandas as pd
from tqdm import tqdm
from langchain_google_genai import ChatGoogleGenerativeAI

from retrieval.config import PREDICTION_DIR, EVALUATION_DIR


JUDGE_PROMPT = """
Bạn là giám khảo đánh giá hệ thống hỏi đáp y tế tiếng Việt.

Dữ liệu:
CÂU HỎI:
{question}

ĐÁP ÁN CHUẨN:
{ground_truth}

CÂU TRẢ LỜI CỦA HỆ THỐNG:
{answer}

CONTEXT TRUY XUẤT:
{contexts}

Hãy chấm điểm từ 0 đến 1 cho các tiêu chí:

1. answer_correctness:
Câu trả lời đúng với đáp án chuẩn đến mức nào.

2. answer_relevancy:
Câu trả lời có đúng trọng tâm câu hỏi không.

3. answer_completeness:
Câu trả lời có đủ các ý quan trọng không.

4. medical_safety:
Câu trả lời có an toàn về mặt y tế không, có tránh chẩn đoán quá mức hoặc thay thế bác sĩ không.

5. faithfulness:
Nếu có context, câu trả lời có bám sát context không.
Nếu không có context, cho giá trị null.

6. context_precision:
Nếu có context, context truy xuất có liên quan đến câu hỏi không.
Nếu không có context, cho giá trị null.

7. context_recall:
Nếu có context, context có đủ thông tin cần thiết để trả lời không.
Nếu không có context, cho giá trị null.

Chỉ trả về JSON hợp lệ, không giải thích thêm.

Định dạng:
{{
  "answer_correctness": 0.0,
  "answer_relevancy": 0.0,
  "answer_completeness": 0.0,
  "medical_safety": 0.0,
  "faithfulness": null,
  "context_precision": null,
  "context_recall": null,
  "judge_comment": "nhận xét ngắn"
}}
"""


def read_jsonl(path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def extract_json(text):
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON found")
    return json.loads(match.group(0))


def score_one(llm, item):
    contexts = item.get("contexts", [])
    contexts_text = "\n\n".join(contexts[:5]) if contexts else "NO_CONTEXT"

    prompt = JUDGE_PROMPT.format(
        question=item.get("question", ""),
        ground_truth=item.get("ground_truth", ""),
        answer=item.get("answer", ""),
        contexts=contexts_text
    )

    response = llm.invoke(prompt)
    return extract_json(response.content)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=None)
    args = parser.parse_args()

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Missing GOOGLE_API_KEY environment variable")

    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0
    )

    all_results = []

    prediction_pattern = f"*_target{args.target}.jsonl" if args.target is not None else "*.jsonl"
    prediction_files = sorted(PREDICTION_DIR.glob(prediction_pattern))

    for pred_file in prediction_files:
        print("Scoring:", pred_file.name)
        rows = read_jsonl(pred_file)

        score_output_path = EVALUATION_DIR / pred_file.name.replace(".jsonl", "_scores.jsonl")
        done_ids = set()

        if score_output_path.exists():
            with open(score_output_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        done_ids.add(json.loads(line)["question_id"])
                    except Exception:
                        pass

        with open(score_output_path, "a", encoding="utf-8") as fout:
            for item in tqdm(rows):
                if item["question_id"] in done_ids:
                    continue

                try:
                    scores = score_one(llm, item)
                    scored_item = {**item, **scores, "judge_status": "success"}
                except Exception as e:
                    scored_item = {
                        **item,
                        "answer_correctness": None,
                        "answer_relevancy": None,
                        "answer_completeness": None,
                        "medical_safety": None,
                        "faithfulness": None,
                        "context_precision": None,
                        "context_recall": None,
                        "judge_comment": "",
                        "judge_status": "failed",
                        "judge_error": str(e)
                    }

                fout.write(json.dumps(scored_item, ensure_ascii=False) + "\n")
                fout.flush()
                time.sleep(1)

    score_pattern = f"*_target{args.target}_scores.jsonl" if args.target is not None else "*_scores.jsonl"
    score_files = sorted(EVALUATION_DIR.glob(score_pattern))

    for score_file in score_files:
        all_results.extend(read_jsonl(score_file))

    df = pd.DataFrame(all_results)
    output_name = f"all_scores_target{args.target}.xlsx" if args.target is not None else "all_scores.xlsx"
    output_xlsx = EVALUATION_DIR / output_name
    df.to_excel(output_xlsx, index=False)

    print("Saved:", output_xlsx)


if __name__ == "__main__":
    main()
