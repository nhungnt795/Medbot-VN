import argparse
import json
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from retrieval.config import (
    TESTSET_DIR,
    VECTOR_DB_DIR,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_MAX_SEQ_LENGTH
)


def target_testset_path(target):
    return TESTSET_DIR / f"gold_testset_{target}.xlsx"


def output_paths(method, k, target):
    prefix = f"retrieval_quality_{method}_k{k}_{target}"
    return (
        TESTSET_DIR / f"{prefix}.csv",
        TESTSET_DIR / f"{prefix}.xlsx",
        TESTSET_DIR / f"{prefix}_summary.json"
    )


def normalize_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def split_expected_sources(value):
    value = normalize_text(value)
    if not value:
        return []

    parts = re.split(r"[;,]", value)
    return [part.strip() for part in parts if part.strip()]


def load_vector_db(device):
    if not VECTOR_DB_DIR.exists():
        raise FileNotFoundError(f"Missing vector DB: {VECTOR_DB_DIR}")

    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )
    embedding_model._client.max_seq_length = EMBEDDING_MAX_SEQ_LENGTH

    return Chroma(
        persist_directory=str(VECTOR_DB_DIR),
        embedding_function=embedding_model
    )


def read_testset(path):
    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    return pd.read_csv(path)


def retrieve_docs(vector_db, question, search_type, k, fetch_k, lambda_mult):
    if search_type == "mmr":
        return vector_db.max_marginal_relevance_search(
            question,
            k=k,
            fetch_k=max(fetch_k, k),
            lambda_mult=lambda_mult
        )

    return vector_db.similarity_search(question, k=k)


def doc_source_keys(doc):
    metadata = doc.metadata or {}
    keys = set()

    for field in ["chunk_id", "source_file"]:
        value = metadata.get(field)
        if value:
            keys.add(str(value).strip())

    return keys


def compute_hits(expected_sources, docs, cutoffs):
    normalized_expected = {source.lower() for source in expected_sources}
    rows = {}

    for cutoff in cutoffs:
        retrieved_keys = set()
        for doc in docs[:cutoff]:
            retrieved_keys.update(key.lower() for key in doc_source_keys(doc))

        matched = sorted(source for source in normalized_expected if source in retrieved_keys)
        rows[f"hit_any_at_{cutoff}"] = int(len(matched) > 0)
        rows[f"hit_all_at_{cutoff}"] = int(normalized_expected.issubset(retrieved_keys))
        rows[f"matched_at_{cutoff}"] = ";".join(matched)

    return rows


def format_sources(docs):
    sources = []

    for rank, doc in enumerate(docs, 1):
        metadata = doc.metadata or {}
        sources.append({
            "rank": rank,
            "chunk_id": metadata.get("chunk_id", ""),
            "disease": metadata.get("disease", ""),
            "section_title": metadata.get("section_title", ""),
            "source_file": metadata.get("source_file", ""),
            "page_start": metadata.get("page_start", ""),
            "page_end": metadata.get("page_end", ""),
            "content_type": metadata.get("content_type", "")
        })

    return sources


def summarize(details_df, cutoffs):
    evaluated = details_df[details_df["retrieval_evaluated"] == 1]
    summary = {
        "total_rows": int(len(details_df)),
        "evaluated_rows": int(len(evaluated)),
        "skipped_rows": int(len(details_df) - len(evaluated))
    }

    for cutoff in cutoffs:
        if len(evaluated) == 0:
            summary[f"hit_any_at_{cutoff}"] = None
            summary[f"hit_all_at_{cutoff}"] = None
        else:
            summary[f"hit_any_at_{cutoff}"] = round(float(evaluated[f"hit_any_at_{cutoff}"].mean()), 4)
            summary[f"hit_all_at_{cutoff}"] = round(float(evaluated[f"hit_all_at_{cutoff}"].mean()), 4)

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=300)
    parser.add_argument("--testset", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--output-xlsx", default=None)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--method", choices=["mmr", "similarity"], default="mmr")
    parser.add_argument("--search-type", choices=["mmr", "similarity"], default=None)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--fetch-k", type=int, default=20)
    parser.add_argument("--lambda-mult", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    method = args.search_type or args.method
    default_output_csv, default_output_xlsx, default_summary_json = output_paths(
        method=method,
        k=args.k,
        target=args.target
    )

    testset_path = Path(args.testset) if args.testset else target_testset_path(args.target)
    if not testset_path.exists():
        raise FileNotFoundError(f"Missing testset: {testset_path}")

    output_csv = Path(args.output_csv) if args.output_csv else default_output_csv
    output_xlsx = Path(args.output_xlsx) if args.output_xlsx else default_output_xlsx
    summary_json = Path(args.summary_json) if args.summary_json else default_summary_json
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_xlsx.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    df = read_testset(testset_path)
    if "question" not in df.columns:
        raise ValueError("Missing required column: question")

    vector_db = load_vector_db(args.device)
    cutoffs = sorted(set([1, 3, args.k]))
    details = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Checking retrieval"):
        question = normalize_text(row.get("question", ""))
        expected_sources = split_expected_sources(row.get("expected_source", ""))
        expected_behavior = normalize_text(row.get("expected_behavior", ""))
        should_evaluate = bool(question and expected_sources and expected_behavior == "answer_from_context")

        detail = row.to_dict()
        detail.update({
            "retrieval_search_type": method,
            "retrieval_k": args.k,
            "retrieval_fetch_k": args.fetch_k,
            "retrieval_lambda_mult": args.lambda_mult,
            "expected_source_count": len(expected_sources),
            "retrieval_evaluated": int(should_evaluate)
        })

        if not should_evaluate:
            for cutoff in cutoffs:
                detail[f"hit_any_at_{cutoff}"] = None
                detail[f"hit_all_at_{cutoff}"] = None
                detail[f"matched_at_{cutoff}"] = ""
            detail["retrieved_sources_json"] = "[]"
            details.append(detail)
            continue

        docs = retrieve_docs(
            vector_db=vector_db,
            question=question,
            search_type=method,
            k=args.k,
            fetch_k=args.fetch_k,
            lambda_mult=args.lambda_mult
        )
        detail.update(compute_hits(expected_sources, docs, cutoffs))
        detail["retrieved_sources_json"] = json.dumps(format_sources(docs), ensure_ascii=False)
        details.append(detail)

    details_df = pd.DataFrame(details)
    summary = summarize(details_df, cutoffs)

    details_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(output_xlsx) as writer:
        details_df.to_excel(writer, index=False, sheet_name="details")
        pd.DataFrame([summary]).to_excel(writer, index=False, sheet_name="summary")

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Done retrieval quality check")
    print("Saved:", output_csv)
    print("Saved:", output_xlsx)
    print("Saved:", summary_json)
    print("Summary:", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
