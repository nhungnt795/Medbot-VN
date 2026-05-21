import os
from pathlib import Path

PROJECT_DIR = Path(os.getenv("MEDICAL_RAG_PROJECT_DIR", Path(__file__).resolve().parents[1])).resolve()

DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
VECTOR_DB_DIR = DATA_DIR / "vector_db" / "vihealthbert_chroma"
TESTSET_DIR = DATA_DIR / "testset"

OUTPUT_DIR = PROJECT_DIR / "outputs"
PREDICTION_DIR = OUTPUT_DIR / "predictions"
EVALUATION_DIR = OUTPUT_DIR / "evaluation"
LOG_DIR = OUTPUT_DIR / "logs"

CHUNKS_PKL_PATH = PROCESSED_DIR / "chunks_gold.pkl"
CHUNKS_JSONL_PATH = PROCESSED_DIR / "chunks_gold.jsonl"
PREPROCESSING_REPORT_PATH = PROCESSED_DIR / "preprocessing_report.xlsx"

TESTSET_XLSX_PATH = TESTSET_DIR / "gold_testset_300.xlsx"
TESTSET_CSV_PATH = TESTSET_DIR / "gold_testset_300.csv"

EMBEDDING_MODEL_NAME = "demdecuong/vihealthbert-base-word"
EMBEDDING_MAX_SEQ_LENGTH = 256

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

OLLAMA_BASE_URL = "http://localhost:11434"

EXPERIMENTS = [
    {
        "config_name": "B1_Llama_no_rag",
        "model_name": "llama3.1:8b",
        "mode": "no_rag",
        "k": 0
    },
    {
        "config_name": "B2_Llama_rag_mmr_k5",
        "model_name": "llama3.1:8b",
        "mode": "rag",
        "k": 5
    },
    {
        "config_name": "B1_Qwen_no_rag",
        "model_name": "qwen2.5:7b",
        "mode": "no_rag",
        "k": 0
    },
    {
        "config_name": "B2_Qwen_rag_mmr_k5",
        "model_name": "qwen2.5:7b",
        "mode": "rag",
        "k": 5
    }
]
