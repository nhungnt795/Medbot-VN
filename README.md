# Medical Chatbot Vietnam (MedBot VN)

Một hệ thống hỏi-đáp y tế sử dụng **Retrieval-Augmented Generation (RAG)** để cung cấp câu trả lời chính xác dựa trên tài liệu y tế tiếng Việt.

##Mục Tiêu

Xây dựng và đánh giá hệ thống RAG cho các mô hình ngôn ngữ lớn để:
- Trả lời các câu hỏi y tế bằng tiếng Việt
- Cải thiện độ chính xác bằng cách lấy tài liệu liên quan
- So sánh hiệu năng giữa các mô hình với/không sử dụng RAG

## Kiến Trúc Dự Án

```
medbotvn-nckh/
├── data/                      # Dữ liệu
│   ├── raw/                   # Tài liệu y tế gốc (PDF)
│   ├── processed/             # Dữ liệu đã xử lý (chunks, embeddings)
│   ├── testset/               # Bộ câu hỏi kiểm thử
│   └── vector_db/             # Vector database (ChromaDB)
├── src/                       # Source code
│   ├── build_data/            # Xử lý và chuẩn bị dữ liệu
│   ├── build_testset/         # Tạo bộ câu hỏi kiểm thử
│   ├── retrieval/             # Engine RAG chính
│   └── evaluate/              # Đánh giá kết quả
└── outputs/                   # Kết quả đầu ra
    ├── predictions/           # Câu trả lời sinh ra
    └── evaluation/            # Điểm đánh giá
```

## Công Nghệ Sử Dụng

### Stack Chính
- **LangChain**: Orchestration LLM
- **ChromaDB**: Vector database
- **HuggingFace Transformers**: Embedding models (ViHealthBERT)
- **Ollama**: Chạy LLM local (Llama 3.1, Qwen 2.5)
- **Python 3.9+**: Ngôn ngữ chính

### Models
- **Embeddings**: `demdecuong/vihealthbert-base-word` (tiếng Việt, chuyên về y tế)
- **LLMs**:
  - Llama 3.1 8B
  - Qwen 2.5 7B

### Thư Viện Khác
- `pandas`, `numpy`: Xử lý dữ liệu
- `pdfplumber`, `PyMuPDF`, `pdf2image`: Xử lý PDF
- `pytesseract`: OCR cho hình ảnh trong PDF
- `datasets`, `ragas`: Evaluation metrics

## Cài Đặt

### Yêu Cầu Tiên Quyết
- Python 3.9 hoặc cao hơn
- [Ollama](https://ollama.ai/) đã được cài đặt và chạy (mặc định port 11434)
- CUDA (nếu muốn sử dụng GPU cho embeddings)

### Bước 1: Clone và Setup
```bash
# Clone repository
git clone <repo-url>
cd medbotvn-nckh

# Tạo virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate  # Windows

# Cài đặt dependencies
pip install -r requirements.txt
```

### Bước 2: Cấu Hình Ollama
```bash
# Chạy Ollama server
ollama serve

# Ở terminal khác, pull models cần thiết
ollama pull llama3.1:8b
ollama pull qwen2.5:7b
```

### Bước 3: Thiết Lập Biến Môi Trường
```bash
# Tạo file .env (nếu cần)
export MEDICAL_RAG_PROJECT_DIR="<đường-dẫn-project>"
```

## Cách Sử Dụng

### 1. Xử Lý Dữ Liệu

#### Tiền xử lý tài liệu
```bash
cd src/build_data
python preprocess.py
# -> Sinh ra chunks_gold.jsonl trong data/processed/
```

#### Xây dựng Vector Database
```bash
cd src/build_data
python build_vector_db.py
# -> Tạo ChromaDB trong data/vector_db/vihealthbert_chroma/
```

### 2. Tạo Bộ Câu Hỏi Kiểm Thử

#### Từ template
```bash
cd src/build_testset
python generate_testset_template.py
```

#### Tự động từ dữ liệu
```bash
cd src/build_testset
python generate_testset_auto.py
```

#### Lọc theo chất lượng retrieval
```bash
cd src/build_data
python filter_testset_by_retrieval.py
```

### 3. Chạy Experiments

#### Chạy tất cả experiments
```bash
cd src/retrieval
python run_experiments.py --target 400
# --target: số lượng câu hỏi trong bộ test (mặc định: 400)
```

#### Chat interactive
```bash
cd src/retrieval
python run_chat_terminal.py
# Nhập câu hỏi y tế để nhận câu trả lời
```

### 4. Đánh Giá Kết Quả

#### Kiểm tra chất lượng retrieval
```bash
cd src/evaluate
python check_retrieval_quality.py
```

#### Đánh giá câu trả lời
```bash
cd src/evaluate
python evaluate_answers.py
```

#### Phân tích kết quả
```bash
cd src/evaluate
python analyze_results.py
```

#### Làm sạch output
```bash
cd src/evaluate
python clean_generated_outputs.py
```

## Cấu Trúc Experiments

Dự án chạy các experiments sau:

| Config | Model | Mode | Description |
|--------|-------|------|-------------|
| B1_Llama_no_rag | Llama 3.1 8B | No RAG | Llama trả lời trực tiếp |
| B2_Llama_rag_mmr_k5 | Llama 3.1 8B | RAG (k=5) | Llama + lấy 5 chunks liên quan |
| B1_Qwen_no_rag | Qwen 2.5 7B | No RAG | Qwen trả lời trực tiếp |
| B2_Qwen_rag_mmr_k5 | Qwen 2.5 7B | RAG (k=5) | Qwen + lấy 5 chunks liên quan |

## Cấu Trúc Dữ Liệu

### Input
- **Raw**: Tài liệu y tế (PDF) trong `data/raw/`
- **Testset**: Bộ câu hỏi trong `data/testset/` (CSV/XLSX)

### Processing
- **Chunks**: Dữ liệu được chia thành chunks trong `data/processed/chunks_gold.jsonl`
- **Vector DB**: ChromaDB lưu embeddings trong `data/vector_db/vihealthbert_chroma/`

### Output
- **Predictions**: Câu trả lời sinh ra (JSONL) trong `outputs/predictions/`
- **Evaluation**: Điểm số (JSONL, CSV) trong `outputs/evaluation/`

## File Quan Trọng

### Core
- [`src/retrieval/rag_engine.py`](src/retrieval/rag_engine.py) - Engine RAG chính
- [`src/retrieval/config.py`](src/retrieval/config.py) - Cấu hình toàn project
- [`src/retrieval/prompts.py`](src/retrieval/prompts.py) - Prompts cho RAG/No-RAG

### Data Processing
- [`src/build_data/preprocess.py`](src/build_data/preprocess.py) - Tiền xử lý tài liệu
- [`src/build_data/build_vector_db.py`](src/build_data/build_vector_db.py) - Xây dựng vector DB

### Experiments & Evaluation
- [`src/retrieval/run_experiments.py`](src/retrieval/run_experiments.py) - Chạy experiments
- [`src/evaluate/evaluate_answers.py`](src/evaluate/evaluate_answers.py) - Đánh giá answers

## Outputs

### Predictions Format (JSONL)
```json
{
  "question_id": "Q001",
  "question": "Bệnh sốt xuất huyết có triệu chứng gì?",
  "answer": "...",
  "retrieved_chunks": [...],
  "model": "llama3.1:8b",
  "mode": "rag"
}
```

### Evaluation Format (JSONL)
```json
{
  "question_id": "Q001",
  "config_name": "B2_Llama_rag_mmr_k5",
  "scores": {
    "rouge": 0.75,
    "bleu": 0.68,
    "faithfulness": 0.85
  }
}
```

## Medical Signals

Dự án tập trung vào 30+ tín hiệu y tế tiếng Việt:
- Bệnh: `bệnh`, `triệu chứng`, `lâm sàng`
- Chẩn đoán: `chẩn đoán`, `xét nghiệm`
- Điều trị: `điều trị`, `thuốc`, `phác đồ`
- Phòng chống: `phòng bệnh`, `vắc xin`, `cách ly`
- v.v.

## Troubleshooting

### Ollama không kết nối được
```bash
# Kiểm tra Ollama server chạy trên localhost:11434
curl http://localhost:11434/api/tags
```

### CUDA out of memory
```bash
# Giảm batch size hoặc sử dụng CPU
# Chỉnh trong src/retrieval/rag_engine.py
device = "cpu"  # Thay "cuda" bằng "cpu"
```

### Embeddings model không download
```bash
# Manual download
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('demdecuong/vihealthbert-base-word')"
```

## Kết Quả Thực Tế

Dự án đã chạy 4 experiments trên 400 câu hỏi y tế và cho kết quả như sau:

### Bảng So Sánh Kết Quả

| Config | Model | Mode | Correctness | Relevancy | Completeness | Medical Safety |
|--------|-------|------|-------------|-----------|--------------|-----------------|
| B1_Llama_no_rag | Llama 3.1 8B | No RAG | 0.48 | 0.71 | 0.54 | 0.88 |
| B2_Llama_rag_mmr_k5 | Llama 3.1 8B | RAG (k=5) | 0.85 | 0.93 | 0.85 | 0.95 |
| B1_Qwen_no_rag | Qwen 2.5 7B | No RAG | 0.54 | 0.80 | 0.63 | 0.90 |
| B2_Qwen_rag_mmr_k5 | Qwen 2.5 7B | RAG (k=5) | 0.88 | 0.94 | 0.88 | 0.98 |

### Nhận Xét Kết Quả

1. Tác động của RAG:
   - Llama: Correctness tăng 47% (0.48 -> 0.85), Relevancy tăng 31% (0.71 -> 0.93)
   - Qwen: Correctness tăng 63% (0.54 -> 0.88), Relevancy tăng 18% (0.80 -> 0.94)

2. So sánh Models:
   - Qwen vượt trội hơn Llama trong cả hai kịch bản (No-RAG và RAG)
   - Qwen + RAG là tổ hợp tối ưu nhất, đạt Correctness 0.88 và Medical Safety 0.98

3. Độ An Toàn Y Tế:
   - Tất cả configs đều có Medical Safety cao (> 0.88)
   - Qwen + RAG đạt điểm an toàn y tế cao nhất (0.98), đảm bảo câu trả lời không có thông tin sai lệch nguy hiểm

4. Kết Luận:
   - RAG cải thiện chất lượng câu trả lời từ 40-63% tùy theo model
   - Kết hợp Qwen 2.5 với RAG (MMR k=5) cho hiệu suất tốt nhất
   - Hệ thống có thể trả lời câu hỏi y tế với độ chính xác và an toàn cao

## Ghi Chú

- Tất cả câu hỏi và tài liệu sử dụng **tiếng Việt**
- Embedding model được **tinh chỉnh cho miền y tế**
- Sử dụng **local LLMs** (Ollama) thay vì API cloud
- Có thể mở rộng với các models khác

## Phát Triển

### Thêm Model Mới
1. Pull model trong Ollama: `ollama pull <model-name>`
2. Thêm vào `EXPERIMENTS` trong [`src/retrieval/config.py`](src/retrieval/config.py)

### Thêm Evaluation Metrics
Chỉnh sửa [`src/evaluate/evaluate_answers.py`](src/evaluate/evaluate_answers.py)

### Tùy Chỉnh Prompts
Chỉnh sửa [`src/retrieval/prompts.py`](src/retrieval/prompts.py)

