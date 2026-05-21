import os
import re
import json
import pickle
import pdfplumber
import pandas as pd
import pytesseract
import fitz
from pdf2image import convert_from_path
from tqdm import tqdm
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from retrieval.config import (
    RAW_DIR,
    CHUNKS_PKL_PATH,
    CHUNKS_JSONL_PATH,
    PREPROCESSING_REPORT_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP
)


MEDICAL_SIGNALS = [
    "bệnh", "triệu chứng", "lâm sàng", "chẩn đoán", "điều trị",
    "xét nghiệm", "dịch tễ", "phòng bệnh", "dự phòng", "virus",
    "vi rút", "người bệnh", "bệnh nhân", "nhiễm", "sốt", "viêm",
    "dịch", "biến chứng", "đường lây", "thuốc", "vắc xin", "vaccine",
    "tác nhân", "đường truyền", "phác đồ", "cách ly", "kháng sinh",
    "ổ chứa", "thời kỳ ủ bệnh", "miễn dịch", "dịch tiết", "đường hô hấp"
]

ADMIN_SIGNALS = [
    "cộng hòa xã hội chủ nghĩa việt nam",
    "độc lập - tự do - hạnh phúc",
    "quyết định",
    "căn cứ",
    "điều 1",
    "điều 2",
    "điều 3",
    "ban biên soạn",
    "chủ biên",
    "tham gia biên soạn",
    "nhà xuất bản",
    "mục lục"
]

DISEASE_KEYWORDS = [
    "SỐT XUẤT HUYẾT",
    "DENGUE",
    "ZIKA",
    "EBOLA",
    "VIÊM GAN",
    "CÚM",
    "H5N1",
    "MERS",
    "COVID",
    "CORONA",
    "SARS",
    "BẠCH HẦU",
    "HO GÀ",
    "SỞI",
    "RUBELLA",
    "DẠI",
    "TAY CHÂN MIỆNG",
    "LIÊN CẦU",
    "VIÊM MÀNG NÃO",
    "VIÊM NÃO",
    "LAO",
    "TẢ",
    "THƯƠNG HÀN",
    "SỐT RÉT",
    "SỐT VÀNG",
    "HIV",
    "AIDS"
]

INVALID_DISEASE_PHRASES = [
    "BỆNH VIỆN",
    "BỆNH NHÂN",
    "NGƯỜI BỆNH",
    "BỆNH ÁN",
    "BỆNH PHẨM",
    "BỆNH LÝ",
    "MẮC BỆNH",
    "CA BỆNH",
    "Ổ BỆNH",
    "PHÒNG BỆNH",
    "CHỮA BỆNH",
    "DỰ PHÒNG BỆNH"
]

SECTION_PREFIXES = [
    "ĐẠI CƯƠNG",
    "TÁC NHÂN",
    "NGUYÊN NHÂN",
    "DỊCH TỄ",
    "NGUỒN BỆNH",
    "ĐƯỜNG LÂY",
    "LÂM SÀNG",
    "TRIỆU CHỨNG",
    "CHẨN ĐOÁN",
    "XÉT NGHIỆM",
    "ĐIỀU TRỊ",
    "PHÒNG BỆNH",
    "DỰ PHÒNG",
    "BIẾN CHỨNG",
    "TIÊN LƯỢNG",
    "GIÁM SÁT",
    "XỬ LÝ",
    "CÁCH LY",
    "KHUYẾN CÁO",
    "PHÂN LOẠI",
    "QUẢN LÝ",
    "KIỂM SOÁT"
]

GENERIC_SOURCE_MARKERS = [
    "TRUYEN NHIEM",
    "TRUYỀN NHIỄM",
    "GIÁO TRÌNH",
    "SÁCH",
    "TÀI LIỆU"
]


def normalize_space(text: str) -> str:
    text = str(text).replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()


def clean_text(text: str) -> str:
    if not text:
        return ""

    text = str(text)
    text = text.replace("\x00", " ")
    text = re.sub(r"---\s*PAGE\s*\d+\s*---", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"(?m)^\s*\d+\s*$", "", text)
    text = re.sub(r"-\n", "", text)
    text = re.sub(r"([a-zà-ỹ])\n([a-zà-ỹ])", r"\1 \2", text)
    text = normalize_space(text)
    return text


def normalize_title(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text))
    text = text.strip(" :-–—\n\t")
    return text.upper()


def strip_heading_number(text: str) -> str:
    text = re.sub(r"^\s*(?:CHƯƠNG|BÀI|PHẦN)\s+([IVXLCDM]+|\d+)\s*[:.\-–—]*\s*", "", str(text), flags=re.IGNORECASE)
    text = re.sub(r"^\s*([IVXLCDM]+|\d+(?:\.\d+)*)(?:[).]|[:.\-–—])?\s+", "", text, flags=re.IGNORECASE)
    return text.strip()


def clean_disease_title(title: str) -> str:
    title = normalize_title(title)

    if title.startswith("DỊCH "):
        title = title[len("DỊCH "):]
    if title.startswith("BỆNH ") and not title.startswith("BỆNH DO "):
        title = title[len("BỆNH "):]

    return title


def guess_disease_from_filename(filename: str) -> str:
    stem = os.path.splitext(filename)[0]
    stem = re.sub(r"[_\-]+", " ", stem)
    stem = re.sub(r"\s+", " ", stem)
    return clean_disease_title(stem)


def is_generic_source_file(filename: str) -> bool:
    title = normalize_title(os.path.splitext(filename)[0])
    return any(marker in title for marker in GENERIC_SOURCE_MARKERS)


def has_invalid_disease_phrase(title: str) -> bool:
    return any(phrase in title for phrase in INVALID_DISEASE_PHRASES)


def is_heading_like(title: str, max_words: int = 14) -> bool:
    if len(title) < 3 or len(title) > 120:
        return False
    if re.search(r"[;!?]", title):
        return False

    words = re.findall(r"[A-ZÀ-Ỹ0-9]+", title)
    if len(words) == 0 or len(words) > max_words:
        return False

    return True


def detect_disease_line(line: str):
    title = normalize_title(strip_heading_number(line))

    if not is_heading_like(title):
        return None

    lower = title.lower()

    if any(x in lower for x in [
        "cộng hòa", "độc lập", "quyết định", "căn cứ", "điều ",
        "ban biên soạn", "chủ biên", "mục lục", "nhà xuất bản"
    ]):
        return None

    if has_invalid_disease_phrase(title):
        return None

    if any(title.startswith(prefix) for prefix in SECTION_PREFIXES):
        return None

    if title.startswith("BỆNH "):
        return clean_disease_title(title)

    for keyword in DISEASE_KEYWORDS:
        if title == keyword:
            return clean_disease_title(title)
        if title.startswith(keyword + " "):
            return clean_disease_title(title)
        if title.startswith("BỆNH " + keyword):
            return clean_disease_title(title)

    return None


def detect_section_line(line: str):
    title = normalize_title(strip_heading_number(line))

    if not is_heading_like(title):
        return None
    if detect_disease_line(title):
        return None

    for prefix in SECTION_PREFIXES:
        if title == prefix or title.startswith(prefix + " "):
            return title

    return None


def is_admin_heavy(text: str) -> bool:
    lower = text.lower()
    admin_count = sum(1 for s in ADMIN_SIGNALS if s in lower)
    medical_count = sum(1 for s in MEDICAL_SIGNALS if s in lower)

    if admin_count >= 3 and medical_count <= 1:
        return True

    if "mục lục" in lower and len(text) < 1200:
        return True

    return False


def is_useful_chunk(text: str) -> bool:
    text = clean_text(text)

    if len(text) < 150:
        return False

    lower = text.lower()
    medical_count = sum(1 for s in MEDICAL_SIGNALS if s in lower)

    if medical_count == 0 and len(text) < 500:
        return False

    if is_admin_heavy(text) and len(text) < 1000:
        return False

    return True


def extract_with_pdfplumber(pdf_path):
    pages = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            text = page.extract_text(x_tolerance=1.5, y_tolerance=3) or ""
            text = clean_text(text)

            pages.append({
                "page": page_idx + 1,
                "text": text,
                "method": "pdfplumber",
                "char_count": len(text)
            })

    return pages


def extract_with_pymupdf(pdf_path):
    pages = []

    doc = fitz.open(pdf_path)

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        text = page.get_text("text") or ""
        text = clean_text(text)

        pages.append({
            "page": page_idx + 1,
            "text": text,
            "method": "pymupdf",
            "char_count": len(text)
        })

    doc.close()
    return pages


def get_num_pages(pdf_path):
    doc = fitz.open(pdf_path)
    n_pages = len(doc)
    doc.close()
    return n_pages


def extract_with_ocr(pdf_path, dpi=220):
    pages = []
    n_pages = get_num_pages(pdf_path)

    for page_num in tqdm(range(1, n_pages + 1), desc=f"OCR {os.path.basename(pdf_path)}", leave=False):
        images = convert_from_path(
            pdf_path,
            dpi=dpi,
            first_page=page_num,
            last_page=page_num
        )

        if not images:
            text = ""
        else:
            text = pytesseract.image_to_string(images[0], lang="vie+eng")

        text = clean_text(text)

        pages.append({
            "page": page_num,
            "text": text,
            "method": "ocr",
            "char_count": len(text)
        })

    return pages


def total_chars(pages):
    return sum(len(p.get("text", "")) for p in pages)


def extract_text_by_page(pdf_path):
    plumber_pages = extract_with_pdfplumber(pdf_path)
    plumber_chars = total_chars(plumber_pages)

    if plumber_chars >= 5000:
        return plumber_pages, "pdfplumber"

    fitz_pages = extract_with_pymupdf(pdf_path)
    fitz_chars = total_chars(fitz_pages)

    if fitz_chars >= 5000 and fitz_chars > plumber_chars:
        return fitz_pages, "pymupdf"

    if max(plumber_chars, fitz_chars) < 3000:
        ocr_pages = extract_with_ocr(pdf_path)
        ocr_chars = total_chars(ocr_pages)

        if ocr_chars > max(plumber_chars, fitz_chars):
            return ocr_pages, "ocr"

    if fitz_chars > plumber_chars:
        return fitz_pages, "pymupdf"

    return plumber_pages, "pdfplumber"


def rows_to_markdown(rows):
    cleaned_rows = []

    for row in rows:
        cleaned_row = []
        for cell in row:
            cell = "" if cell is None else re.sub(r"\s+", " ", str(cell)).strip()
            cleaned_row.append(cell)
        cleaned_rows.append(cleaned_row)

    cleaned_rows = [r for r in cleaned_rows if any(c.strip() for c in r)]

    if len(cleaned_rows) < 2:
        return ""

    max_cols = max(len(r) for r in cleaned_rows)
    normalized_rows = []

    for row in cleaned_rows:
        row = row + [""] * (max_cols - len(row))
        normalized_rows.append(row)

    header = normalized_rows[0]
    fixed_header = []
    used = set()

    for idx, cell in enumerate(header):
        name = cell.strip() if cell.strip() else f"Cột {idx + 1}"
        if name in used:
            name = f"{name}_{idx + 1}"
        used.add(name)
        fixed_header.append(name)

    body = normalized_rows[1:]

    lines = []
    lines.append("| " + " | ".join(fixed_header) + " |")
    lines.append("| " + " | ".join(["---"] * len(fixed_header)) + " |")

    for row in body:
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def build_page_metadata_map(pages, source_file):
    default_disease = guess_disease_from_filename(source_file)
    current_disease = default_disease
    current_section_title = ""
    generic_source = is_generic_source_file(source_file)
    page_metadata_map = {}

    for page in pages:
        text = page.get("text", "")
        lines = [l.strip() for l in text.split("\n") if l.strip()]

        for line in lines[:40]:
            detected_disease = detect_disease_line(line)
            if detected_disease:
                if generic_source or detected_disease == default_disease or default_disease in detected_disease:
                    current_disease = detected_disease
                current_section_title = ""
                continue

            detected_section = detect_section_line(line)
            if detected_section:
                current_section_title = detected_section

        page_metadata_map[page["page"]] = {
            "disease": current_disease,
            "section_title": current_section_title
        }

    return page_metadata_map


def extract_tables(pdf_path, source_file, page_metadata_map):
    table_docs = []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_idx, page in enumerate(pdf.pages):
                page_num = page_idx + 1
                tables = page.extract_tables() or []

                for table_idx, table in enumerate(tables):
                    if not table or len(table) < 2:
                        continue

                    table_markdown = rows_to_markdown(table)

                    if len(table_markdown) < 150:
                        continue

                    lower_table = table_markdown.lower()

                    skip_keywords = [
                        "họ và tên",
                        "chức danh",
                        "đơn vị công tác",
                        "ban biên soạn",
                        "chủ biên"
                    ]

                    if any(k in lower_table for k in skip_keywords):
                        continue

                    if not any(s in lower_table for s in MEDICAL_SIGNALS) and len(table_markdown) < 600:
                        continue

                    page_meta = page_metadata_map.get(page_num, {
                        "disease": guess_disease_from_filename(source_file),
                        "section_title": ""
                    })
                    disease = page_meta.get("disease", guess_disease_from_filename(source_file))
                    section_title = page_meta.get("section_title", "")
                    chunk_id = f"{os.path.splitext(source_file)[0]}_table_p{page_num}_{table_idx + 1}"

                    content = "BẢNG DỮ LIỆU Y TẾ:\n" + table_markdown

                    table_docs.append(Document(
                        page_content=content,
                        metadata={
                            "chunk_id": chunk_id,
                            "disease": disease,
                            "section_title": section_title,
                            "source_file": source_file,
                            "page_start": page_num,
                            "page_end": page_num,
                            "content_type": "table",
                            "char_length": len(content),
                            "extraction_method": "pdfplumber_table"
                        }
                    ))
    except Exception:
        pass

    return table_docs


def process_pdf(pdf_path):
    source_file = os.path.basename(pdf_path)
    pages, extraction_method = extract_text_by_page(pdf_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "; ", ", ", " "]
    )

    page_metadata_map = build_page_metadata_map(pages, source_file)

    docs = []
    chunk_counter = 0
    pages_with_text = 0
    total_text_chars = 0

    for page in pages:
        page_num = page["page"]
        text = clean_text(page.get("text", ""))

        if text:
            pages_with_text += 1
            total_text_chars += len(text)

        if len(text) < 100:
            continue

        chunks = splitter.split_text(text)

        for chunk in chunks:
            chunk = clean_text(chunk)

            if not is_useful_chunk(chunk):
                continue

            chunk_counter += 1
            chunk_id = f"{os.path.splitext(source_file)[0]}_text_{chunk_counter:04d}"
            page_meta = page_metadata_map.get(page_num, {
                "disease": guess_disease_from_filename(source_file),
                "section_title": ""
            })

            docs.append(Document(
                page_content=chunk,
                metadata={
                    "chunk_id": chunk_id,
                    "disease": page_meta.get("disease", guess_disease_from_filename(source_file)),
                    "section_title": page_meta.get("section_title", ""),
                    "source_file": source_file,
                    "page_start": page_num,
                    "page_end": page_num,
                    "content_type": "text",
                    "char_length": len(chunk),
                    "extraction_method": page.get("method", extraction_method)
                }
            ))

    table_docs = extract_tables(pdf_path, source_file, page_metadata_map)
    docs.extend(table_docs)

    stats = {
        "source_file": source_file,
        "num_pages": len(pages),
        "pages_with_text": pages_with_text,
        "total_text_chars": total_text_chars,
        "num_text_chunks": chunk_counter,
        "num_table_chunks": len(table_docs),
        "num_chunks": len(docs),
        "num_diseases": len(set(doc.metadata.get("disease", "") for doc in docs if doc.metadata.get("disease"))),
        "num_sections": len(set(doc.metadata.get("section_title", "") for doc in docs if doc.metadata.get("section_title"))),
        "extraction_method": extraction_method,
        "status": "success"
    }

    return docs, stats


def save_jsonl(docs, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for doc in docs:
            item = {
                "page_content": doc.page_content,
                "metadata": doc.metadata
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    CHUNKS_PKL_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHUNKS_JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREPROCESSING_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted([f for f in os.listdir(RAW_DIR) if f.lower().endswith(".pdf")])

    all_docs = []
    report_rows = []

    for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
        pdf_path = RAW_DIR / pdf_file

        try:
            docs, stats = process_pdf(str(pdf_path))
            all_docs.extend(docs)
            report_rows.append(stats)
        except Exception as e:
            report_rows.append({
                "source_file": pdf_file,
                "num_pages": None,
                "pages_with_text": None,
                "total_text_chars": None,
                "num_text_chunks": 0,
                "num_table_chunks": 0,
                "num_chunks": 0,
                "extraction_method": "failed",
                "status": f"error: {str(e)}"
            })

    with open(CHUNKS_PKL_PATH, "wb") as f:
        pickle.dump(all_docs, f)

    save_jsonl(all_docs, CHUNKS_JSONL_PATH)

    df_report = pd.DataFrame(report_rows)
    df_report.to_excel(PREPROCESSING_REPORT_PATH, index=False)

    print("Done preprocessing")
    print("PDF files:", len(pdf_files))
    print("Total chunks:", len(all_docs))
    print("Saved:", CHUNKS_PKL_PATH)
    print("Saved:", CHUNKS_JSONL_PATH)
    print("Saved:", PREPROCESSING_REPORT_PATH)

    if len(all_docs) < 800:
        print("WARNING: Total chunks hơi thấp. Hãy kiểm tra các file extraction_method = ocr/failed.")
    else:
        print("Chunk count looks good.")


if __name__ == "__main__":
    main()
