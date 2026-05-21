import time
import requests
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from retrieval.config import (
    VECTOR_DB_DIR,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_MAX_SEQ_LENGTH,
    OLLAMA_BASE_URL
)
from retrieval.prompts import (
    RAG_SYSTEM_PROMPT,
    RAG_USER_PROMPT,
    NO_RAG_SYSTEM_PROMPT,
    NO_RAG_USER_PROMPT
)


class MedicalRAGEngine:
    def __init__(self):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": True}
        )
        self.embedding_model._client.max_seq_length = EMBEDDING_MAX_SEQ_LENGTH

        self.vector_db = Chroma(
            persist_directory=str(VECTOR_DB_DIR),
            embedding_function=self.embedding_model
        )

    def call_ollama(self, model_name, system_prompt, user_prompt):
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False,
            "options": {
                "temperature": 0.0,
                "top_p": 0.9
            }
        }

        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=600
        )
        response.raise_for_status()
        return response.json()["message"]["content"]

    def format_context(self, docs):
        context_parts = []

        for idx, doc in enumerate(docs, 1):
            metadata = doc.metadata
            source = metadata.get("source_file", "unknown")
            disease = metadata.get("disease", "unknown")
            section_title = metadata.get("section_title", "")
            chunk_id = metadata.get("chunk_id", "unknown")
            page_start = metadata.get("page_start", "unknown")

            context_parts.append(
                f"[Tài liệu {idx}]\n"
                f"chunk_id: {chunk_id}\n"
                f"bệnh/chủ đề: {disease}\n"
                f"mục/tiểu mục: {section_title}\n"
                f"nguồn: {source}\n"
                f"trang: {page_start}\n"
                f"nội dung:\n{doc.page_content}"
            )

        return "\n\n".join(context_parts)

    def get_sources(self, docs):
        sources = []

        for doc in docs:
            metadata = doc.metadata
            sources.append({
                "chunk_id": metadata.get("chunk_id"),
                "disease": metadata.get("disease"),
                "section_title": metadata.get("section_title"),
                "source_file": metadata.get("source_file"),
                "page_start": metadata.get("page_start"),
                "page_end": metadata.get("page_end"),
                "content_type": metadata.get("content_type")
            })

        return sources

    def retrieve(self, question, k=5, fetch_k=20, lambda_mult=0.5):
        return self.vector_db.max_marginal_relevance_search(
            question,
            k=k,
            fetch_k=max(fetch_k, k),
            lambda_mult=lambda_mult
        )

    def answer(self, question, model_name, mode="rag", k=5):
        start_time = time.time()

        if mode == "no_rag":
            user_prompt = NO_RAG_USER_PROMPT.format(question=question)
            answer = self.call_ollama(
                model_name=model_name,
                system_prompt=NO_RAG_SYSTEM_PROMPT,
                user_prompt=user_prompt
            )

            return {
                "question": question,
                "model_name": model_name,
                "mode": mode,
                "k": 0,
                "retrieval_method": "none",
                "answer": answer,
                "contexts": [],
                "sources": [],
                "latency_seconds": round(time.time() - start_time, 3),
                "status": "success"
            }

        docs = self.retrieve(question, k=k)
        context = self.format_context(docs)

        user_prompt = RAG_USER_PROMPT.format(
            context=context,
            question=question
        )

        answer = self.call_ollama(
            model_name=model_name,
            system_prompt=RAG_SYSTEM_PROMPT,
            user_prompt=user_prompt
        )

        return {
            "question": question,
            "model_name": model_name,
            "mode": mode,
            "k": k,
            "retrieval_method": "mmr",
            "answer": answer,
            "contexts": [doc.page_content for doc in docs],
            "sources": self.get_sources(docs),
            "latency_seconds": round(time.time() - start_time, 3),
            "status": "success"
        }
