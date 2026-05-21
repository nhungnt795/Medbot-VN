import shutil
import pickle
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from retrieval.config import (
    CHUNKS_PKL_PATH,
    VECTOR_DB_DIR,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_MAX_SEQ_LENGTH
)


def main():
    with open(CHUNKS_PKL_PATH, "rb") as f:
        docs = pickle.load(f)

    print("Loaded chunks:", len(docs))
    VECTOR_DB_DIR.parent.mkdir(parents=True, exist_ok=True)

    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True}
    )

    embedding_model._client.max_seq_length = EMBEDDING_MAX_SEQ_LENGTH

    if VECTOR_DB_DIR.exists():
        shutil.rmtree(VECTOR_DB_DIR)

    vector_db = Chroma.from_documents(
        documents=docs,
        embedding=embedding_model,
        persist_directory=str(VECTOR_DB_DIR)
    )

    print("Vector DB created at:", VECTOR_DB_DIR)

    test_query = "Triệu chứng của sốt xuất huyết Dengue là gì?"
    results = vector_db.max_marginal_relevance_search(
        test_query,
        k=5,
        fetch_k=20,
        lambda_mult=0.5
    )

    print("\nTest retrieval with MMR k=5:")
    for i, doc in enumerate(results, 1):
        print("=" * 80)
        print("Rank:", i)
        print("Disease:", doc.metadata.get("disease"))
        print("Section:", doc.metadata.get("section_title"))
        print("Source:", doc.metadata.get("source_file"))
        print(doc.page_content[:500])


if __name__ == "__main__":
    main()
