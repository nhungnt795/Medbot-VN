import argparse
from retrieval.rag_engine import MedicalRAGEngine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llama3.1:8b")
    parser.add_argument("--mode", default="rag", choices=["rag", "no_rag"])
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()

    engine = MedicalRAGEngine()

    print("Medical RAG Chatbot Terminal")
    print("Model:", args.model)
    print("Mode:", args.mode)
    print("k:", args.k)
    print("Gõ 'exit' để thoát.")
    print("=" * 80)

    while True:
        question = input("\nCâu hỏi: ").strip()

        if question.lower() in ["exit", "quit", "q"]:
            break

        if not question:
            continue

        result = engine.answer(
            question=question,
            model_name=args.model,
            mode=args.mode,
            k=args.k
        )

        print("\nTrả lời:")
        print(result["answer"])

        if result["sources"]:
            print("\nNguồn truy xuất:")
            for i, src in enumerate(result["sources"], 1):
                print(
                    f"{i}. {src.get('disease')} | "
                    f"{src.get('source_file')} | "
                    f"trang {src.get('page_start')} | "
                    f"{src.get('chunk_id')}"
                )

        print("\nThời gian:", result["latency_seconds"], "giây")


if __name__ == "__main__":
    main()
