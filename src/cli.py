import argparse
import uuid
from pathlib import Path
from typing import Iterable, List, Tuple

from src.chroma_store import ChromaVectorStore
from src.embeddings import QwenEmbeddingClient
from src.llm import QwenLocalGenerator
from src.settings import Settings


def load_text_files(root: Path) -> List[Tuple[str, str]]:
    files: List[Tuple[str, str]] = []
    for path in root.rglob("*.txt"):
        files.append((str(path), path.read_text(encoding="utf-8")))
    return files


def chunk_text(text: str, chunk_size: int = 480, chunk_overlap: int = 80) -> List[str]:
    """Simple word-based chunking to keep context manageable for embeddings."""
    words = text.split()
    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = min(len(words), start + chunk_size)
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start += max(1, chunk_size - chunk_overlap)
    return chunks


def build_database(
    settings: Settings,
    docs_dir: Path,
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    embedder = QwenEmbeddingClient(settings.embed_model_path, device=settings.device)
    store = ChromaVectorStore(settings.chroma_dir, settings.chroma_collection)

    text_files = load_text_files(docs_dir)
    documents: List[str] = []
    metadatas: List[dict] = []
    ids: List[str] = []

    for file_path, content in text_files:
        for chunk in chunk_text(content, chunk_size=chunk_size, chunk_overlap=chunk_overlap):
            documents.append(chunk)
            metadatas.append({"source": file_path})
            ids.append(str(uuid.uuid4()))

    embeddings = embedder.encode(documents)
    store.add_texts(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)


def ask_question(settings: Settings, question: str, top_k: int) -> str:
    embedder = QwenEmbeddingClient(settings.embed_model_path, device=settings.device)
    store = ChromaVectorStore(settings.chroma_dir, settings.chroma_collection)
    llm = QwenLocalGenerator(settings.llm_model_path, device=settings.device)

    query_embedding = embedder.encode([question])[0]
    results = store.query(query_embedding=query_embedding, top_k=top_k)
    retrieved_docs = results.get("documents", [[]])[0]

    answer = llm.generate(question=question, context_chunks=retrieved_docs)
    return answer


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and query a local Chroma DB with Qwen models.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Ingest documents and build the Chroma database.")
    build_parser.add_argument("--docs-dir", type=Path, required=True, help="Directory containing .txt files.")
    build_parser.add_argument("--chunk-size", type=int, default=480, help="Word count per chunk.")
    build_parser.add_argument("--chunk-overlap", type=int, default=80, help="Overlap between chunks (words).")

    ask_parser = subparsers.add_parser("ask", help="Query the database and get an answer from Qwen3-32B.")
    ask_parser.add_argument("--question", type=str, required=True, help="Question to ask.")
    ask_parser.add_argument("--top-k", type=int, default=4, help="Number of results to retrieve.")

    args = parser.parse_args()
    settings = Settings()

    if args.command == "build":
        build_database(
            settings=settings,
            docs_dir=args.docs_dir,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
    elif args.command == "ask":
        answer = ask_question(settings=settings, question=args.question, top_k=args.top_k)
        print("\n--- Answer ---\n")
        print(answer)


if __name__ == "__main__":
    main()
