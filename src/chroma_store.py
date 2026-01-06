from typing import Iterable, List, Sequence

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.types import EmbeddingFunction, Embeddings


class PassthroughEmbeddingFunction(EmbeddingFunction):
    """Allows manual embeddings to be provided to Chroma."""

    def __call__(self, input: List[str]) -> Embeddings:  # pragma: no cover - not used directly
        raise NotImplementedError("Embeddings must be provided explicitly.")


class ChromaVectorStore:
    """Wrapper around Chroma for persistence and retrieval."""

    def __init__(self, persist_dir: str, collection_name: str):
        self.client: ClientAPI = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=PassthroughEmbeddingFunction(),
        )

    def add_texts(
        self,
        ids: Sequence[str],
        documents: Sequence[str],
        metadatas: Sequence[dict],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        self.collection.upsert(
            ids=list(ids),
            documents=list(documents),
            metadatas=list(metadatas),
            embeddings=list(embeddings),
        )

    def query(self, query_embedding: Sequence[float], top_k: int = 4) -> dict:
        return self.collection.query(query_embeddings=[query_embedding], n_results=top_k)

    def clear(self) -> None:
        self.collection.delete(where={})
