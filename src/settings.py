from dataclasses import dataclass
import os


@dataclass
class Settings:
    """Runtime configuration sourced from environment variables."""

    embed_model_path: str = os.getenv("QWEN_EMBED_MODEL_PATH", "Qwen/Qwen3-Embedding-8B")
    llm_model_path: str = os.getenv("QWEN_LLM_PATH", "Qwen/Qwen3-32B")
    chroma_dir: str = os.getenv("CHROMA_DB_DIR", "./chroma_db")
    chroma_collection: str = os.getenv("CHROMA_COLLECTION", "qwen_docs")
    device: str = os.getenv("DEVICE", "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu")
