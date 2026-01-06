from typing import List

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


class QwenEmbeddingClient:
    """Generate embeddings with Qwen3-Embedding-8B (or a compatible local checkpoint)."""

    def __init__(self, model_path: str, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=self._preferred_dtype(),
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
        )
        self.model.eval()

    def encode(self, texts: List[str]) -> np.ndarray:
        """Return L2-normalized embeddings for the given texts."""
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=2048,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        token_embeddings = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"]
        pooled = self._mean_pooling(token_embeddings, attention_mask)
        normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return normalized.cpu().numpy()

    def _mean_pooling(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Mean pool using the attention mask."""
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def _preferred_dtype(self) -> torch.dtype:
        if self.device == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if self.device == "cuda":
            return torch.float16
        return torch.float32
