from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class QwenLocalGenerator:
    """Generate answers with a local Qwen3-32B (or compatible) checkpoint."""

    def __init__(self, model_path: str, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto" if self.device == "cuda" else None,
            torch_dtype=self._preferred_dtype(),
            trust_remote_code=True,
        )
        self.model.eval()

    def generate(
        self,
        question: str,
        context_chunks: List[str],
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.9,
    ) -> str:
        system = system_prompt or "You are a helpful assistant. Use the provided context to answer."
        context_block = "\n\n".join(context_chunks) if context_chunks else "无可用上下文。"
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"已知信息：\n{context_block}\n\n问题：{question}"},
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated_text = self.tokenizer.decode(generated[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)
        return generated_text.strip()

    def _preferred_dtype(self) -> torch.dtype:
        if self.device == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if self.device == "cuda":
            return torch.float16
        return torch.float32
