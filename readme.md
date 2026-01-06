# Qwen Chroma RAG Starter

This project shows how to build a local vector database with **Qwen/Qwen3-Embedding-8B** and Chroma, then query it with a locally hosted **Qwen/Qwen3-32B** model.

## Prerequisites

- Local copies (or cached checkpoints) of:
  - `Qwen/Qwen3-Embedding-8B`
  - `Qwen/Qwen3-32B`
- Python 3.10+
- A GPU with enough memory for Qwen3-32B (FP16/BF16 is recommended). CPU inference is possible for the embedding model, but Qwen3-32B will be very slow without a GPU.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set these environment variables (via `.env` or your shell) to point to your local checkpoints:

- `QWEN_EMBED_MODEL_PATH` (default: `Qwen/Qwen3-Embedding-8B`)
- `QWEN_LLM_PATH` (default: `Qwen/Qwen3-32B`)
- `CHROMA_DB_DIR` (default: `./chroma_db`)
- `CHROMA_COLLECTION` (default: `qwen_docs`)

An example `.env`:

```bash
QWEN_EMBED_MODEL_PATH=/models/Qwen3-Embedding-8B
QWEN_LLM_PATH=/models/Qwen3-32B
CHROMA_DB_DIR=./chroma_db
CHROMA_COLLECTION=qwen_docs
```

## Usage

### 1) Build the vector database

Place your source documents under a directory (e.g., `data/sample_docs/`). Then run:

```bash
python -m src.cli build --docs-dir data/sample_docs --chunk-size 480 --chunk-overlap 80
```

This will:

1. Load all `.txt` files recursively from `--docs-dir`.
2. Chunk them into overlapping segments.
3. Generate embeddings with Qwen3-Embedding-8B.
4. Persist them into a Chroma collection at `CHROMA_DB_DIR`.

### 2) Ask a question with Qwen3-32B

After the database is built, query it with:

```bash
python -m src.cli ask --question "给我总结一下项目说明里的步骤" --top-k 4
```

The CLI will:

- Embed the query with the same embedding model.
- Retrieve the top results from Chroma.
- Compose a context prompt and ask Qwen3-32B to generate the answer.

## Notes

- The pipeline keeps all components local—no external API calls are required after the models are available on disk.
- If you want to inspect or clear the vector store, delete the folder at `CHROMA_DB_DIR`.
- Adjust `--chunk-size` and `--chunk-overlap` to match your document style; smaller chunks usually improve retrieval granularity.
