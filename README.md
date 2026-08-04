# Pharmaceutical Catalog RAG

药品说明书 RAG 问答系统，使用 FastAPI、Vue 3、Milvus、Redis 和 Ollama。

## Prerequisites

- Python 3.11+
- Node.js 20+
- Docker Desktop
- Ollama with the configured model available locally

## Run locally

1. Copy `.env.example` to `.env` and adjust local paths or service addresses if needed.
2. Start the infrastructure services:

   ```powershell
   docker compose up -d
   ```

3. Install Python dependencies and start the API:

   ```powershell
   uv sync
   uv run uvicorn api.main:app --host 127.0.0.1 --port 8001
   ```

4. In another terminal, install and start the Vue frontend:

   ```powershell
   cd frontend
   npm ci
   npm run dev
   ```

Open the address printed by Vite, normally `http://127.0.0.1:5173`.

## Data and models

The raw drug documents, generated indexes, downloaded Hugging Face models, uploads, logs, virtual environments, and `.env` are deliberately ignored by Git. They may contain large files, local data, or credentials. Import source documents with the application or scripts in `scripts/` after cloning.

## Project layout

- `api/`: FastAPI routes
- `core/`, `retrieval/`, `generation/`, `vectorization/`: RAG pipeline
- `frontend/`: Vue 3 + Vite application
- `scripts/`: data import and indexing scripts
- `docker-compose.yml`: Milvus, Redis, MinIO, and etcd
