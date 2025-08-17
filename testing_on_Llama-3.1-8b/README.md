# HippoRAG — Project Overview

HippoRAG (Hierarchical Interleaved Retrieval-Augmented Generation) is a RAG system that combines a Knowledge Graph with traditional retrieval to improve multi-hop reasoning and factual accuracy. This project ships with a Docker-based deployment, an interactive Gradio dashboard, an evaluation toolkit, and a clean, extensible Python codebase. By default, it uses the `llama3:8b` LLM (via Ollama) and `facebook/contriever` for embeddings.

## Goals
- Integrate a Knowledge Graph into the RAG pipeline to answer complex, multi-hop questions.
- Support NVIDIA GPUs for fast inference with a local LLM service (Ollama).
- Provide a lightweight Docker deployment and a comprehensive evaluation suite.

## Key features
- Knowledge Graph-enhanced multi-hop reasoning.
- Gradio dashboard (port 7860) for interactive testing/evaluation.
- Ollama integration (port 11434) as the LLM service with GPU support.
- Tests and evaluations across multiple datasets (AM, DS, MCS).
- Logging, result persistence, and flexible configuration via environment variables.

## Architecture & Technology
- Core services:
  - `hipporag` (Gradio app): 7860
  - `ollama` (LLM service): 11434
- Default LLM: `llama3:8b` (override via environment variable).
- Embedding model: `facebook/contriever`.
- Optional: Qdrant Vector DB (not enabled by default in docker-compose).

## System requirements (recommended)
- Docker Desktop (Windows/Linux/macOS)
- NVIDIA GPU ≥ 8GB VRAM (optimal for `llama3:8b`)
- ≥ 16GB RAM, ≥ 20GB free disk space for models

## Quick start
### Manual (Docker)
```bash
# Start services
docker compose up -d

# Check status
docker compose ps

# Health checks
curl http://localhost:7860            # Gradio Dashboard
curl http://localhost:11434/api/tags  # Ollama models
```

### Open the dashboard
- Visit: `http://localhost:7860`

## Quick evaluation run
```bash
# Quick evaluation (2 questions) on AM dataset
docker exec hipporag_app python main_hipporag.py --dataset AM --test_type closed_end --max_questions 2

# Interactive mode
docker exec -it hipporag_app python main_hipporag.py --dataset AM --interactive

# Full evaluation
docker exec hipporag_app python main_hipporag.py --dataset AM --test_type all
```

## Project structure (condensed)
```
my_docker_project_docker/
├── config/                 # Config (docker, host, embedding model)
├── utils/                  # Utilities: embedding, rerank, Ollama client, graph analytics
├── tests/                  # Tests: QA, Docker integration, deployment validation
├── scripts/                # Setup/start scripts (Win/Linux)
├── docs/                   # Detailed docs, quickstart, Docker guide
├── QA for testing/         # Sample datasets (AM, DS, MCS)
├── outputs/, logs/, results/, embedding_stores/
├── docker-compose.yml, Dockerfile, requirements.txt
├── main_hipporag.py, hipporag_complete.py, hipporag_evaluation.py
└── hipporag_gradio_app.py, simple_gradio_app.py
```

## Quick config (env variables)
```env
LLM_NAME=llama3:8b
OLLAMA_BASE_URL=http://ollama:11434/api/generate
EMBEDDING_MODEL=facebook/contriever
SAVE_DIR=/app/outputs
LOGS_DIR=/app/logs
GRADIO_SERVER_NAME=0.0.0.0
GRADIO_SERVER_PORT=7860
DOCKER_ENV=true
```

## Useful commands
```bash
# View logs
docker logs hipporag_app
docker logs hipporag_ollama

# Restart services
docker compose restart

# Full reset (removes volumes)
docker compose down -v
docker system prune -a
docker volume prune
```

---

If you run into issues, check container logs (`docker logs hipporag_app`).


