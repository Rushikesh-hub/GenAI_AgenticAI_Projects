# RAG QA System

> Part of the [30-Day GenAI & Agentic AI Engineering Roadmap]
> (../README.md)  —  Day 31

A production-ready **Retrieval-Augmented Generation (RAG)** API.
Ingests documents, builds a semantic search index, and answers
natural language questions grounded in your actual content.

## Architecture
```
Documents → Chunker → SentenceTransformer → FAISS Index
                                                  ↓
Question  → Embedding  → FAISS Search  → Context → GPT-4 → Answer
```

## Tech Stack
| Layer | Technology |
|-------|-----------|
| API   | FastAPI + Uvicorn |
| Embed | all-MiniLM-L6-v2 (SentenceTransformers) |
| Store | FAISS IndexFlatL2 |
| LLM   | GPT-4o-mini / GPT-4o (OpenAI) |
| Infra | Docker |

## Quick Start
```bash
# With Docker
docker build -t rag-qa-system .
docker run -p 8000:8000 -e OPENAI_API_KEY=your-key rag-qa-system

# Without Docker
pip install -r requirements.txt
cd app && uvicorn main:app --reload
```

## API
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET    | /        | Health check |
| GET    | /status  | Index ready? |
| POST   | /ingest  | Index data/ docs |
| POST   | /ask     | Ask a question |

## Example
```bash
curl -X POST http://localhost:8000/ingest
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the benefits of RAG?"}'
```
