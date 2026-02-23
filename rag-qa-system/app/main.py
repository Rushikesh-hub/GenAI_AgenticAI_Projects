# app/main.py  —  rag-qa-system project
"""
RAG QA REST API.

Endpoints:
  GET  /         health check
  GET  /status   is the FAISS index ready?
  POST /ingest   index all docs in data/
  POST /ask      answer a question via RAG
"""

import os, logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv

from rag_pipeline import ingest_documents, retrieve

# ── Bootstrap ─────────────────────────────────────────────────────
load_dotenv()   # reads OPENAI_API_KEY from .env

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

if not os.getenv('OPENAI_API_KEY'):
    raise RuntimeError('OPENAI_API_KEY not set — check your .env file')

openai_client = OpenAI()   # auto-reads OPENAI_API_KEY

app = FastAPI(
    title='RAG QA API',
    description='Q&A powered by Retrieval-Augmented Generation',
    version='1.0.0'
)
app.add_middleware(CORSMiddleware, allow_origins=['*'],
                   allow_methods=['*'], allow_headers=['*'])

# ── Schemas ───────────────────────────────────────────────────────
class AskRequest(BaseModel):
    question: str  = Field(..., min_length=3, max_length=500,
                           example='What are the benefits of RAG?')
    top_k:    int  = Field(default=5, ge=1, le=10)
    model:    str  = Field(default='gpt-4o-mini')

class AskResponse(BaseModel):
    question:         str
    answer:           str
    sources:          list[str]
    retrieved_chunks: int
    model_used:       str

# ── System prompt ─────────────────────────────────────────────────
SYSTEM_PROMPT = '''You are an accurate assistant that answers questions
based ONLY on the provided context. Rules:
1. Use only information from the context.
2. If the context is insufficient, say 'I cannot answer from these documents.'
3. Always cite which document your answer comes from.
4. Never invent information not present in the context.'''


def _build_prompt(question: str, chunks: list[dict]) -> str:
    context = '\n\n---\n\n'.join(
        f'[Source: {c["source"]}]\n{c["text"]}' for c in chunks
    )
    return (
        f'CONTEXT:\n{context}\n\n'
        f'QUESTION: {question}\n\n'
        'Answer using ONLY the context above. Cite source documents.'
    )

# ── Endpoints ─────────────────────────────────────────────────────
@app.get('/')
def health():                       return {'status': 'healthy', 'version': '1.0.0'}

@app.get('/status')
def status():
    ready = Path('faiss.index').exists() and Path('chunks.pkl').exists()
    return {'index_ready': ready,
            'message': 'Ready' if ready else 'Run POST /ingest first'}

@app.post('/ingest')
def ingest():
    try:
        n = ingest_documents(data_dir='data')
        return {'success': True, 'chunks_indexed': n}
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, f'Ingestion failed: {e}')

@app.post('/ask', response_model=AskResponse)
def ask(req: AskRequest):
    # 1 — Retrieve
    try:
        chunks = retrieve(query=req.question, k=req.top_k)
    except FileNotFoundError:
        raise HTTPException(400, 'Index not found. POST /ingest first.')
    if not chunks:
        raise HTTPException(404, 'No relevant documents found.')

    # 2 — Build prompt
    prompt = _build_prompt(req.question, chunks)

    # 3 — Generate
    try:
        resp = openai_client.chat.completions.create(
            model=req.model,
            messages=[
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user',   'content': prompt}
            ],
            temperature=0.1,   # low = factual, deterministic
            max_tokens=800,
        )
        answer = resp.choices[0].message.content
    except Exception as e:
        raise HTTPException(502, f'OpenAI error: {e}')

    # 4 — Return
    return AskResponse(
        question=req.question,
        answer=answer,
        sources=list({c['source'] for c in chunks}),
        retrieved_chunks=len(chunks),
        model_used=req.model
    )
