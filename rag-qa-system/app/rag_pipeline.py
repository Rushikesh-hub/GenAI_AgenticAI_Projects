# app/rag_pipeline.py  —  rag-qa-system project
"""
Core RAG pipeline.
Public API:
  ingest_documents(data_dir)  →  int  (chunk count)
  retrieve(query, k)          →  list[dict]
"""

import os, pickle, logging
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

# ── Config ────────────────────────────────────────────────────────
MODEL_NAME    = 'all-MiniLM-L6-v2'   # 384-dim, ~80MB, fast & accurate
CHUNK_SIZE    = 512    # characters per chunk
CHUNK_OVERLAP = 64     # shared chars between consecutive chunks
TOP_K         = 5      # chunks returned per query
INDEX_PATH    = 'faiss.index'
CHUNKS_PATH   = 'chunks.pkl'

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

# Load once at startup — expensive operation
log.info(f'Loading embedding model: {MODEL_NAME}')
embedder = SentenceTransformer(MODEL_NAME)
log.info('Embedding model ready')


# ── Helpers ───────────────────────────────────────────────────────
def _load_documents(data_dir: str) -> list[dict]:
    """Read .txt, .md, .pdf from data_dir. Returns [{content, source}]."""
    docs, path = [], Path(data_dir)
    if not path.exists():
        raise FileNotFoundError(f'Directory not found: {data_dir}')
    for fp in sorted(path.iterdir()):
        if fp.suffix.lower() == '.pdf':
            reader = PdfReader(str(fp))
            text = '\n'.join(p.extract_text() or '' for p in reader.pages)
        elif fp.suffix.lower() in {'.txt', '.md'}:
            text = fp.read_text(encoding='utf-8', errors='ignore')
        else:
            continue
        if text.strip():
            docs.append({'content': text, 'source': fp.name})
            log.info(f'  Loaded: {fp.name}  ({len(text)} chars)')
    log.info(f'Total documents loaded: {len(docs)}')
    return docs


def _chunk_documents(docs: list[dict]) -> list[dict]:
    """Split docs into overlapping chunks. Returns [{text, source}]."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=['\n\n', '\n', '. ', ' ', '']
    )
    chunks = []
    for doc in docs:
        for chunk_text in splitter.split_text(doc['content']):
            chunks.append({'text': chunk_text, 'source': doc['source']})
    log.info(f'Total chunks created: {len(chunks)}')
    return chunks


def _build_index(chunks: list[dict]) -> faiss.Index:
    """Embed chunks and build FAISS L2 index."""
    texts = [c['text'] for c in chunks]
    log.info(f'Embedding {len(texts)} chunks...')
    embeddings = embedder.encode(
        texts, show_progress_bar=True,
        batch_size=32, convert_to_numpy=True
    ).astype('float32')   # FAISS requires float32
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    log.info(f'FAISS index built: {index.ntotal} vectors, dim={dim}')
    return index


# ── Public API ────────────────────────────────────────────────────
def ingest_documents(data_dir: str = 'data') -> int:
    """Full pipeline: load → chunk → embed → index → save. Returns chunk count."""
    docs   = _load_documents(data_dir)
    chunks = _chunk_documents(docs)
    index  = _build_index(chunks)
    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, 'wb') as f:
        pickle.dump(chunks, f)
    log.info(f'Index saved. {len(chunks)} chunks ready.')
    return len(chunks)


def retrieve(query: str, k: int = TOP_K) -> list[dict]:
    """Return top-k most relevant chunks for a query."""
    if not (Path(INDEX_PATH).exists() and Path(CHUNKS_PATH).exists()):
        raise FileNotFoundError('No index found. Call ingest_documents() first.')
    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, 'rb') as f:
        chunks = pickle.load(f)
    q_vec = embedder.encode([query]).astype('float32')
    distances, indices = index.search(q_vec, k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1: continue
        results.append({
            'text':   chunks[idx]['text'],
            'source': chunks[idx]['source'],
            'score':  float(dist)   # lower = more similar
        })
    return results
