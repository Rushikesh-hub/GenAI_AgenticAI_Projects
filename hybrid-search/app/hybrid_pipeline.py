import os, pickle, logging
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_PATH = "faiss.index"
CHUNKS_PATH = "chunks.pkl"
BM25_PATH = "bm25.pkl"

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

embedder = SentenceTransformer(MODEL_NAME)

def tokenize(text):
    return text.lower().split()

def ingest_documents(data_dir="data"):
    docs = []
    for fp in Path(data_dir).iterdir():
        if fp.suffix == ".txt":
            text = fp.read_text()
        elif fp.suffix == ".pdf":
            reader = PdfReader(str(fp))
            text = "\n".join(p.extract_text() or "" for p in reader.pages)
        else:
            continue
        docs.append({"content": text, "source": fp.name})

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = []
    for doc in docs:
        for chunk in splitter.split_text(doc["content"]):
            chunks.append({"text": chunk, "source": doc["source"]})

    texts = [c["text"] for c in chunks]

    # Dense embeddings
    embeddings = embedder.encode(texts).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # Sparse BM25
    tokenized = [tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized)

    faiss.write_index(index, INDEX_PATH)
    pickle.dump(chunks, open(CHUNKS_PATH, "wb"))
    pickle.dump(bm25, open(BM25_PATH, "wb"))

    return len(chunks)

def hybrid_retrieve(query, k=5, alpha=0.5):
    """
    alpha = weight for dense score
    (1-alpha) = weight for BM25 score
    """
    index = faiss.read_index(INDEX_PATH)
    chunks = pickle.load(open(CHUNKS_PATH, "rb"))
    bm25 = pickle.load(open(BM25_PATH, "rb"))

    # Dense
    q_vec = embedder.encode([query]).astype("float32")
    dense_dist, dense_idx = index.search(q_vec, len(chunks))

    dense_scores = {
        idx: 1/(1+dist)  # convert L2 to similarity
        for dist, idx in zip(dense_dist[0], dense_idx[0])
    }

    # Sparse
    tokenized_query = tokenize(query)
    sparse_scores_list = bm25.get_scores(tokenized_query)

    sparse_scores = {
        i: sparse_scores_list[i]
        for i in range(len(chunks))
    }

    # Normalize sparse
    max_sparse = max(sparse_scores.values()) + 1e-6
    for i in sparse_scores:
        sparse_scores[i] /= max_sparse

    # Fusion
    combined = {}
    for i in range(len(chunks)):
        combined[i] = (
            alpha * dense_scores.get(i, 0) +
            (1 - alpha) * sparse_scores.get(i, 0)
        )

    ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:k]

    results = []
    for idx, score in ranked:
        results.append({
            "text": chunks[idx]["text"],
            "source": chunks[idx]["source"],
            "hybrid_score": score
        })

    return results