from fastapi import FastAPI
from pydantic import BaseModel
from hybrid_pipeline import ingest_documents, hybrid_retrieve

app = FastAPI(title="Hybrid RAG API")

class Query(BaseModel):
    question: str
    top_k: int = 5
    alpha: float = 0.5

@app.post("/ingest")
def ingest():
    n = ingest_documents("data")
    return {"chunks_indexed": n}

@app.post("/search")
def search(q: Query):
    results = hybrid_retrieve(
        q.question,
        k=q.top_k,
        alpha=q.alpha
    )
    return {"results": results}