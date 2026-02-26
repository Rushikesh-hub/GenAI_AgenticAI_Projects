import sys
sys.path.append("../../rag-qa-system/app")

from rag_pipeline import retrieve
from openai import OpenAI

client = OpenAI()

def rag_function(question):
    chunks = retrieve(question, k=5)

    context = "\n".join([c["text"] for c in chunks])

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system",
             "content": "Answer only from context."},
            {"role": "user",
             "content": f"Context:\n{context}\n\nQuestion:{question}"}
        ],
        temperature=0
    )

    return {
        "answer": response.choices[0].message.content,
        "chunks": chunks
    }