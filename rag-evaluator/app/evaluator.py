import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
client = OpenAI()

def load_test_set(path="data/test_set.json"):
    return json.load(open(path))

def retrieval_precision(retrieved_chunks, expected_source):
    hits = sum(
        1 for c in retrieved_chunks
        if c["source"] == expected_source
    )
    return hits / len(retrieved_chunks)

def llm_judge(question, answer):
    prompt = f"""
    Evaluate this answer.

    Question: {question}
    Answer: {answer}

    Score 0-5 based on:
    - factual correctness
    - completeness
    - faithfulness to provided context

    Return JSON:
    {{
      "score": int,
      "reason": "brief explanation"
    }}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return json.loads(response.choices[0].message.content)

def run_evaluation(rag_function):
    """
    rag_function(question) → {
        "answer": str,
        "chunks": list[dict]
    }
    """

    test_set = load_test_set()
    results = []

    for item in tqdm(test_set):
        q = item["question"]
        expected_source = item["relevant_source"]

        output = rag_function(q)

        precision = retrieval_precision(
            output["chunks"],
            expected_source
        )

        judge = llm_judge(q, output["answer"])

        results.append({
            "question": q,
            "retrieval_precision": precision,
            "llm_score": judge["score"],
            "judge_reason": judge["reason"]
        })

    df = pd.DataFrame(results)
    df.to_csv("results/evaluation_report.csv", index=False)

    print("\n===== SUMMARY =====")
    print("Avg Retrieval Precision:",
          df["retrieval_precision"].mean())
    print("Avg LLM Score:",
          df["llm_score"].mean())

    return df