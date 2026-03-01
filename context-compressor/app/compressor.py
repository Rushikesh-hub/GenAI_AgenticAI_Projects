import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

load_dotenv()
client = OpenAI()

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200

def load_document(path="data/long_doc.txt"):
    return open(path).read()

def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_text(text)

def summarize_chunk(chunk):
    prompt = f"""
    Summarize the following text concisely while preserving key facts:

    {chunk}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content

def compress_document():
    text = load_document()
    chunks = chunk_text(text)

    summaries = []

    print(f"Total chunks: {len(chunks)}")

    for chunk in tqdm(chunks):
        summary = summarize_chunk(chunk)
        summaries.append(summary)

    combined = "\n".join(summaries)

    # Final reduction step
    final_prompt = f"""
    Combine the following summaries into a single coherent,
    compressed summary preserving all critical information:

    {combined}
    """

    final_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": final_prompt}],
        temperature=0
    )

    return final_response.choices[0].message.content