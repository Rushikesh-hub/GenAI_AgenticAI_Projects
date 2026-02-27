def zero_shot(task):
    return f"Answer the following clearly:\n\n{task}"

def few_shot(task):
    return f"""
Example:
Q: What is RAG?
A: RAG combines retrieval with LLM generation.

Now answer:
{task}
"""

def chain_of_thought(task):
    return f"""
Think step by step before answering.

Question:
{task}
"""

def self_critique(task):
    return f"""
Answer the question. Then critique your answer and improve it.

Question:
{task}
"""

def react_style(task):
    return f"""
You are a reasoning agent.

Thought: break problem into steps.
Action: reason internally.
Final Answer:

Question:
{task}
"""