import json
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

from prompt_strategies import (
    zero_shot,
    few_shot,
    chain_of_thought,
    self_critique,
    react_style
)

load_dotenv()
client = OpenAI()

strategies = {
    "zero_shot": zero_shot,
    "few_shot": few_shot,
    "chain_of_thought": chain_of_thought,
    "self_critique": self_critique,
    "react_style": react_style
}

def load_tasks():
    return json.load(open("data/test_prompts.json"))

def run():
    tasks = load_tasks()
    results = []

    for item in tqdm(tasks):
        task = item["task"]

        for name, strategy in strategies.items():
            prompt = strategy(task)

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )

            output = response.choices[0].message.content

            results.append({
                "task": task,
                "strategy": name,
                "output": output,
                "tokens_used": response.usage.total_tokens
            })

    df = pd.DataFrame(results)
    df.to_csv("results/prompt_experiments.csv", index=False)

    print("\nExperiment completed.")
    print("Saved to results/prompt_experiments.csv")

if __name__ == "__main__":
    run()