#!/usr/bin/env python3
"""
Part 1 — Using Hugging Face + an open-source chat model for a simple NLP task.

Task: Sentiment classification with justification on finance sentences.
- Dataset: 'takala/financial_phrasebank' (text-only finance dataset)
- Model: default 'mistralai/Mistral-7B-Instruct-v0.3' (open instruct model)
- API: huggingface_hub.InferenceClient (hosted inference)

Set HF_TOKEN via environment or .env.
"""
import os
import json
import argparse
from pathlib import Path

from huggingface_hub import InferenceClient
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from src.prompts import build_sentiment_prompts
from src.data_utils import load_finance_dataset

console = Console()

def run_inference(model: str, texts, max_new_tokens: int = 256, temperature: float = 0.2):
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is not set. See README or .env.")

    client = InferenceClient(model=model, token=token, timeout=60)

    outputs = []
    for t in texts:
        sys_prompt, user_prompt = build_sentiment_prompts(t)

        # Chat messages format expected by conversational providers
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Call the chat completion endpoint
        resp = client.chat_completion(
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
        )

        # Extract text from the first choice
        # huggingface_hub returns an object with choices[0].message.content
        raw = resp.choices[0].message["content"] if resp.choices else ""

        # Try to parse JSON from the model output; fallback to raw text
        try:
            start = raw.find("{")
            end = raw.rfind("}")
            json_str = raw[start : end + 1] if (start != -1 and end != -1) else raw
            parsed = json.loads(json_str)
        except Exception:
            parsed = {"sentiment": "unknown", "rationale": raw}

        outputs.append({"text": t, "prediction": parsed})

    return outputs

def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Part 1 — HF Inference on finance text")
    parser.add_argument(
        "--model",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.3",
        help="Hugging Face model id. (default: mistralai/Mistral-7B-Instruct-v0.3)",
    )
    parser.add_argument("--num_samples", type=int, default=12, help="Number of sentences to evaluate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument("--out", type=str, default="outputs/predictions.jsonl", help="Where to save JSONL outputs.")
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="sentences_allagree",
        choices=["sentences_allagree", "sentences_75agree", "sentences_66agree", "sentences_50agree"],
        help="Financial PhraseBank agreement threshold",
    )
    args = parser.parse_args()

    console.rule("[bold]Part 1: Using Hugging Face — Finance Sentiment Task")
    console.print("[bold]Dataset:[/bold] takala/financial_phrasebank")
    console.print(f"[bold]Config:[/bold] {args.dataset_config}")
    console.print(f"[bold]Model:[/bold] {args.model}")
    console.print(f"[bold]Samples:[/bold] {args.num_samples}\n")

    texts = load_finance_dataset(
        split_size=args.num_samples,
        seed=args.seed,
        config=args.dataset_config,
    )
    results = run_inference(args.model, texts)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    table = Table(title="Predictions Preview")
    table.add_column("Sentence", overflow="fold", max_width=80)
    table.add_column("Sentiment")
    table.add_column("Rationale", overflow="fold", max_width=60)

    for r in results[: min(5, len(results))]:
        sent = r["prediction"].get("sentiment", "unknown")
        rat = r["prediction"].get("rationale", "") or ""
        table.add_row(r["text"], str(sent), rat)

    console.print(table)
    console.print(f"\nSaved all predictions to [bold]{out_path}[/bold]")

if __name__ == "__main__":
    main()