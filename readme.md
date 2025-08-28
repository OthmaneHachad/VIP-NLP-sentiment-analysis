# VIP Assignment 1 — Part 1 (Hugging Face + LLM)

This repo implements **Part 1** of the assignment: using a small open-source chat model on Hugging Face to run a simple NLP task against a **finance-related text dataset**.

---

## What this does

- Loads a small subset of sentences from the **`takala/financial_phrasebank`** dataset.  
  - Config options: `sentences_allagree`, `sentences_75agree`, `sentences_66agree`, `sentences_50agree`.  
  - Default: `sentences_allagree`.
- Defines an **LLM task**: *sentiment classification with a one-sentence rationale*.
- Sends **system + user prompts** to an **open instruct model** via **Hugging Face InferenceClient**. (Here we chose Mistral 7B)
- Saves predictions to `outputs/predictions.jsonl` (JSONL format = one JSON object per line).
- Prints a preview table in the terminal.

---

## Setup

1. **Python**: 3.9–3.11 recommended.

2. **Create a Hugging Face account** and generate a **Read** token:  
   [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt

4. **Run Task**:
(default) ```bash 
python main.py

(with custom options)
```bash
python main.py \
  --model YOUR_MODEL \
  --dataset_config CONFIG_OPTION \
  --num_samples 20 \
  --out outputs/custom.jsonl
