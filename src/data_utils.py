from datasets import load_dataset
from typing import List
import random

def load_finance_dataset(split_size: int = 20, seed: int = 42, config: str = "sentences_allagree") -> List[str]:
    """
    Loads a small subset of sentences from 'takala/financial_phrasebank' with the given config.
    Valid configs: sentences_allagree, sentences_75agree, sentences_66agree, sentences_50agree
    """
    ds = load_dataset("takala/financial_phrasebank", config)
    all_texts = [row["sentence"] for row in ds["train"]]
    rng = random.Random(seed)
    rng.shuffle(all_texts)
    return all_texts[:split_size]