"""Frozen Alpaca split shared by every variant.

The point of this module is that all variants see byte-identical data.  The
seed, the split fraction, the prompt template and the truncation length are
fixed here and nowhere else, so a difference in the results table is a
difference in the model rather than in the data pipeline.

Prompt formatting matches the team's notebooks exactly, including masking the
prompt tokens out of the labels with -100, so perplexity is measured on
response tokens only.

Author: Mohammad Al Dridi
"""

from dataclasses import dataclass

import torch
from datasets import load_dataset

COLUMNS = ("input_ids", "attention_mask", "labels")


def collate(features):
    """Stack pre-padded examples into a batch of tensors.

    Used instead of ``Dataset.set_format("torch")``. That path routes through
    the datasets torch formatter, which imports ``torchvision.io.VideoReader``
    for video columns -- an import that fails outright on torchvision builds
    that no longer export it (Colab, as of this writing). Every example here is
    already padded to ``max_length``, so a plain stack is all that is needed.
    """
    return {
        key: torch.tensor([f[key] for f in features], dtype=torch.long)
        for key in COLUMNS
    }

SEED = 42
DATASET = "tatsu-lab/alpaca"

PROMPT_WITH_INPUT = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes "
    "the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)

PROMPT_NO_INPUT = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n"
)


@dataclass
class DataConfig:
    """Training budget.  ``max_length`` drives cost more than anything else."""
    max_length: int = 256
    n_train: int = 8000
    n_eval: int = 1000
    test_size: float = 0.05


PRESETS = {
    # Runs on CPU in a couple of minutes. For checking the plumbing only.
    "smoke": DataConfig(max_length=64, n_train=32, n_eval=16),
    # ~20-30 min per variant on a T4. The setting for the real sweep.
    "budget": DataConfig(max_length=256, n_train=8000, n_eval=1000),
    # Matches the team's notebooks. Hours per variant.
    "full": DataConfig(max_length=512, n_train=None, n_eval=None),
}


def format_prompt(instruction, input_text=""):
    if input_text:
        return PROMPT_WITH_INPUT.format(instruction=instruction, input=input_text)
    return PROMPT_NO_INPUT.format(instruction=instruction)


def build_splits(tokenizer, config: DataConfig, cache_dir=None):
    """Return tokenized ``(train, eval)`` datasets.

    Truncation is applied to the full prompt+response string, so an example
    whose prompt already fills ``max_length`` contributes no supervised tokens.
    Those examples are dropped rather than left to contribute empty rows.
    """
    raw = load_dataset(DATASET, cache_dir=cache_dir)["train"]
    raw = raw.train_test_split(test_size=config.test_size, seed=SEED)

    train, eval_ = raw["train"], raw["test"]
    if config.n_train is not None:
        train = train.select(range(min(config.n_train, len(train))))
    if config.n_eval is not None:
        eval_ = eval_.select(range(min(config.n_eval, len(eval_))))

    max_length = config.max_length

    def preprocess(examples):
        input_ids, attention_mask, labels = [], [], []

        for instruction, input_text, output in zip(
            examples["instruction"], examples["input"], examples["output"]
        ):
            prompt = format_prompt(instruction, input_text)
            full = prompt + output + tokenizer.eos_token

            prompt_ids = tokenizer(prompt, truncation=True, max_length=max_length)["input_ids"]
            full_ids = tokenizer(full, truncation=True, max_length=max_length)["input_ids"]

            prompt_len = len(prompt_ids)
            label = [-100] * prompt_len + full_ids[prompt_len:]

            pad = max_length - len(full_ids)
            input_ids.append(full_ids + [tokenizer.pad_token_id] * pad)
            attention_mask.append([1] * len(full_ids) + [0] * pad)
            labels.append(label + [-100] * pad)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    columns = train.column_names
    train = train.map(preprocess, batched=True, remove_columns=columns)
    eval_ = eval_.map(preprocess, batched=True, remove_columns=columns)

    # Drop examples with no supervised tokens; they add cost and no signal.
    def has_targets(example):
        return any(token != -100 for token in example["labels"])

    train = train.filter(has_targets)
    eval_ = eval_.filter(has_targets)

    # Left in plain python format on purpose; batching goes through collate().
    return train, eval_


# Prompts used for the qualitative generation table and the routing heatmap.
# Grouped by task type so expert usage can be compared across categories.
PROBE_PROMPTS = {
    "Code & Programming": [
        "Write a Python function to calculate the Fibonacci sequence.",
        "How do I parse a JSON string in JavaScript?",
        "Fix the bug in this C++ array iteration loop.",
    ],
    "Math & Logic": [
        "Solve for x in the equation: 3x + 15 = 45.",
        "If all A are B and all B are C, are all A C?",
        "Calculate the probability of rolling a double six with two dice.",
    ],
    "Grammar & Editing": [
        "Correct the grammar in the sentence: He do not have no money.",
        "Rephrase this sentence to sound more professional: I want to quit.",
        "Fix the spelling errors in the word: yesturday.",
    ],
    "Creative Writing": [
        "Write a short poem about a quiet rainy afternoon.",
        "List three healthy snacks for a late-night movie.",
        "Draft an opening sentence for a science fiction novel.",
    ],
    "Factual QA": [
        "What is the capital of Canada?",
        "Who proposed the theory of general relativity?",
        "Which planet in our solar system is the largest?",
    ],
}
