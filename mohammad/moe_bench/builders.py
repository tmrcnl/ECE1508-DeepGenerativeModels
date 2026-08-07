"""Model variant registry.

Every variant in the benchmark is built here so that the sweep, the training
loop and the evaluator all agree on what "moe-top1" means.

Author: Mohammad Al Dridi
"""

from dataclasses import dataclass, field
from typing import Callable, Dict

import torch
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

from .layers import AAGChunkMoELayer, GPT2MoELayer

BASE_MODEL = "openai-community/gpt2"

# Tiny mode swaps the 124M-parameter backbone for a 2-layer random one so the
# pipeline can be exercised on a laptop. A full GPT-2 AAG model is ~295M
# parameters, which needs several GB once gradients exist -- more than a CPU
# box usually has spare. Numbers produced in tiny mode are meaningless; it
# exists to prove the plumbing before spending GPU time.
_TINY = False

TINY_CONFIG = dict(n_layer=2, n_head=2, n_embd=128, n_inner=512, n_positions=256)


def set_tiny(enabled=True):
    global _TINY
    _TINY = enabled


@dataclass
class Variant:
    name: str
    label: str                       # for tables and figure legends
    build: Callable                  # () -> GPT2LMHeadModel
    family: str                      # "dense" | "moe" | "aag"
    notes: str = ""
    meta: Dict = field(default_factory=dict)


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _base_model():
    if _TINY:
        config = GPT2Config(vocab_size=50257, **TINY_CONFIG)
        model = GPT2LMHeadModel(config)
    else:
        model = GPT2LMHeadModel.from_pretrained(BASE_MODEL)
    model.config.pad_token_id = model.config.eos_token_id
    model.config.use_cache = False       # required with gradient checkpointing
    return model


def build_dense():
    """Vanilla GPT-2.  The baseline every MoE claim is measured against."""
    return _base_model()


def build_moe(num_experts=4, routing="top1", aux_loss_coef=0.05):
    model = _base_model()
    for block in model.transformer.h:
        block.mlp = GPT2MoELayer(
            block.mlp,
            num_experts=num_experts,
            routing=routing,
            aux_loss_coef=aux_loss_coef,
        )
    return model


def build_aag(num_chunks=8, num_options=4, aux_loss_coef=0.05, pretrained_init=True):
    model = _base_model()
    for block in model.transformer.h:
        block.mlp = AAGChunkMoELayer(
            original_mlp=block.mlp if pretrained_init else None,
            num_chunks=num_chunks,
            num_options=num_options,
            aux_loss_coef=aux_loss_coef,
        )
    return model


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

def _registry():
    variants = [
        Variant(
            "gpt2-dense", "GPT-2 (dense baseline)", build_dense, "dense",
            "Unmodified GPT-2. Reference point for perplexity and FLOPs.",
        ),
        Variant(
            "moe-top1", "MoE top-1 (4 experts)",
            lambda: build_moe(routing="top1"), "moe",
            "Tamara's sparse MoE. One expert active per token.",
        ),
        Variant(
            "moe-soft", "MoE soft routing (4 experts)",
            lambda: build_moe(routing="soft"), "moe",
            "Tamara's NAM notebook. Dense: all 4 experts run on every token.",
        ),
    ]

    # AAG scaling curve. 3072 and 768 are both divisible by these, which is
    # what lets each chunk be initialised from a slice of the pretrained MLP.
    for n in (1, 2, 4, 8, 16):
        variants.append(
            Variant(
                f"aag-c{n}", f"AAG ({n} chunks x 4 options)",
                (lambda n=n: build_aag(num_chunks=n)), "aag",
                f"Chunked virtual experts, num_chunks={n}.",
                meta={"num_chunks": n, "num_options": 4},
            )
        )

    # Cole's original configuration, for the "what changed" slide.
    variants.append(
        Variant(
            "aag-c1-random", "AAG (1 chunk, random init)",
            lambda: build_aag(num_chunks=1, pretrained_init=False), "aag",
            "Original notebook config: no pretrained init, no combinatorics.",
            meta={"num_chunks": 1, "num_options": 4, "pretrained_init": False},
        )
    )

    return {v.name: v for v in variants}


VARIANTS = _registry()

# Presets so the sweep can be run at three different budgets.
VARIANT_SETS = {
    "smoke": ["gpt2-dense", "aag-c2"],
    "core": ["gpt2-dense", "moe-top1", "moe-soft", "aag-c1", "aag-c8"],
    "scaling": ["aag-c1", "aag-c2", "aag-c4", "aag-c8", "aag-c16"],
    "all": list(VARIANTS),
}


def build(name):
    if name not in VARIANTS:
        raise KeyError(f"unknown variant {name!r}; known: {sorted(VARIANTS)}")
    return VARIANTS[name].build()


def load_checkpoint(model, path, device="cpu"):
    """Load a saved state dict into an already-built variant.

    ``strict=False`` because the team's checkpoints were saved from models
    whose MLPs had already been swapped, so key sets differ between variants.
    Any missing or unexpected keys are returned for inspection rather than
    swallowed.
    """
    from safetensors.torch import load_file

    path = str(path)
    if path.endswith(".safetensors"):
        state = load_file(path)
    else:
        state = torch.load(path, map_location=device)

    result = model.load_state_dict(state, strict=False)
    return {
        "missing": list(result.missing_keys),
        "unexpected": list(result.unexpected_keys),
    }
