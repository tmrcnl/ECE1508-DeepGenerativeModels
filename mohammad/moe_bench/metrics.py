"""Shared metrics for every model variant.

The routing-health definitions (coefficient of variation, Shannon entropy)
follow Tamara's ``GPT2_MoE_aux_NAM.ipynb`` so the numbers stay comparable with
what she has already reported.  Two things are new here:

*   Perplexity is token-weighted.  Averaging per-batch losses, as the
    notebooks do, over-weights batches with few supervised tokens; the last
    batch of a split is usually short.
*   ``capacity_report`` counts *active* parameters per token, which is what
    separates a sparse MoE from a dense one and is not measured anywhere in
    the existing notebooks.

Author: Mohammad Al Dridi
"""

import math
import time
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from .layers import AAGChunkMoELayer, GPT2MoELayer


# --------------------------------------------------------------------------
# Routing telemetry helpers
# --------------------------------------------------------------------------

def routed_layers(model):
    """Yield ``(index, layer)`` for every block whose MLP does routing."""
    for i, block in enumerate(model.transformer.h):
        if isinstance(block.mlp, (GPT2MoELayer, AAGChunkMoELayer)):
            yield i, block.mlp


def set_tracking(model, enabled, reset=True):
    for _, mlp in routed_layers(model):
        if reset:
            mlp.routing_log = []
        mlp.track_routing = enabled


def _selection_counts(logs, num_options, keep_mask=None):
    """Collapse a layer's routing log into per-option token counts.

    ``logs`` is a list of tensors, either ``[N]`` (one expert per token) or
    ``[N, num_chunks]`` (one option per chunk per token).  ``keep_mask`` is a
    flat boolean over tokens used to drop padding.
    """
    if not logs:
        return None

    selections = torch.cat([t.reshape(t.shape[0], -1) for t in logs], dim=0)
    if keep_mask is not None:
        if keep_mask.shape[0] != selections.shape[0]:
            # A log that is not one row per token cannot be aligned with the
            # padding mask; counting it unmasked would silently mix padding
            # into the usage statistics, so drop the batch instead.
            return None
        selections = selections[keep_mask]
    if selections.numel() == 0:
        return None

    counts = torch.bincount(selections.reshape(-1), minlength=num_options)
    return counts.float().numpy()


def _cv_and_entropy(counts):
    total = counts.sum()
    if total == 0:
        return float("nan"), float("nan")
    freq = counts / total
    cv = float(np.std(freq) / (np.mean(freq) + 1e-8))
    nonzero = freq[freq > 0]
    entropy = float(-(nonzero * np.log2(nonzero)).sum())
    return cv, entropy


def num_options_of(mlp):
    return mlp.num_options if isinstance(mlp, AAGChunkMoELayer) else mlp.num_experts


# --------------------------------------------------------------------------
# Quality
# --------------------------------------------------------------------------

@torch.no_grad()
def evaluate_perplexity(model, dataset, device, batch_size=8, collect_routing=True):
    """Token-weighted NLL and perplexity over supervised (label != -100) tokens."""
    from .data import collate

    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate)

    total_nll, total_tokens = 0.0, 0
    per_layer_counts = defaultdict(lambda: 0)

    if collect_routing:
        set_tracking(model, True)

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

        # Standard causal shift: position t predicts token t+1.
        shift_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
        shift_labels = labels[:, 1:].reshape(-1)

        losses = torch.nn.functional.cross_entropy(
            shift_logits.float(), shift_labels, ignore_index=-100, reduction="sum"
        )
        n_tokens = (shift_labels != -100).sum()

        total_nll += losses.item()
        total_tokens += int(n_tokens)

        if collect_routing:
            keep = attention_mask.reshape(-1).bool().cpu()
            for idx, mlp in routed_layers(model):
                counts = _selection_counts(mlp.routing_log, num_options_of(mlp), keep)
                if counts is not None:
                    per_layer_counts[idx] = per_layer_counts[idx] + counts
                mlp.routing_log = []

    if collect_routing:
        set_tracking(model, False, reset=False)

    mean_nll = total_nll / max(total_tokens, 1)
    return {
        "eval_tokens": total_tokens,
        "nll": mean_nll,
        "perplexity": math.exp(mean_nll) if mean_nll < 30 else float("inf"),
        "_routing_counts": {k: v for k, v in per_layer_counts.items()},
    }


def routing_health(routing_counts):
    """Per-layer load imbalance and entropy, plus model-level means."""
    # A dense model has no routers, so every field is legitimately absent.
    empty = {"per_layer": {}, "mean_cv": None, "mean_entropy": None,
             "max_entropy": None, "entropy_ratio": None}
    if not routing_counts:
        return empty

    per_layer, cvs, entropies = {}, [], []
    max_entropy = None

    for idx in sorted(routing_counts):
        counts = routing_counts[idx]
        cv, entropy = _cv_and_entropy(counts)
        share = counts / max(counts.sum(), 1)
        per_layer[idx] = {
            "cv": cv,
            "entropy": entropy,
            "usage": share.tolist(),
            "dominant": int(np.argmax(share)),
            "dominant_share": float(share.max()),
        }
        cvs.append(cv)
        entropies.append(entropy)
        max_entropy = math.log2(len(counts))

    return {
        "per_layer": per_layer,
        "mean_cv": float(np.nanmean(cvs)),
        "mean_entropy": float(np.nanmean(entropies)),
        "max_entropy": max_entropy,
        # 1.0 means perfectly balanced, 0.0 means fully collapsed.
        "entropy_ratio": float(np.nanmean(entropies) / max_entropy) if max_entropy else None,
    }


# --------------------------------------------------------------------------
# Capacity: the column the notebooks are missing
# --------------------------------------------------------------------------

def capacity_report(model):
    """Stored vs active parameters, and how many virtual experts that buys.

    A standard MoE buys ``num_experts`` routing choices for ``num_experts``
    times the storage.  AAG buys ``num_options ** num_chunks`` choices for the
    same storage and the same active compute, which is the claim the project
    proposal rests on.
    """
    total = sum(p.numel() for p in model.parameters())

    mlp_stored = mlp_active = 0
    virtual = 1.0
    n_routed = 0

    for _, mlp in routed_layers(model):
        mlp_stored += mlp.stored_params()
        mlp_active += mlp.active_params_per_token()
        virtual *= mlp.virtual_experts()
        n_routed += 1

    if n_routed == 0:
        # Dense GPT-2: every MLP parameter is active for every token.
        for block in model.transformer.h:
            n = sum(p.numel() for p in block.mlp.parameters())
            mlp_stored += n
            mlp_active += n
        virtual = 1.0

    non_mlp = total - mlp_stored
    return {
        "total_params": total,
        "mlp_stored_params": mlp_stored,
        "mlp_active_params_per_token": mlp_active,
        "active_params_per_token": non_mlp + mlp_active,
        "sparsity_ratio": mlp_active / mlp_stored if mlp_stored else 1.0,
        # log10 because the count overflows past ~16 chunks.
        "log10_virtual_experts": math.log10(virtual) if virtual > 0 else 0.0,
    }


def mlp_flops_multiplier(model, reference_active_mlp_params):
    """Per-token MLP arithmetic relative to vanilla GPT-2."""
    active = capacity_report(model)["mlp_active_params_per_token"]
    return active / reference_active_mlp_params


# --------------------------------------------------------------------------
# System performance
# --------------------------------------------------------------------------

@torch.no_grad()
def benchmark_generation(model, tokenizer, device, prompt="List three healthy snacks.",
                         gen_length=60, warmup=True):
    model.eval()
    model.config.use_cache = True
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    if warmup:
        model.generate(**inputs, max_new_tokens=8,
                       pad_token_id=tokenizer.eos_token_id)

    on_cuda = device.type == "cuda"
    if on_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)

    start = time.perf_counter()
    output = model.generate(**inputs, max_new_tokens=gen_length, do_sample=False,
                            pad_token_id=tokenizer.eos_token_id)
    if on_cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    model.config.use_cache = False
    generated = output.shape[1] - inputs.input_ids.shape[1]

    return {
        "tokens_generated": int(generated),
        "throughput_tok_s": generated / elapsed,
        "latency_ms_per_token": (elapsed / generated) * 1000,
        "peak_memory_mb": (
            torch.cuda.max_memory_allocated(device) / 1024 ** 2 if on_cuda else None
        ),
    }


@torch.no_grad()
def sample_generations(model, tokenizer, device, prompts, max_new_tokens=60,
                       temperature=0.6, top_k=40, top_p=0.9, seed=0):
    """Qualitative outputs, generated identically for every variant."""
    from .data import format_prompt

    model.eval()
    model.config.use_cache = True
    torch.manual_seed(seed)

    samples = []
    for instruction in prompts:
        inputs = tokenizer(format_prompt(instruction), return_tensors="pt").to(device)
        output = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=True,
            temperature=temperature, top_k=top_k, top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )
        text = tokenizer.decode(
            output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        ).strip()
        samples.append({"prompt": instruction, "response": text})

    model.config.use_cache = False
    return samples


@torch.no_grad()
def category_routing(model, tokenizer, device, prompt_categories, target_layer=0,
                     max_new_tokens=30):
    """Expert/option usage per prompt category for one layer.

    Extends the heatmap in the team's notebooks to work for AAG layers too,
    where the selection axis is chunk options rather than experts.
    """
    from .data import format_prompt

    mlp = model.transformer.h[target_layer].mlp
    if not isinstance(mlp, (GPT2MoELayer, AAGChunkMoELayer)):
        return None

    model.eval()
    model.config.use_cache = True
    n_options = num_options_of(mlp)
    table = {}

    for category, prompts in prompt_categories.items():
        totals = np.zeros(n_options)
        for instruction in prompts:
            inputs = tokenizer(format_prompt(instruction), return_tensors="pt").to(device)
            mlp.routing_log = []
            mlp.track_routing = True
            model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                           pad_token_id=tokenizer.eos_token_id)
            mlp.track_routing = False

            counts = _selection_counts(mlp.routing_log, n_options)
            if counts is not None:
                totals += counts

        table[category] = (100 * totals / totals.sum()).tolist() if totals.sum() else None

    model.config.use_cache = False
    return table
