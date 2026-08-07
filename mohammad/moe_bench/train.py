"""Training loop with load-balancing auxiliary loss.

The auxiliary loss is the Switch-Transformer load-balancing term already used
in ``GPT2_MoE_aux.ipynb``, extended to AAG layers: there the balance is
enforced per chunk, across that chunk's options, and averaged over chunks.

Without it the routers collapse onto a single option and the combinatorial
capacity is nominal only -- the same expert-collapse failure the team hit with
the plain MoE.

Author: Mohammad Al Dridi
"""

import torch
from transformers import Trainer, TrainingArguments

from .layers import AAGChunkMoELayer, GPT2MoELayer


def _balance_term(probs, num_options):
    """``num_options * sum_o f_o * P_o`` over a [tokens, options] tensor."""
    hard = probs.argmax(dim=-1)
    fraction = torch.bincount(hard, minlength=num_options).float() / hard.numel()
    mean_prob = probs.mean(dim=0)
    return num_options * torch.sum(fraction.to(probs.device) * mean_prob)


def auxiliary_loss(model, attention_mask=None):
    """Sum of per-layer load-balancing terms, padding tokens excluded."""
    flat_mask = attention_mask.reshape(-1).bool() if attention_mask is not None else None
    total = 0.0

    for block in model.transformer.h:
        mlp = block.mlp

        if isinstance(mlp, GPT2MoELayer):
            probs = mlp.saved_router_probs
            if probs is None:
                continue
            if flat_mask is not None and flat_mask.shape[0] == probs.shape[0]:
                probs = probs[flat_mask]
            if probs.numel():
                total = total + mlp.aux_loss_coef * _balance_term(probs, mlp.num_experts)

        elif isinstance(mlp, AAGChunkMoELayer):
            for linear in (mlp.c_fc, mlp.c_proj):
                probs = linear.saved_router_probs      # [tokens, chunks, options]
                if probs is None:
                    continue
                if flat_mask is not None and flat_mask.shape[0] == probs.shape[0]:
                    probs = probs[flat_mask]
                if not probs.numel():
                    continue
                per_chunk = torch.stack([
                    _balance_term(probs[:, c, :], linear.num_options)
                    for c in range(linear.num_chunks)
                ])
                total = total + mlp.aux_loss_coef * per_chunk.mean()

    return total


class MoETrainer(Trainer):
    """Adds the auxiliary loss during training and records it for logging."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs.loss

        if model.training:
            aux = auxiliary_loss(model, inputs.get("attention_mask"))
            if torch.is_tensor(aux):
                self._last_aux = float(aux.detach())
                loss = loss + aux

        return (loss, outputs) if return_outputs else loss


def default_training_args(output_dir, epochs=1, batch_size=8, grad_accum=2,
                          learning_rate=5e-5, logging_steps=25, seed=42, fp16=None):
    """Identical hyperparameters for every variant, so the sweep is controlled."""
    if fp16 is None:
        fp16 = torch.cuda.is_available()

    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.03,          # stabilises the freshly initialised routers
        logging_steps=logging_steps,
        eval_strategy="no",         # evaluation is done once, by the benchmark
        save_strategy="no",
        fp16=fp16,
        report_to="none",
        seed=seed,
        dataloader_pin_memory=torch.cuda.is_available(),
    )


def train_variant(model, train_dataset, args, gradient_checkpointing=False):
    """Fine-tune one variant. Returns the training metrics dict."""
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    trainer = MoETrainer(model=model, args=args, train_dataset=train_dataset)
    model.train()
    result = trainer.train()

    if gradient_checkpointing:
        model.gradient_checkpointing_disable()

    return {
        "train_runtime_s": result.metrics.get("train_runtime"),
        "train_loss": result.metrics.get("train_loss"),
        "train_samples_per_s": result.metrics.get("train_samples_per_second"),
    }
