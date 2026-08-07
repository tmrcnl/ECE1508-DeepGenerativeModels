"""Sweep driver: train and evaluate every variant under identical conditions.

Usage
-----
Smoke test the plumbing on CPU (a couple of minutes, meaningless numbers)::

    python -m moe_bench.runner --preset smoke --variants smoke

The real sweep, on a GPU::

    python -m moe_bench.runner --preset budget --variants core --out results/

Results are appended to ``<out>/results.json`` one variant at a time, so a
crashed or pre-empted run keeps everything completed so far.

Author: Mohammad Al Dridi
"""

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from . import builders, data, metrics, train


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def environment():
    info = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "cuda": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def run_variant(name, tokenizer, train_ds, eval_ds, device, args_factory,
                do_train=True, checkpoint=None, gradient_checkpointing=False):
    """Build, optionally train, then evaluate a single variant."""
    variant = builders.VARIANTS[name]
    print(f"\n{'=' * 70}\n{variant.label}  [{name}]\n{'=' * 70}", flush=True)

    started = time.perf_counter()
    model = variant.build()

    if checkpoint:
        load_info = builders.load_checkpoint(model, checkpoint, device=str(device))
        print(f"  loaded checkpoint: {len(load_info['missing'])} missing, "
              f"{len(load_info['unexpected'])} unexpected keys")

    capacity = metrics.capacity_report(model)
    print(f"  stored MLP params : {capacity['mlp_stored_params']:,}")
    print(f"  active per token  : {capacity['mlp_active_params_per_token']:,}")
    print(f"  virtual experts   : 1e{capacity['log10_virtual_experts']:.1f}")

    model.to(device)

    train_metrics = {}
    if do_train:
        train_metrics = train.train_variant(
            model,
            train_ds,
            args_factory(name),
            gradient_checkpointing=gradient_checkpointing,
        )
        print(f"  train loss: {train_metrics.get('train_loss')}")

    quality = metrics.evaluate_perplexity(model, eval_ds, device)
    health = metrics.routing_health(quality.pop("_routing_counts"))
    speed = metrics.benchmark_generation(model, tokenizer, device)
    samples = metrics.sample_generations(
        model, tokenizer, device,
        ["What is the capital of Canada?",
         "List three healthy snacks.",
         "Correct the grammar: He do not have no money."],
    )
    heatmap = metrics.category_routing(
        model, tokenizer, device, data.PROBE_PROMPTS, target_layer=0
    )

    def _fmt(value):
        return "n/a" if value is None else f"{value:.4f}"

    print(f"  perplexity: {quality['perplexity']:.3f}   "
          f"mean CV: {_fmt(health['mean_cv'])}   "
          f"entropy ratio: {_fmt(health['entropy_ratio'])}")

    record = {
        "variant": name,
        "label": variant.label,
        "family": variant.family,
        "notes": variant.notes,
        "meta": variant.meta,
        "capacity": capacity,
        "quality": quality,
        "routing": health,
        "speed": speed,
        "train": train_metrics,
        "samples": samples,
        "category_routing": heatmap,
        "wall_clock_s": time.perf_counter() - started,
    }

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default="budget", choices=list(data.PRESETS),
                        help="data/compute budget")
    parser.add_argument("--variants", default="core",
                        help="a name from VARIANT_SETS, or a comma-separated list")
    parser.add_argument("--out", default="results", help="output directory")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--no-train", action="store_true",
                        help="evaluate without fine-tuning")
    parser.add_argument("--gradient-checkpointing", action="store_true",
                        help="trade speed for memory on small GPUs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tiny", action="store_true",
                        help="2-layer random backbone; verifies the pipeline "
                             "on a laptop, produces meaningless numbers")
    opts = parser.parse_args(argv)

    if opts.tiny:
        builders.set_tiny(True)
        print("TINY MODE: 2-layer random backbone, results are not meaningful")

    if opts.variants in builders.VARIANT_SETS:
        names = builders.VARIANT_SETS[opts.variants]
    else:
        names = [n.strip() for n in opts.variants.split(",") if n.strip()]

    out_dir = Path(opts.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.json"

    device = _device()
    config = data.PRESETS[opts.preset]
    print(f"device={device}  preset={opts.preset}  variants={names}")

    tokenizer = builders.load_tokenizer()
    train_ds, eval_ds = data.build_splits(tokenizer, config)
    print(f"train={len(train_ds)}  eval={len(eval_ds)}  max_length={config.max_length}")

    def args_factory(name):
        return train.default_training_args(
            output_dir=str(out_dir / "checkpoints" / name),
            epochs=opts.epochs,
            batch_size=opts.batch_size,
            grad_accum=opts.grad_accum,
            learning_rate=opts.lr,
            seed=opts.seed,
        )

    payload = {
        "environment": environment(),
        "config": {
            "preset": opts.preset,
            "max_length": config.max_length,
            "n_train": len(train_ds),
            "n_eval": len(eval_ds),
            "epochs": opts.epochs,
            "batch_size": opts.batch_size,
            "grad_accum": opts.grad_accum,
            "learning_rate": opts.lr,
            "seed": opts.seed,
            "trained": not opts.no_train,
        },
        "results": [],
    }

    for name in names:
        try:
            record = run_variant(
                name, tokenizer, train_ds, eval_ds, device, args_factory,
                do_train=not opts.no_train,
                gradient_checkpointing=opts.gradient_checkpointing,
            )
            payload["results"].append(record)
        except torch.cuda.OutOfMemoryError as exc:
            print(f"  OOM on {name}: {exc}")
            payload["results"].append({"variant": name, "error": "OOM"})
            torch.cuda.empty_cache()

        # Written after every variant so a lost session keeps its progress.
        results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"  -> {results_path}")

    print(f"\nDone. {len(payload['results'])} variants written to {results_path}")
    return payload


if __name__ == "__main__":
    main()
