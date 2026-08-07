# Unified benchmark + repaired AAG

**Author: Mohammad Al Dridi**

Everything in this directory is mine. It builds on Tamara's MoE work and Cole's
AAG idea, and does not modify either of their notebooks.

Two things live here:

1. **One benchmark** that puts every model variant on the same frozen data
   split with the same hyperparameters, so the report has one comparable
   results table instead of per-notebook numbers.
2. **A working AAG layer** — Cole's chunked virtual-expert idea, with the
   three defects that were stopping it from training fixed.

---

## Why AAG is worth measuring

| | Stored MLP params | Active per token | Distinct routings |
|---|---|---|---|
| Vanilla GPT-2 | 56.7M | 56.7M | 1 |
| MoE top-1, 4 experts | 226.7M | 56.7M | 4<sup>12</sup> ≈ 1.7e7 |
| MoE soft routing | 226.7M | **226.7M** | 4<sup>12</sup> ≈ 1.7e7 |
| **AAG, 8 chunks** | 226.7M | **56.7M** | **4<sup>192</sup> ≈ 1e115** |

Verified by `metrics.capacity_report`, not estimated. AAG's active parameter
count is *exactly* vanilla GPT-2's 4,722,432 per block regardless of chunk
count, because one option is selected per chunk and the chunks partition the
output dimension. Turning up `num_chunks` grows the routing space
exponentially and the arithmetic not at all.

Soft routing is the interesting contrast: it runs all four experts on every
token, so it costs 4x vanilla GPT-2 in MLP arithmetic and buys the same four
choices a top-1 router gets for 1x. That column does not exist in the current
notebooks, and it is the clearest argument in the report for why chunked
routing is the right direction.

---

## What was fixed in the AAG layer

All three are in `Cole_GPT2_AMoE_fine_tuning.ipynb`; see `layers.py` for the
implementations.

| Defect | Effect | Fix |
|---|---|---|
| `num_chunks=1` at the call site | 4 virtual experts, i.e. a plain MoE — the combinatorics were switched off | Chunk count is a swept parameter; 3072 and 768 are both divisible by 1/2/4/8/16 |
| `chunk_weights` initialised from `randn` | 221M random parameters replace 56.7M pretrained ones; GPT-2's language knowledge discarded | Each chunk-option starts from the matching row-slice of the pretrained MLP, plus small noise to break symmetry between options |
| Per-token weight gather | `[N, chunk_dim, in_features]` = 17.5 GiB at batch 4 x seq 512, x24 per forward pass | Loop over the 4 options and mask; identical arithmetic in `O(N * out_features)` |

A fourth issue surfaced while testing the fix. Scaling each chunk by its
router probability (~1/4 at initialisation) shrinks the layer's output to a
quarter of the pretrained MLP's, which throws away the head start the slice
initialisation just bought. `preserve_init=True` divides by the detached
probability: the forward pass is numerically unchanged, the router still
receives a gradient. Measured deviation from the pretrained MLP drops from
**0.81 to 0.021** — and the residual 0.021 is just the symmetry-breaking noise.

---

## Layout

```
moe_bench/
  layers.py     GPT2MoELayer (Tamara's, both routing modes) + AAGChunkMoELayer
  builders.py   variant registry; every variant is defined in exactly one place
  data.py       the frozen Alpaca split; seed, template and truncation fixed here
  metrics.py    perplexity, routing health, capacity accounting, throughput
  train.py      Trainer subclass with the load-balancing auxiliary loss
  runner.py     sweep driver, writes results.json
  report.py     results.md / results.tex / the figures
notebooks/
  Mohammad_Benchmark_Colab.ipynb    Colab driver
```

`GPT2MoELayer` keeps Tamara's parameter names, so her saved checkpoints load
into it directly and can be scored on the same split as everything else.

---

## Running it

### 1. Verify the plumbing (any machine, ~2 minutes)

```bash
pip install -r requirements.txt
python -m moe_bench.runner --preset smoke --variants core --tiny --out results_smoke
```

`--tiny` swaps in a 2-layer random backbone. The numbers are meaningless; this
only proves the pipeline runs before spending GPU time. Full GPT-2 AAG is
~295M parameters and needs several GB once gradients exist, which is more than
a typical laptop has spare on CPU.

### 2. The real sweep (GPU)

```bash
python -m moe_bench.runner --preset budget --variants core    --out results/core
python -m moe_bench.runner --preset budget --variants scaling --out results/scaling
```

`budget` is 8000 train / 1000 eval examples at sequence length 256 — roughly
20-30 minutes per variant on a T4. Every variant sees byte-identical data.
`results.json` is rewritten after each variant, so an interrupted run keeps
whatever finished.

Add `--gradient-checkpointing` if memory is tight, and
`--optim adamw_bnb_8bit` (via `TrainingArguments`) on a 6 GB card.

### 3. Tables and figures

```bash
python -m moe_bench.report results/core/results.json
```

Writes `results.md`, `results.tex` (paste straight into the Prism report), and
`fig_perplexity.png`, `fig_capacity.png`, `fig_scaling.png`, `fig_routing.png`.

---

## Reading the metrics

- **Perplexity** is token-weighted over supervised (label != -100) tokens, so
  it measures response quality only, not the prompt. This differs from the
  notebooks, which average per-batch losses — that over-weights short batches.
- **Routing entropy ratio** is 1.0 when every expert is used equally and 0.0
  when the router has collapsed onto one. CV is the same story inverted.
  Definitions match Tamara's so the numbers stay comparable with hers.
- **Active params/token** counts only what a single token actually touches.
  This is the sparse-vs-dense distinction and drives the `Compute` column.
- **Virtual experts** is the product over layers of each layer's routing
  choices, reported as log10 because it overflows int64 past ~16 chunks.
