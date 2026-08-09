# AAG on GPT-2 — standalone fine-tuning notebook

**Mohammad Al Dridi**

`Mohammad_GPT2_AAG_fine_tuning.ipynb` — self-contained, run top to bottom in
Colab. No package imports, nothing to install beyond what Colab ships.

## What it does

Replaces the MLP in each of GPT-2's 12 transformer blocks with a **chunked
expert bank**. Rather than a router selecting one of four whole experts, the
MLP's weight matrix is sliced into 8 horizontal bands and the router picks one
of 4 stored versions **per band, independently**. A token's effective weight
matrix is assembled from the winning bands.

| | Standard MoE | AAG (this) |
|---|---|---|
| Router decisions per token per layer | 1 | **8** |
| Routing configurations | 4 | **4⁸ = 65,536** per projection |
| Stored MLP parameters | 4× | 4× — the same |
| Arithmetic per token | 1× | **1× — the same** |

## Settings

Training deliberately matches Cole's ChunkMoE run so the two are comparable:
Alpaca, sequence length 512, batch 4 × grad-accum 8, 1 epoch, lr 5e-5, fp16.

Architecture: `NUM_CHUNKS = 8`, `NUM_OPTIONS = 4`, `intermediate_dim = 3072`
(GPT-2's real MLP width — required for the pretrained-slice initialisation).

Weights save to `MyDrive/gpt2-aag-alpaca`.

## Four things this fixes versus the earlier ChunkMoE

1. **Pretrained-slice initialisation.** Each band-option starts from the matching
   row-slice of GPT-2's own MLP, not from `randn`. Random init discards 221M
   pretrained parameters and tries to relearn language from one epoch of Alpaca.
   Small noise separates the options, or the router has no signal to tell them
   apart.

2. **Masked dispatch, not a per-token gather.** `chunk_weights[c, idx]` with a
   per-token index materialises `[N, chunk_dim, in_features]` — ~17 GiB at batch
   4 × seq 512. Looping over the 4 options and masking is identical arithmetic in
   `O(N × out_features)`.

3. **Gate normalisation.** Scaling a band by its router probability (~1/4 at
   init) shrinks the output to a quarter of the pretrained MLP's, undoing fix 1.
   Dividing by the detached probability leaves the forward pass unchanged while
   still passing a gradient to the router. Measured deviation from the pretrained
   MLP: **0.81 → 0.021**.

4. **Load-balancing auxiliary loss.** Without it the routers collapse onto one
   option per band and the combinatorial capacity is nominal only — the same
   expert-collapse failure the plain MoE hit.

Two further correctness fixes are baked in: the output buffer is allocated from
the computed result rather than from the input, so mixed precision does not trip
a dtype mismatch on the index-put; and the LM loss is delegated to
`Trainer.compute_loss` so gradient-accumulation normalisation is applied (calling
`model(**inputs)` directly makes the effective learning rate `GRAD_ACCUM`×
too large).

## Sanity checks while it runs

- **First logged loss ≈ 2.5–3.5**, `grad_norm` ≈ 1–5. Above 10 means something is
  wrong: `ln(50257) = 10.8` is what an *untrained* model scores.
- **Routing entropy ratio** near 1.0 at the end means every option stays in use;
  near 0.0 means the routers collapsed and the capacity is only on paper.

## Output

The notebook ends with perplexity on the held-out split, a per-layer routing
table, sample generations, and a summary block for the report.
