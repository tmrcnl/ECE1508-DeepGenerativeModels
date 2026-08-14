# ECE1508 — Routing architectures on GPT-2

Group project, Deep Generative Models, Summer 2026.

We replace the feed-forward sub-layer of every GPT-2 block with a routed layer
and compare four designs on instruction-following data. The question is whether
the *granularity* of the routing decision can be decoupled from the *number of
stored experts* — that is, whether a model can reach many more distinct
computations without storing or executing more.

---

## The models

| # | Notebook | Model | Trained on |
|---|---|---|---|
| 01 | `notebooks/01_moe_top1.ipynb` | Mixture of Experts, top-1 routing, 4 experts | alpaca-cleaned |
| 02 | `notebooks/02_moe_soft_routing.ipynb` | Soft routing — all experts weighted and summed (NAM-inspired) | alpaca |
| 03 | `notebooks/03_moe_conditioned.ipynb` | Top-1 MoE with a task-category-conditioned router | alpaca-cleaned |
| 04 | `notebooks/04_aag.ipynb` | AAG — chunked expert bank with pretrained-slice initialisation, fused dispatch, per-chunk load balancing | alpaca |
| 05 | `notebooks/05_benchmark.ipynb` | **Evaluation harness** — scores every model with identical code on identical data | — |

Notebooks 01–04 each train and save one model. Notebook 05 trains nothing; it
loads the saved weights and evaluates them together, alongside an untuned GPT-2
as a reference line.

---

## Setup

### Google Colab (recommended)

Everything was developed and run on Colab. No installation is required — the
runtime already provides every dependency in `requirements.txt`.

1. Open a notebook in Colab.
2. **Runtime → Change runtime type → L4 GPU.** A T4 works but is slower; an A100
   is not worth the compute units for this workload.
3. **Runtime → enable background execution** for the training notebooks, so a run
   survives the browser tab closing.

### Local

Requires a CUDA GPU with at least 16 GB of memory for training. The notebooks
call `google.colab.drive`, so those cells must be removed or replaced with local
paths.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Model weights

Trained weights are **not** stored in this repository — each is roughly 1.2 GB,
well past GitHub's limits. They live in Google Drive.

The notebooks expect this layout:

```
MyDrive/
├── ece1508_checkpoints/
│   ├── gpt2-moe-alpacaclean-aux/          # 01  MoE top-1
│   │   └── model.safetensors
│   ├── gpt2-moe-alpaca-aux-nam/           # 02  soft routing
│   │   └── model.safetensors
│   ├── gpt2-moe-alpacaclean-aux-cond/     # 03  conditioned
│   │   └── model.safetensors
│   └── gp2-aag-alpaca/                    # 04  AAG
│       └── model.safetensors
└── ece1508_results/                       # written by the notebooks
```

Only `model.safetensors` is needed from each folder. `optimizer.pt` is a further
2.3 GB and is not used for evaluation.

<!-- Add the shared Drive link here before submission. -->
**Shared Drive folder:** [_(link to be added)_](https://drive.google.com/drive/folders/169nT0wSNAhW7nODAPabC1RqzyYau7XzN?usp=sharing)

If a checkpoint is missing, notebook 05 reports it by name and stops before
consuming GPU time.

---

## Usage

### Reproducing a single model

Open the relevant notebook (01–04), set the runtime to L4, and run all cells.
Each notebook mounts Drive, loads GPT-2, replaces the feed-forward layers, tunes
on the instruction dataset, and saves the result to Drive.

Approximate cost of one epoch on an L4:

| Notebook | Time |
|---|---|
| 01, 03 — top-1 MoE | ~1.5 h |
| 02 — soft routing | ~3 h (runs every expert on every token) |
| 04 — AAG | ~1.3 h |

Training hyperparameters are identical across all models — sequence length 512,
batch 4 with gradient accumulation 8, one epoch, learning rate 5e-5, seed 42 — so
that differences in the results are attributable to the architecture rather than
the recipe.

### Running the benchmark

Open `notebooks/05_benchmark.ipynb` and run all cells (~20–30 minutes, no
training). It rebuilds every architecture from source, loads each saved
checkpoint, and scores them all with the same code on the same inputs.

Two evaluation sets are used, deliberately:

- **alpaca-cleaned held-out** — in-domain quality. Note this overlaps the training
  data of the models trained on plain Alpaca, because the two corpora differ in
  size (51,760 vs 52,002 rows) and so a seeded split yields different held-out
  rows.
- **dolly-15k held-out** — no model trained on this, so it is the neutral
  comparison. **Prefer this column when comparing models trained on different
  corpora.**

The loader aborts if any checkpoint fails to match its architecture, rather than
silently scoring a partly-random model.

---

## Results

Pre-computed outputs from the reported run are in `results/`:

| File | Contents |
|---|---|
| `results.json` | Full record — metrics, routing statistics, generated samples |
| `results.csv` | The main comparison table |
| `results.tex` | The same table, formatted for the report |
| `generations.md` | Side-by-side model outputs on fixed prompts |
| `fig_perplexity.png` | Quality on both evaluation sets |
| `fig_capacity.png` | Reachable configurations against per-token compute |
| `fig_routing.png` | Per-layer routing entropy |
| `fig_specialisation.png` | Expert usage by task category |

### How to read the metrics

- **Perplexity** — `exp(mean negative log-likelihood)`, computed over response
  tokens only with the prompt masked out. Lower is better.
- **Active parameters per token** — what one token actually passes through, as
  opposed to what the model stores. This is what separates a sparse model from a
  dense one.
- **Routing entropy (H/H_max)** — 1.0 means every expert is used equally, 0.0
  means the router collapsed onto one and the rest are dead weight.
- **Specialisation** — mutual information in bits between a prompt's task
  category and the router's choice. 0 means routing is independent of the task.

---

## Repository layout

```
├── notebooks/     the five notebooks behind the report
├── results/       metrics, tables and figures from the reported run
├── archive/       superseded and exploratory work, kept for reference
├── requirements.txt
```

`archive/` holds early GPT-2 exploration, the pre-auxiliary-loss MoE, the
per-notebook evaluation that the unified benchmark replaced, runs on alternative
datasets (dolly-15k, infinity-instruct), byte-level prototypes, an incomplete
chunked-MoE implementation, and an earlier package-based benchmark framework.
Nothing has been deleted — the dead ends and earlier iterations are part of the
record.
