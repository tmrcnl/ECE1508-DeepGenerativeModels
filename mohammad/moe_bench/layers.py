"""MoE / AAG layer implementations used by the benchmark.

Two families live here:

``GPT2MoELayer``
    A faithful re-implementation of Tamara's MoE layer (top-1 and soft
    routing), with parameter names matching hers so her saved checkpoints
    load without renaming.

``AAGChunkMoELayer``
    Cole's Autonomous Architecture Generation layer, with three fixes:
    pretrained-slice initialisation, memory-safe routing, and the chunk
    count actually turned up past 1.  See ``mohammad/README.md``.

Author: Mohammad Al Dridi
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN


# --------------------------------------------------------------------------
# Tamara's MoE layer (top-1 and soft routing)
# --------------------------------------------------------------------------

class GPT2MoELayer(nn.Module):
    """Replaces a GPT-2 block's MLP with a router plus ``num_experts`` clones.

    Each expert is a deepcopy of the pretrained MLP, so all experts start from
    GPT-2's learned weights.  ``routing`` selects between the two variants in
    the team's notebooks:

    ``"top1"``
        Dispatch each token to its highest-probability expert only.
        Active compute is one expert per token.

    ``"soft"``
        Run every expert on every token and sum the outputs weighted by the
        router probabilities.  This is what ``GPT2_MoE_aux_NAM.ipynb`` does.
        Note that it is *dense*: active compute is ``num_experts`` experts per
        token, so it costs 4x a vanilla MLP.
    """

    def __init__(self, original_mlp, num_experts=4, routing="top1", aux_loss_coef=0.05):
        super().__init__()
        if routing not in ("top1", "soft"):
            raise ValueError(f"routing must be 'top1' or 'soft', got {routing!r}")

        self.num_experts = num_experts
        self.routing = routing
        self.aux_loss_coef = aux_loss_coef

        hidden_dim = original_mlp.c_fc.weight.shape[0]
        self.router = nn.Linear(hidden_dim, num_experts)
        self.experts = nn.ModuleList(
            [copy.deepcopy(original_mlp) for _ in range(num_experts)]
        )

        # Telemetry, off by default so training is unaffected.
        self.routing_log = []
        self.track_routing = False
        self.saved_router_probs = None

    def forward(self, hidden_states, *args, **kwargs):
        orig_shape = hidden_states.shape
        flat = hidden_states.view(-1, orig_shape[-1])

        router_probs = torch.softmax(self.router(flat), dim=-1)
        if self.training:
            self.saved_router_probs = router_probs

        top1_indices = router_probs.argmax(dim=-1)
        if self.track_routing:
            # One [N] tensor per forward pass. AAG logs [N, num_chunks], so
            # metrics.py treats both as "one selection axis per token".
            self.routing_log.append(top1_indices.detach().cpu())

        if self.routing == "soft":
            out = torch.zeros_like(flat)
            for e in range(self.num_experts):
                out += self.experts[e](flat) * router_probs[:, e].unsqueeze(-1)
        else:
            top1_weights = router_probs.gather(1, top1_indices.unsqueeze(-1))
            out = None
            for e in range(self.num_experts):
                mask = top1_indices == e
                if not mask.any():
                    continue
                expert_out = self.experts[e](flat[mask]) * top1_weights[mask]
                if out is None:
                    # Under autocast the expert returns half/bfloat16 while
                    # `flat` is still float, and an index-put needs both sides
                    # to share a dtype -- so allocate from the result.
                    out = torch.zeros(
                        flat.shape, dtype=expert_out.dtype, device=expert_out.device
                    )
                out[mask] = expert_out
            if out is None:
                out = torch.zeros_like(flat)

        return out.view(orig_shape)

    # -- capacity accounting -------------------------------------------------

    def stored_params(self):
        return sum(p.numel() for p in self.experts.parameters())

    def active_params_per_token(self):
        """Expert parameters touched for a single token."""
        per_expert = sum(p.numel() for p in self.experts[0].parameters())
        return per_expert * (self.num_experts if self.routing == "soft" else 1)

    def virtual_experts(self):
        return self.num_experts


# --------------------------------------------------------------------------
# Cole's AAG layer, repaired
# --------------------------------------------------------------------------

class ChunkLinearMoE(nn.Module):
    """One Linear split into ``num_chunks`` output chunks.

    Each chunk carries ``num_options`` candidate sub-matrices and the router
    picks one option per chunk, per token.  A token's effective weight matrix
    is therefore assembled from ``num_chunks`` independently chosen pieces,
    giving ``num_options ** num_chunks`` distinct virtual matrices while only
    ever *executing* one full matrix worth of arithmetic.

    Three differences from the original implementation:

    1.  **Pretrained initialisation.**  Given ``pretrained_weight``, option
        ``o`` of chunk ``c`` starts from the matching row-slice of GPT-2's
        MLP rather than ``randn``.  Options are separated by a small amount of
        noise so the router has a gradient signal to distinguish them.

    2.  **Memory-safe dispatch.**  The original gathered a per-token weight
        matrix (``chunk_weights[c, c_indices]``), materialising an
        ``[N, chunk_dim, in_features]`` tensor -- 17.5 GiB at batch 4 x seq
        512.  Looping over the ``num_options`` options and masking is the same
        arithmetic in ``O(N * out_features)`` memory.

    3.  **Router probabilities are retained** for the load-balancing auxiliary
        loss, matching the treatment of expert collapse in the team's MoE.
    """

    # "fused" computes every option in one matmul and gathers the winner;
    # "masked" loops over options and masks. Numerically identical -- verified
    # to float32 epsilon on forward, all gradients, and gradient sparsity.
    # Which is faster is hardware-dependent, so measure before choosing:
    # fused wins where kernel dispatch dominates (GPU), masked wins where raw
    # arithmetic dominates (CPU), because fused does num_options times the work.
    DISPATCH = "fused"

    def __init__(self, in_features, out_features, num_chunks=8, num_options=4,
                 pretrained_weight=None, pretrained_bias=None, init_noise=0.02,
                 preserve_init=True, dispatch=None):
        super().__init__()
        self.dispatch = dispatch or self.DISPATCH
        if out_features % num_chunks != 0:
            raise ValueError(
                f"out_features ({out_features}) must be divisible by "
                f"num_chunks ({num_chunks})"
            )

        self.in_features = in_features
        self.out_features = out_features
        self.num_chunks = num_chunks
        self.num_options = num_options
        self.preserve_init = preserve_init
        self.chunk_dim = out_features // num_chunks

        self.router = nn.Linear(in_features, num_chunks * num_options)
        self.chunk_weights = nn.Parameter(
            torch.empty(num_chunks, num_options, self.chunk_dim, in_features)
        )
        self.chunk_biases = nn.Parameter(
            torch.zeros(num_chunks, num_options, self.chunk_dim)
        )
        self._init_weights(pretrained_weight, pretrained_bias, init_noise)

        self.saved_router_probs = None
        self.routing_log = []
        self.track_routing = False

    def _init_weights(self, weight, bias, noise):
        if weight is None:
            nn.init.normal_(self.chunk_weights, std=0.02)
            return

        # GPT-2 uses Conv1D, whose weight is stored [in_features, out_features].
        w = weight.t().contiguous()                      # -> [out, in]
        if w.shape != (self.out_features, self.in_features):
            raise ValueError(
                f"pretrained weight {tuple(w.shape)} does not match layer "
                f"({self.out_features}, {self.in_features})"
            )

        with torch.no_grad():
            scale = w.std() * noise
            for c in range(self.num_chunks):
                lo, hi = c * self.chunk_dim, (c + 1) * self.chunk_dim
                for o in range(self.num_options):
                    self.chunk_weights[c, o].copy_(w[lo:hi])
                    # Break the symmetry between options; identical options
                    # give the router no reason to prefer one over another.
                    self.chunk_weights[c, o].add_(
                        torch.randn_like(self.chunk_weights[c, o]) * scale
                    )
                    if bias is not None:
                        self.chunk_biases[c, o].copy_(bias[lo:hi])

    def forward(self, x):
        n_tokens = x.shape[0]

        router_logits = self.router(x).view(n_tokens, self.num_chunks, self.num_options)
        router_probs = F.softmax(router_logits, dim=-1)
        if self.training:
            self.saved_router_probs = router_probs

        top_weights, top_indices = router_probs.max(dim=-1)   # [N, num_chunks]
        if self.track_routing:
            self.routing_log.append(top_indices.detach().cpu())

        if self.dispatch == "fused":
            out = self._dispatch_fused(x, top_indices, n_tokens)
        else:
            out = self._dispatch_masked(x, top_indices, n_tokens)

        gate = top_weights.unsqueeze(-1)
        if self.preserve_init:
            # Scaling a chunk by its router probability (~1/num_options at
            # initialisation) would shrink the output to a quarter of the
            # pretrained MLP's, throwing away the head start the slice
            # initialisation just bought. Dividing by the detached value leaves
            # the forward pass numerically untouched while still routing a
            # gradient back to the router.
            gate = gate / gate.detach().clamp_min(1e-9)

        return (out * gate).reshape(n_tokens, self.out_features)

    def _dispatch_fused(self, x, top_indices, n_tokens):
        """Every (chunk, option) pair in one matmul, then gather the winner.

        Does num_options times the arithmetic of what is strictly needed, but in
        a single large kernel with no host synchronisation. Both the extra
        arithmetic and the extra activation memory are out_features *
        num_options, independent of num_chunks -- so a 16-chunk model costs
        exactly what a 1-chunk model does.
        """
        all_options = F.linear(
            x,
            self.chunk_weights.reshape(-1, self.in_features),
            self.chunk_biases.reshape(-1),
        ).view(n_tokens, self.num_chunks, self.num_options, self.chunk_dim)

        selection = top_indices[:, :, None, None].expand(
            n_tokens, self.num_chunks, 1, self.chunk_dim
        )
        return all_options.gather(2, selection).squeeze(2)

    def _dispatch_masked(self, x, top_indices, n_tokens):
        """Only the arithmetic that is needed, at the cost of many small kernels.

        num_chunks * num_options matmuls per layer, each preceded by a
        `mask.any()` that synchronises the device against Python.
        """
        chunks = []
        for c in range(self.num_chunks):
            idx_c = top_indices[:, c]
            out_c = None
            for o in range(self.num_options):
                mask = idx_c == o
                if not mask.any():
                    continue
                option_out = F.linear(
                    x[mask], self.chunk_weights[c, o], self.chunk_biases[c, o]
                )
                if out_c is None:
                    # Allocated from the result, not from x: under autocast
                    # F.linear returns half while x is float, and an index-put
                    # requires both sides to share a dtype.
                    out_c = torch.zeros(
                        n_tokens, self.chunk_dim,
                        dtype=option_out.dtype, device=option_out.device,
                    )
                out_c[mask] = option_out
            if out_c is None:
                out_c = x.new_zeros(n_tokens, self.chunk_dim)
            chunks.append(out_c)

        return torch.stack(chunks, dim=1)


class AAGChunkMoELayer(nn.Module):
    """Drop-in replacement for a GPT-2 block MLP built from two chunked Linears."""

    def __init__(self, original_mlp=None, hidden_dim=768, intermediate_dim=3072,
                 num_chunks=8, num_options=4, aux_loss_coef=0.05,
                 activation="gelu_new", preserve_init=True):
        super().__init__()
        self.num_chunks = num_chunks
        self.num_options = num_options
        self.aux_loss_coef = aux_loss_coef

        w_fc = b_fc = w_proj = b_proj = None
        if original_mlp is not None:
            w_fc, b_fc = original_mlp.c_fc.weight, original_mlp.c_fc.bias
            w_proj, b_proj = original_mlp.c_proj.weight, original_mlp.c_proj.bias
            hidden_dim = w_fc.shape[0]
            intermediate_dim = w_fc.shape[1]

        self.c_fc = ChunkLinearMoE(
            hidden_dim, intermediate_dim, num_chunks, num_options, w_fc, b_fc,
            preserve_init=preserve_init,
        )
        self.act = ACT2FN[activation]
        self.c_proj = ChunkLinearMoE(
            intermediate_dim, hidden_dim, num_chunks, num_options, w_proj, b_proj,
            preserve_init=preserve_init,
        )

    # Telemetry flags are proxied so callers treat every layer type alike.
    #
    # Only the up-projection is tracked. It is the direct analogue of "which
    # expert handled this token" in a standard MoE, and tracking one projection
    # keeps the log one row per token, which is what lets the padding mask line
    # up. Logging both would interleave two selections per token.
    @property
    def track_routing(self):
        return self.c_fc.track_routing

    @track_routing.setter
    def track_routing(self, value):
        self.c_fc.track_routing = value

    @property
    def routing_log(self):
        return self.c_fc.routing_log

    @routing_log.setter
    def routing_log(self, value):
        self.c_fc.routing_log = value
        self.c_proj.routing_log = []

    def forward(self, hidden_states, *args, **kwargs):
        orig_shape = hidden_states.shape
        flat = hidden_states.view(-1, orig_shape[-1])
        out = self.c_proj(self.act(self.c_fc(flat)))
        return out.view(orig_shape)

    # -- capacity accounting -------------------------------------------------

    def stored_params(self):
        return (
            self.c_fc.chunk_weights.numel() + self.c_fc.chunk_biases.numel()
            + self.c_proj.chunk_weights.numel() + self.c_proj.chunk_biases.numel()
        )

    def active_params_per_token(self):
        """One chunk-option per chunk means exactly one full matrix per Linear."""
        return (
            self.c_fc.out_features * self.c_fc.in_features + self.c_fc.out_features
            + self.c_proj.out_features * self.c_proj.in_features + self.c_proj.out_features
        )

    def virtual_experts(self):
        """Distinct weight matrices this layer can assemble, as a float.

        ``num_options ** num_chunks`` for each of the two Linears.  This
        overflows int64 past roughly num_chunks=16, hence float.
        """
        per_linear = float(self.num_options) ** self.num_chunks
        return per_linear ** 2
