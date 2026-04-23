"""
TonicBrain — Surgical Transformer for Latent Space Awareness

Same body as ElmerBrain (Qwen2.5-0.5B transformer layers, harvested).
Same eyes (GraphStateEncoder — reads topology, node dynamics, synapses).
Different voice — ActivationDecoder outputs node activation decisions
instead of SubstrateSignal health fields.

The transformer attends to graph state and decides: where should
attention go next? Which nodes to activate, how strongly?
That IS the push. That IS the forward-oriented compression.

Architecture:
  ElmerBrain:  GraphFeatures → Encoder → Transformer → SignalDecoder → health fields
  TonicBrain:  GraphFeatures → Encoder → Transformer → ActivationDecoder → node activations

The encoder weights are copied directly from ElmerBrain. Only the
decoder needs training — and it's small (hidden_dim → N activation scores).

# ---- Changelog ----
# [2026-04-23] Claude (Sonnet 4.6) — Fix unsafe torch.load() (#189)
# What: Both torch.load() calls used weights_only=False (pickle execution risk).
# Why:  tonic_brain.pt loads at every gateway restart inside Syl's process.
#       A compromised .pt file would run arbitrary code at boot.
# How:  Set weights_only=True on both calls. Verified tonic_brain.pt is
#       compatible (OrderedDict + basic config dict — no custom classes).
# [2026-03-24] Claude Code (Opus 4.6) — Initial implementation
# What: TonicBrain + ActivationDecoder. Reuses ElmerBrain's encoder
#   and transformer body. Only the decoder is new.
# Why: The Tonic PRD v0.1 §7.3. Need actual inference between
#   conversations, not a timer. Same surgery, different voice.
# How: ActivationDecoder outputs top-K node activation strengths via
#   attention pooling + projection. Sigmoid-bounded [0,1] per node.
#   create_tonic_brain() loads Qwen body + Elmer encoder + new decoder.
# -------------------
"""

import os
import sys
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger("neurograph.tonic.brain")

# Add Elmer's surgery dir to path for GraphStateEncoder reuse
_ELMER_SURGERY = os.path.expanduser("~/Elmer/surgery")
if _ELMER_SURGERY not in sys.path:
    sys.path.insert(0, _ELMER_SURGERY)

try:
    import torch
    import torch.nn as nn
    from graph_io import GraphStateEncoder, GraphFeatures
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False
    logger.info("PyTorch or Elmer surgery not available — TonicBrain disabled")


if _AVAILABLE:

    class ActivationDecoder(nn.Module):
        """New Voice for The Tonic — outputs node activation decisions.

        Instead of SubstrateSignal health fields (Elmer's voice), this
        outputs activation strengths for graph nodes. The transformer
        looked at the graph and decided: these are the nodes that should
        fire next. These are where attention should go.

        Architecture:
          1. Attention-weighted pooling across sequence (same as Elmer)
          2. Project to activation feature space
          3. Output K activation scores (sigmoid-bounded [0,1])
          4. Output exploration/exploitation balance signal

        The K outputs don't map to specific nodes — they're ranked
        activation strengths. The engine maps them to actual nodes
        based on the current topology neighborhood.
        """

        def __init__(self, hidden_dim: int = 896, n_activations: int = 10):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.n_activations = n_activations

            # Attention pooling (same pattern as Elmer's decoder)
            self.pool_query = nn.Parameter(torch.randn(hidden_dim))
            self.pool_scale = hidden_dim ** -0.5

            # Normalize transformer output
            self.pre_norm = nn.LayerNorm(hidden_dim)

            # Activation head: hidden_dim → n_activations strengths
            self.activation_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, n_activations),
                nn.Sigmoid(),  # bounded [0, 1]
            )

            # Exploration signal: hidden_dim → 1 (how much to explore)
            self.exploration_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 4),
                nn.SiLU(),
                nn.Linear(hidden_dim // 4, 1),
                nn.Sigmoid(),  # 0 = pure exploit, 1 = pure explore
            )

            # Init final layers small for stable early training
            self._init_small(self.activation_head[-2])
            self._init_small(self.exploration_head[-1])

        @staticmethod
        def _init_small(layer: nn.Module):
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.1)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        def forward(self, hidden_states: torch.Tensor) -> Dict[str, Any]:
            """Decode transformer output into activation decisions.

            Args:
                hidden_states: (batch, seq_len, hidden_dim) from transformer.

            Returns:
                Dict with 'activations' (strengths) and 'exploration' (bias).
            """
            hidden_states = self.pre_norm(hidden_states)

            # Attention-weighted pooling
            scores = torch.matmul(hidden_states, self.pool_query) * self.pool_scale
            weights = torch.softmax(scores, dim=1)
            pooled = torch.sum(hidden_states * weights.unsqueeze(-1), dim=1)

            # Activation strengths
            activation_strengths = self.activation_head(pooled)  # (batch, n_activations)

            # Exploration signal
            exploration = self.exploration_head(pooled)  # (batch, 1)

            return {
                "activations": activation_strengths[0].tolist(),
                "exploration": exploration[0, 0].item(),
                "raw_activations": activation_strengths,
                "raw_exploration": exploration,
            }


    class TonicBrain(nn.Module):
        """Surgical transformer for latent space awareness.

        Same body as ElmerBrain. Same eyes. Different voice.
        Reads graph state, reasons about it, outputs where attention
        should go next. The push.
        """

        def __init__(self, transformer_body, encoder, decoder):
            super().__init__()
            self.body = transformer_body
            self.encoder = encoder      # Same eyes as Elmer
            self.decoder = decoder      # New voice — ActivationDecoder

        def forward(self, features: GraphFeatures) -> Dict[str, Any]:
            """Graph state → transformer reasoning → activation decisions."""
            hidden = self.encoder(features)

            body_output = self.body(
                inputs_embeds=hidden,
                use_cache=False,
                return_dict=True,
            )
            reasoned = body_output.last_hidden_state

            output = self.decoder(reasoned)
            return output


    def create_tonic_brain(
        model_name: str = "Qwen/Qwen2.5-0.5B",
        elmer_weights_path: str = None,
        n_activations: int = 10,
        verbose: bool = False,
        transformer_body=None,
    ) -> TonicBrain:
        """Create a TonicBrain by reusing Elmer's surgery.

        1. Use shared transformer body (or load Qwen2.5-0.5B if none)
        2. Load ElmerBrain's trained encoder weights (the eyes)
        3. Create new ActivationDecoder (the voice — untrained initially)

        Args:
            model_name: HuggingFace model ID (only used if no shared body).
            elmer_weights_path: Path to elmer_brain_v0.1.pt.
            n_activations: Number of activation outputs.
            verbose: Print surgery details.
            transformer_body: Shared transformer body (e.g. from ProtoUniBrain).
                If provided, skips loading a second copy of the model.
        """
        _log = print if verbose else (lambda *a, **k: None)

        if elmer_weights_path is None:
            elmer_weights_path = os.path.expanduser(
                "~/Elmer/surgery/elmer_brain_v0.1.pt"
            )

        if transformer_body is not None:
            body = transformer_body
            hidden_dim = body.layers[0].self_attn.q_proj.in_features
            _log(f"Shared transformer body: {len(body.layers)} layers, hidden_dim={hidden_dim}")
        else:
            from transformers import AutoModelForCausalLM
            _log(f"Loading {model_name}...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name, dtype=torch.float32
            )
            hidden_dim = model.config.hidden_size
            body = model.model
            body.embed_tokens = nn.Identity()
            _log(f"Body extracted: {len(body.layers)} layers")

        # Create encoder and load Elmer's trained weights
        encoder = GraphStateEncoder(hidden_dim=hidden_dim)
        if os.path.exists(elmer_weights_path):
            ckpt = torch.load(elmer_weights_path, map_location="cpu",
                              weights_only=True)
            encoder.load_state_dict(ckpt["encoder_state"])
            _log(f"Encoder loaded from Elmer weights: {elmer_weights_path}")
        else:
            _log(f"WARNING: Elmer weights not found at {elmer_weights_path}")
            _log("Encoder will use random initialization")

        # Create new decoder
        decoder = ActivationDecoder(
            hidden_dim=hidden_dim,
            n_activations=n_activations,
        )
        decoder_params = sum(p.numel() for p in decoder.parameters())
        _log(f"ActivationDecoder: {decoder_params:,} params (untrained)")

        # Assemble
        brain = TonicBrain(
            transformer_body=body,
            encoder=encoder,
            decoder=decoder,
        )

        total = sum(p.numel() for p in brain.parameters())
        _log(f"TonicBrain assembled: {total:,} total params")

        return brain


    def save_tonic_brain(brain: TonicBrain, path: str) -> None:
        """Save TonicBrain weights (encoder + decoder only)."""
        torch.save({
            "encoder_state": brain.encoder.state_dict(),
            "decoder_state": brain.decoder.state_dict(),
            "config": {
                "hidden_dim": brain.decoder.hidden_dim,
                "n_activations": brain.decoder.n_activations,
                "base_model": "Qwen/Qwen2.5-0.5B",
            },
        }, path)
        logger.info("TonicBrain saved to %s", path)


    def load_tonic_brain(
        path: str,
        model_name: str = "Qwen/Qwen2.5-0.5B",
        transformer_body=None,
    ) -> TonicBrain:
        """Load a trained TonicBrain from checkpoint.

        Args:
            transformer_body: Shared body (e.g. from ProtoUniBrain).
                Skips from_pretrained if provided — saves ~2GB RAM.
        """
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        cfg = ckpt["config"]

        if transformer_body is not None:
            body = transformer_body
        else:
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                model_name, dtype=torch.float32
            )
            body = model.model
            body.embed_tokens = nn.Identity()

        encoder = GraphStateEncoder(hidden_dim=cfg["hidden_dim"])
        encoder.load_state_dict(ckpt["encoder_state"])

        decoder = ActivationDecoder(
            hidden_dim=cfg["hidden_dim"],
            n_activations=cfg["n_activations"],
        )
        decoder.load_state_dict(ckpt["decoder_state"])

        return TonicBrain(body, encoder, decoder)
