"""Causal-to-Spatial Perceiver Bridge model architecture."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .attention import AsymmetricRoPECrossAttention


class RMSNorm(nn.Module):
    """Root Mean Square Normalization."""

    def __init__(self, dim: int, eps:float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalizes the input tensor using RMSNorm."""
        # Variance calculation in float32 for stability
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x_norm = x * torch.rsqrt(variance + self.eps)
        return (x_norm.to(x.dtype) * self.weight)


class SwiGLU(nn.Module):
    """Swish Gated Linear Unit (SwiGLU) Feed-Forward Network."""

    def __init__(self, dim: int, hidden_dim: Optional[int] = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        # Two linear layers for the gating mechanism
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        # Final output projection
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights using Xavier Uniform."""
        nn.init.xavier_uniform_(self.w1.weight)
        nn.init.xavier_uniform_(self.w2.weight)
        nn.init.xavier_uniform_(self.w3.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the SwiGLU activation function."""
        # x: [B, Seq, Dim] -> [B, Seq, Hidden]
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class CSPBBlock(nn.Module):
    """A single Transformer block of the Causal-to-Spatial Perceiver Bridge."""

    def __init__(self, dim: int = 1024, heads: int = 16) -> None:
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            batch_first=True
        )
        self.norm2 = RMSNorm(dim)
        self.cross_attn = AsymmetricRoPECrossAttention(dim=dim, heads=heads)
        self.norm3 = RMSNorm(dim)
        self.mlp = SwiGLU(dim=dim)
    
    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through the block."""
        # 1. Self-Attention with residual connection
        norm_x = self.norm1(x)
        self_attn_out, _ = self.self_attn(query=norm_x, key=norm_x, value=norm_x)
        x = x + self_attn_out

        # 2. Cross-Attention with residual connection
        x = x + self.cross_attn(q_x=self.norm2(x), kv_x=self.norm2(context), mask=mask)

        # 3. SwiGLU MLP with residual connection
        x = x + self.mlp(self.norm3(x))

        return x


class CausalToSpatialPerceiverBridge(nn.Module):
    """The full Adapter model bridging Qwen and SDXL."""

    def __init__(
        self,
        depth: int = 6,
        qwen_dim: int = 1024,
        internal_dim: int = 1024,
        sdxl_context_dim: int = 2048,
        sdxl_pooled_dim: int = 1280,
        num_queries: int = 78
    ) -> None:
        super().__init__()
        self.num_queries = num_queries

        # Learnable latent queries acting as the 'receptive slots'
        # Shape: [1, 78, 1024]
        self.latent_queries = nn.Parameter(torch.randn(1, num_queries, internal_dim))

        # Adapter blocks
        self.blocks = nn.ModuleList([
            CSPBBlock(dim=internal_dim) for _ in range(depth)
        ])

        # Manifold output projections
        # Projects the first 77 tokens from 1024 to 2048
        self.ctx_proj = nn.Linear(internal_dim, sdxl_context_dim)

        # Deep projection for the 78th token to match SDXL's EOS pooled vector
        self.pool_proj = nn.Sequential(
            nn.Linear(internal_dim, internal_dim),
            nn.SiLU(),
            nn.Linear(internal_dim, sdxl_pooled_dim)
        )

        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize parameters for stable training."""
        nn.init.normal_(self.latent_queries, std=0.02)
        nn.init.xavier_uniform_(self.ctx_proj.weight)
        if self.ctx_proj.bias is not None:
            nn.init.zeros_(self.ctx_proj.bias)
    
    def forward(
        self,
        qwen_hidden_states: torch.Tensor,
        qwen_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the Adapter.
        
        Translates causal LLM states into SDXL spatial manifolds.
        
        Args:
            qwen_hidden_states: Cached output from Qwen3.5.
                - Shape: [B, N, 1024].
            qwen_mask: Optional Boolean mask where True is valid text.
                - Shape: [B, N].
        
        Returns:
            A tuple containing:
                - sdxl_context: Spatial cross-attention embeddings for SDXL.
                    - Shape: [B, 77, 2048].
                - sdxl_pooled: Global pooled embeddings.
                    - Shape: [B, 1280].
        """
        batch_size = qwen_hidden_states.shape[0]

        # Expand static latent queries to match batch size
        # x: [B, 78, 1024]
        x = self.latent_queries.expand(batch_size, -1, -1)

        # Pass through the Perceiver Resampler blocks
        for block in self.blocks:
            x = block(x, context=qwen_hidden_states, mask=qwen_mask)
        
        # Manifold routing
        # Split the sequence into Context (slots 0-76) and Pooled (slot 77)
        ctx_latents = x[:, :77, :]          # [B, 77, 1024]
        cls_latent = x[:, 77, :]            # [B, 1024]

        # Project to target topological manifolds
        # sdxl_context: [B, 77, 2048]
        sdxl_context = self.ctx_proj(ctx_latents)

        # sdxl_pooled: [B, 1280]
        sdxl_pooled = self.pool_proj(cls_latent)

        return sdxl_context, sdxl_pooled
