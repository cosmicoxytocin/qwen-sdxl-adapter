"""Bidirectional Perceiver Adapter architecture modeling."""

import torch
import torch.nn as nn
from config import AdapterConfig


class PerceiverBlock(nn.Module):
    """A single Perceiver cross-attention and self-attention block."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm3 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor, context: torch.Tensor, context_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Perceiver block."""
        # Cross-attention (Queries attend to LLM Context)
        x_norm = self.norm1(x)
        # context_mask needs to be formatted for key_padding_mask
        attn_out, _ = self.cross_attn(
            query=x_norm,
            key=context,
            value=context,
            key_padding_mask=context_mask
        )
        x = x + attn_out  # Residual connection

        # Self-attention (Queries attend to themselves bidirectionally)
        x_norm = self.norm2(x)
        attn_out, _ = self.self_attn(query=x_norm, key=x_norm, value=x_norm)
        x = x + attn_out

        # MLP
        x = x + self.mlp(self.norm3(x))
        return x


class QwenToSDXLAdapter(nn.Module):
    """Translates Qwen3.5 embeddings to SDXL UNet conditioning space."""
    
    def __init__(self, config: AdapterConfig):
        super().__init__()
        self.config = config

        # We need num_queries for context (77) + 1 for the pooled vector = 78 total slots
        self.total_queries = config.num_queries + 1
        self.latent_queries = nn.Parameter(torch.randn(1, self.total_queries, config.target_dim) * 0.02)  # Learnable latent queries
        self.context_proj = nn.Linear(config.source_dim, config.target_dim)  # Project Qwen3.5 embeddings to target dim
        self.blocks = nn.ModuleList([
            PerceiverBlock(
                dim=config.target_dim,
                num_heads=config.num_heads,
                dropout=config.dropout
            )
            for _ in range(config.num_layers)
        ])
        self.pooled_proj = nn.Sequential(
            nn.LayerNorm(config.target_dim),
            nn.Linear(config.target_dim, config.pooled_dim)
        )
        self.out_norm = nn.LayerNorm(config.target_dim)
    
    def forward(self, qwen_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the adapter."""
        B = qwen_hidden_states.size(0)

        # Project context to target dim. SHAPE: (B, N, 2048)
        context = self.context_proj(qwen_hidden_states)

        # Expand latent queries for batch. SHAPE: (B, 78, 2048)
        x = self.latent_queries.expand(B, -1, -1)

        # Invert mask for PyTorch MHA (True = ignore).SHAPE: (B, N)
        key_padding_mask = (attention_mask == 0)

        for block in self.blocks:
            x = block(x, context, key_padding_mask)
        
        # Split output. CONTEXT: 77 tokens, POOLED: 1 token
        # SHAPE of context_embeds: (B, 77, 2048)
        context_embeds = self.out_norm(x[:, :self.config.num_queries, :])

        # SHAPE of pooled_embeds: (B, 1280)
        pooled_token = x[:, self.config.num_queries, :]
        pooled_embeds = self.pooled_proj(pooled_token)

        return context_embeds, pooled_embeds
