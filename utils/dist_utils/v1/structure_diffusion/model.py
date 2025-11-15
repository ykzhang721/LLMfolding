import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Any, Optional, Tuple


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention module."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask == 0, float('-inf'))
        
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        return x


class MLP(nn.Module):
    """MLP module for transformer block."""
    def __init__(self, embed_dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with self-attention and MLP."""
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)
        
    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.norm1(x), attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class DistanceMatrixEmbedding(nn.Module):
    """Embedding layer for distance matrices."""
    def __init__(self, in_channels: int = 3, embed_dim: int = 256):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        # Use a more sophisticated embedding that considers pairwise interactions
        self.pair_proj = nn.Sequential(
            nn.Linear(in_channels * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
    def forward(self, distance_matrix):
        """
        Args:
            distance_matrix: [N, N, 3] distance matrix
        Returns:
            embedded: [N, embed_dim] token embeddings
        """
        N = distance_matrix.shape[0]
        
        # Use sophisticated pairwise embedding
        # For each token i, consider its relationships with all other tokens
        token_embeddings = []
        for i in range(N):
            # Get distances from token i to all other tokens
            distances_from_i = distance_matrix[i]  # [N, 3]
            distances_to_i = distance_matrix[:, i]   # [N, 3]
            
            # Concatenate pairwise information
            pairwise_features = torch.cat([distances_from_i, distances_to_i], dim=-1)  # [N, 6]
            
            # Project and aggregate
            pairwise_embed = self.pair_proj(pairwise_features)  # [N, embed_dim]
            token_embed = pairwise_embed.mean(dim=0)  # [embed_dim]
            token_embeddings.append(token_embed)
        
        embedded = torch.stack(token_embeddings, dim=0)  # [N, embed_dim]
        return embedded.unsqueeze(0)  # [1, N, embed_dim] - add batch dimension


class StructureViT(nn.Module):
    """
    Vision Transformer-based model for predicting coordinates from distance matrices.
    Input: Variable-length distance matrix [N, N, 3]
    Output: Coordinates [N, 3]
    """
    
    def __init__(
        self, 
        in_channels: int = 3,
        embed_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        output_dim: int = 3,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        
        # Distance matrix embedding
        self.distance_embedding = DistanceMatrixEmbedding(in_channels, embed_dim)
        
        # Positional encoding (learnable for variable lengths)
        self.max_seq_len = 2048  # Support up to 2048 tokens
        self.pos_embed = nn.Parameter(torch.randn(1, self.max_seq_len, embed_dim) * 0.02)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Output head for coordinate prediction
        self.coord_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, output_dim)
        )
        
    def forward(self, distance_matrix):
        """
        Forward pass through the structure prediction model.
        
        Args:
            distance_matrix: [N, N, 3] distance matrix
            
        Returns:
            coordinates: [N, 3] predicted coordinates
        """
        N = distance_matrix.shape[0]
        
        # Embed distance matrix to token embeddings
        x = self.distance_embedding(distance_matrix)  # [1, N, embed_dim]
        
        # Add positional encoding
        if N <= self.max_seq_len:
            pos_embed = self.pos_embed[:, :N, :]
        else:
            # For sequences longer than max_seq_len, interpolate positional embeddings
            pos_embed = F.interpolate(
                self.pos_embed.transpose(1, 2), 
                size=N, 
                mode='linear', 
                align_corners=False
            ).transpose(1, 2)
        
        x = x + pos_embed
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)  # [1, N, embed_dim]
        
        # Predict coordinates
        coordinates = self.coord_head(x.squeeze(0))  # [N, 3]
        
        return coordinates


class StructurePredictionModel(nn.Module):
    """
    Complete structure prediction model that learns mapping from distance matrices to coordinates.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        output_dim: int = 3,
    ):
        super().__init__()
        
        self.vit = StructureViT(
            in_channels=in_channels,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            output_dim=output_dim,
        )
        
    def forward(self, distance_matrix):
        """
        Forward pass.
        
        Args:
            distance_matrix: [N, N, 3] distance matrix
            
        Returns:
            coordinates: [N, 3] predicted coordinates
        """
        return self.vit(distance_matrix)
