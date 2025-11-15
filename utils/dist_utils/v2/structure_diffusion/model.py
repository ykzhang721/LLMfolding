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
        
    def forward(self, x, key_padding_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=key_padding_mask.unsqueeze(1).unsqueeze(1),
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False,
        )

        x = attn_out.transpose(1, 2).reshape(B, N, C)
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

    def forward(self, x, key_padding_mask=None):
        x = x + self.attn(self.norm1(x), key_padding_mask=key_padding_mask)
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
        
    def forward(self, distance_matrix: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            distance_matrix: [B, N, N, 3] distance matrix (or [N, N, 3] for single sample)
            mask: Optional boolean mask of shape [B, N] (or [N]) indicating valid tokens
        Returns:
            embedded: [B, N, embed_dim] token embeddings
        """
        B, N, _, _ = distance_matrix.shape

        distances_from = distance_matrix
        distances_to = distance_matrix.transpose(1, 2)
        pairwise_features = torch.cat([distances_from, distances_to], dim=-1)  # [B, N, N, 6]

        pairwise_embed = self.pair_proj(pairwise_features)  # [B, N, N, embed_dim]

        mask = mask.to(pairwise_embed.device)
        mask_bool = mask.unsqueeze(1).unsqueeze(-1)  # [B, 1, N, 1]
        mask_float = mask_bool.to(pairwise_embed.dtype)
        pairwise_embed = pairwise_embed * mask_float

        counts = mask_float.sum(dim=2).clamp_min(1.0)
        token_embed = pairwise_embed.sum(dim=2) / counts
        token_embed = token_embed * mask.unsqueeze(-1).to(pairwise_embed.dtype)

        return token_embed


class StructureViT(nn.Module):
    """
    Vision Transformer-based model for predicting residue coordinates from distance matrices.
    Input: Variable-length distance matrix [N, N, 3]
    Output: Frame coordinates [N, 3, 3] or CA coordinates [N, 3]
    """

    def __init__(
        self, 
        in_channels: int = 3,
        embed_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        max_seq_len: int = 2048,
        use_sinusoidal_pos_embed: bool = False,
        use_frame_coordinates: bool = True,
        output_dim: Optional[int] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.use_sinusoidal_pos_embed = use_sinusoidal_pos_embed
        self.use_frame_coordinates = use_frame_coordinates
        self.output_dim = output_dim
        
        # Distance matrix embedding
        self.distance_embedding = DistanceMatrixEmbedding(in_channels, embed_dim)
        
        # Positional encoding (learnable or sinusoidal)
        self.pos_embed = None
        
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
        
    def forward(self, distance_matrix: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Forward pass through the structure prediction model.
        
        Args:
            distance_matrix: [B, N, N, 3] distance matrix (or [N, N, 3] for single sample)
            mask: Optional boolean tensor [B, N] (or [N]) marking valid tokens
            
        Returns:
            coordinates: [B, N, 3, 3] predicted residue frames if use_frame_coordinates else [B, N, 3]
        """
        B, N, _, _ = distance_matrix.shape

        x = self.distance_embedding(distance_matrix, mask)  # [B, N, embed_dim]

        pos_embed = self._get_pos_embed(N, dtype=x.dtype, device=x.device)
        x = x + pos_embed
        for block in self.blocks:
            x = block(x, key_padding_mask=mask)

        x = self.norm(x)

        coordinates = self.coord_head(x)  # [B, N, output_dim]

        if self.use_frame_coordinates:
            coordinates = coordinates.view(B, N, 3, 3)
        else:
            out_dim = self.output_dim if self.output_dim is not None else 3
            coordinates = coordinates.view(B, N, out_dim)

        return coordinates

    def _get_pos_embed(self, seq_len: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if self.use_sinusoidal_pos_embed:
            return self._build_sinusoidal_table(seq_len, self.embed_dim, device, dtype).unsqueeze(0)

        if seq_len <= self.max_seq_len:
            pos_embed = self.pos_embed[:, :seq_len, :]
        else:
            pos_embed = F.interpolate(
                self.pos_embed.transpose(1, 2),
                size=seq_len,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)

        return pos_embed.to(device=device, dtype=dtype)

    def _build_sinusoidal_table(
        self,
        seq_len: int,
        embed_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        base_dtype = torch.float32
        position = torch.arange(seq_len, device=device, dtype=base_dtype).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, device=device, dtype=base_dtype) * (-(math.log(10000.0) / embed_dim))
        )

        table = torch.zeros(seq_len, embed_dim, device=device, dtype=base_dtype)
        angles = position * div_term
        table[:, 0::2] = torch.sin(angles)
        table[:, 1::2] = torch.cos(angles)

        if dtype != base_dtype:
            table = table.to(dtype=dtype)

        return table


class StructurePredictionModel(nn.Module):
    """
    Complete structure prediction model that maps distance matrices to residue coordinates.
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        max_seq_len: int = 2048,
        use_sinusoidal_pos_embed: bool = False,
        use_frame_coordinates: bool = True,
        output_dim: Optional[int] = None,
    ):
        super().__init__()
        self.use_frame_coordinates = use_frame_coordinates

        expected_dim = 9 if self.use_frame_coordinates else 3
        if output_dim is None:
            output_dim = expected_dim
        elif output_dim != expected_dim:
            raise ValueError(
                f"StructurePredictionModel expected output_dim={expected_dim} when use_frame_coordinates="
                f"={self.use_frame_coordinates}, but received {output_dim}."
            )

        self.vit = StructureViT(
            in_channels=in_channels,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            max_seq_len=max_seq_len,
            use_sinusoidal_pos_embed=use_sinusoidal_pos_embed,
            use_frame_coordinates=self.use_frame_coordinates,
            output_dim=output_dim,
        )
        
    def forward(self, distance_matrix, mask: Optional[torch.Tensor] = None):
        """
        Forward pass.
        
        Args:
            distance_matrix: [B, N, N, 3] distance matrix (or [N, N, 3] for single sample)
            mask: Optional boolean tensor [B, N] (or [N])
            
        Returns:
            coordinates: [B, N, 3, 3] predicted residue frames if use_frame_coordinates else [B, N, 3]
        """
        if mask is None:
            B, N, _, _ = distance_matrix.shape
            mask = torch.ones(B, N, dtype=torch.bool, device=distance_matrix.device)
        return self.vit(distance_matrix, mask=mask)
