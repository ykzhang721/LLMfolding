import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Any, Optional, Tuple
from ..vector_quantize_pytorch import (
    VectorQuantize, FSQ, LFQ, SimVQ, 
    ResidualVQ, ResidualFSQ, ResidualLFQ, ResidualSimVQ
)


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
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
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
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViTEncoder(nn.Module):
    """ViT-based encoder."""
    def __init__(
        self, 
        in_channels: int = 3,
        embed_dim: int = 256,
        patch_size: int = 16,
        num_layers: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        z_channels: int = 64,
        double_z: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.z_channels = z_channels

        # Conv and patch embedding
        self.conv_in = nn.Conv2d(in_channels, embed_dim // 4, kernel_size=3, stride=1, padding=1)
        self.patch_embed = nn.Conv2d(embed_dim // 4, embed_dim, kernel_size=patch_size, stride=patch_size)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        output_channels = 2 * z_channels if double_z else z_channels
        self.head = nn.Linear(embed_dim, output_channels)
        
    def get_pos_embed(self, h_patches, w_patches):
        num_patches = h_patches * w_patches
        pos_embed = torch.zeros(1, num_patches, self.embed_dim, device=next(self.parameters()).device)
        pos_h = torch.arange(h_patches, device=pos_embed.device).unsqueeze(1).repeat(1, w_patches).flatten()
        pos_w = torch.arange(w_patches, device=pos_embed.device).unsqueeze(0).repeat(h_patches, 1).flatten()
        dim = self.embed_dim // 2
        div_term = torch.exp(torch.arange(0, dim, 2, device=pos_embed.device) * -(math.log(10000.0) / dim))

        pos_embed[0, :, 0::4] = torch.sin(pos_h.unsqueeze(1) * div_term).view(num_patches, -1)
        pos_embed[0, :, 1::4] = torch.cos(pos_h.unsqueeze(1) * div_term).view(num_patches, -1)
        pos_embed[0, :, 2::4] = torch.sin(pos_w.unsqueeze(1) * div_term).view(num_patches, -1)
        pos_embed[0, :, 3::4] = torch.cos(pos_w.unsqueeze(1) * div_term).view(num_patches, -1)

        return pos_embed
        
    def forward(self, x):
        # Convert from [B, L, L, C] to [B, C, L, L] if needed
        if x.dim() == 4 and x.shape[-1] == self.in_channels:  # [B, L, L, C] format
            x = x.permute(0, 3, 1, 2)  # [B, C, L, L]
        
        B, C, H, W = x.shape

        x = self.conv_in(x)     # [B, embed_dim//4, H, W]
        x = self.patch_embed(x) # [B, embed_dim, H//patch_size, W//patch_size]

        _, embed_dim, h_patches, w_patches = x.shape
        x = x.flatten(2).transpose(1, 2)    # [B, (H//patch_size)*(W//patch_size), embed_dim]

        pos_embed = self.get_pos_embed(h_patches, w_patches)
        x = x + pos_embed
        
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = self.head(x)    # [B, num_patches, z_channels]
        
        x = x.transpose(1, 2).reshape(B, -1, h_patches, w_patches)
        return x


class PatchReconstruction(nn.Module):
    """Patch reconstruction module."""
    def __init__(self, embed_dim: int, patch_size: int, out_channels: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.ConvTranspose2d(embed_dim, out_channels, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, target_h, target_w):
        x = self.proj(x)

        if x.shape[-2] != target_h or x.shape[-1] != target_w:
            x = F.interpolate(x, size=(target_h, target_w), mode='bilinear', align_corners=False)
        return x


class ViTDecoder(nn.Module):
    """ViT-based decoder."""
    def __init__(
        self, 
        out_channels: int = 3,
        embed_dim: int = 256,
        patch_size: int = 16,
        num_layers: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        z_channels: int = 64,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.z_channels = z_channels

        self.input_proj = nn.Linear(z_channels, embed_dim)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        self.patch_recon = PatchReconstruction(embed_dim, patch_size, embed_dim // 4)
        self.conv_out = nn.Conv2d(embed_dim // 4, out_channels, kernel_size=3, stride=1, padding=1)
        
    def get_pos_embed(self, h_patches, w_patches):
        num_patches = h_patches * w_patches
        pos_embed = torch.zeros(1, num_patches, self.embed_dim, device=next(self.parameters()).device)

        pos_h = torch.arange(h_patches, device=pos_embed.device).unsqueeze(1).repeat(1, w_patches).flatten()
        pos_w = torch.arange(w_patches, device=pos_embed.device).unsqueeze(0).repeat(h_patches, 1).flatten()
        dim = self.embed_dim // 2
        div_term = torch.exp(torch.arange(0, dim, 2, device=pos_embed.device) * -(math.log(10000.0) / dim))

        pos_embed[0, :, 0::4] = torch.sin(pos_h.unsqueeze(1) * div_term).view(num_patches, -1)
        pos_embed[0, :, 1::4] = torch.cos(pos_h.unsqueeze(1) * div_term).view(num_patches, -1)
        pos_embed[0, :, 2::4] = torch.sin(pos_w.unsqueeze(1) * div_term).view(num_patches, -1)
        pos_embed[0, :, 3::4] = torch.cos(pos_w.unsqueeze(1) * div_term).view(num_patches, -1)

        return pos_embed
        
    def forward(self, z):
        B, z_channels, H, W = z.shape

        z = z.flatten(2).transpose(1, 2)  # [B, H*W, z_channels]
        x = self.input_proj(z)  # [B, H*W, embed_dim]

        pos_embed = self.get_pos_embed(H, W)
        x = x + pos_embed

        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = x.transpose(1, 2).reshape(B, -1, H, W)  # [B, embed_dim, H, W]
        
        target_h = target_w = H * self.patch_size
        x = self.patch_recon(x, target_h, target_w)  # [B, embed_dim//4, target_h, target_w]
        x = self.conv_out(x)  # [B, out_channels, target_h, target_w]
        
        # Convert back to [B, L, L, C] format
        x = x.permute(0, 2, 3, 1)  # [B, target_h, target_w, out_channels]
        return x


def create_quantizer(quantizer_type: str, **kwargs) -> nn.Module:
    """Create quantizer based on configuration."""
    quantizer_map = {
        'vq': VectorQuantize,
        'fsq': FSQ,
        'lfq': LFQ,
        'simvq': SimVQ,
        'residual_vq': ResidualVQ,
        'residual_fsq': ResidualFSQ,
        'residual_lfq': ResidualLFQ,
        'residual_simvq': ResidualSimVQ,
    }
    
    if quantizer_type not in quantizer_map:
        raise ValueError(f"Unknown quantizer type: {quantizer_type}. "
                        f"Available types: {list(quantizer_map.keys())}")
    
    quantizer_class = quantizer_map[quantizer_type]
    
    return quantizer_class(**kwargs)


class DiscreteTokenizer(nn.Module):
    
    encoder: nn.Module
    decoder: nn.Module
    quantizer: Any
    
    """
    Discrete tokenizer model using ViT architecture.
    Structure: ViT encoder -> quantization -> ViT decoder
    """
    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 256,
        patch_size: int = 16,
        num_layers: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        z_channels: int = 64,
        double_z: bool = False,
        quantizer_type: str = 'vq',
        quantizer_kwargs: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.z_channels = z_channels
        self.quantizer_type = quantizer_type
        
        if quantizer_kwargs is None:
            quantizer_kwargs = {}
        
        # ViT Encoder
        self.encoder = ViTEncoder(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            z_channels=z_channels,
            double_z=double_z
        )
        
        # Create quantizer
        # Ensure quantizer dim parameter is set correctly
        quantizer_kwargs['dim'] = z_channels
        self.quantizer = create_quantizer(quantizer_type, **quantizer_kwargs)
        
        # ViT Decoder
        self.decoder = ViTDecoder(
            out_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            z_channels=z_channels
        )
    
    def encode(self, x):
        """Encode input to quantized representation.
        
        Args:
            x: Input tensor [B, L, L, C]
        Returns:
            quantized: Quantized tensor [B, H, W, z_channels]
            indices: Quantization indices
            commit_loss: Commitment loss from quantizer
        """
        # ViT Encoder
        encoded = self.encoder(x)  # [B, z_channels, H, W]
        
        # Quantization
        B, D, H, W = encoded.shape
        encoded_flat = encoded.permute(0, 2, 3, 1).view(B, H * W, D)  # [B, H*W, z_channels]
        
        if self.quantizer_type in ['vq', 'simvq']:
            quantized, indices, commit_loss = self.quantizer(encoded_flat)
        elif self.quantizer_type in ['fsq']:
            quantized, indices = self.quantizer(encoded_flat)
            commit_loss = torch.tensor(0.0, device=x.device)
        elif self.quantizer_type in ['lfq']:
            quantized, indices, commit_loss = self.quantizer(encoded_flat)
        else:  # residual variants
            quantized, indices, commit_loss = self.quantizer(encoded_flat)
        
        # Reshape back to spatial format
        quantized = quantized.view(B, H, W, D).permute(0, 3, 1, 2)  # [B, z_channels, H, W]
        
        return quantized, indices, commit_loss
    
    def decode(self, quantized):
        """Decode quantized representation back to original space.
        
        Args:
            quantized: Quantized tensor [B, z_channels, H, W]
        Returns:
            reconstructed: Reconstructed tensor [B, L, L, C]
        """
        return self.decoder(quantized)
    
    def forward(self, x):
        """Forward pass through the model.
        
        Args:
            x: Input tensor [B, L, L, C]
        Returns:
            reconstructed: Reconstructed tensor [B, L, L, C]
            indices: Quantization indices
            commit_loss: Commitment loss
        """
        quantized, indices, commit_loss = self.encode(x)
        reconstructed = self.decode(quantized)
        
        return reconstructed, indices, commit_loss
    
    def indices_decode(self, indices):
        quantized = self.quantizer.indices_to_codes(indices)
        reconstructed = self.decode(quantized)
        return reconstructed
    
    def get_codebook_usage(self, indices):
        if hasattr(self.quantizer, 'codebook_size'):
            codebook_size = self.quantizer.codebook_size
        elif hasattr(self.quantizer, '_codebook'):
            codebook_size = self.quantizer._codebook.shape[0]
        else:
            return 1.0
        
        unique_codes = indices.unique().numel()
        return unique_codes / codebook_size
