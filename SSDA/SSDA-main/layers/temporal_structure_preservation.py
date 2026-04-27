import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.lora_layers import apply_lora_to_model


class TemporalGroundingAdapter(nn.Module):
    """
    Temporal Structure Alignment Module (TSA)

    Core idea: Use original time series positional encoding to enhance/replace spatial positional encoding of images.
    """

    def __init__(self, num_patches, periodicity, embed_dim):
        super().__init__()
        self.num_patches = num_patches
        self.periodicity = periodicity
        self.embed_dim = embed_dim

        # Compute image dimensions
        self.num_cols = periodicity
        self.num_rows = num_patches // periodicity if periodicity > 0 else 1

        # ========== Core Learnable Parameters ==========
        self.fusion_gate = nn.Parameter(torch.tensor(0.1))
        self.temporal_proj = nn.Linear(embed_dim, embed_dim)

    def temporal_encoding(self, time_indices):
        """
        Generate temporal positional encoding (1D Sinusoidal)

        Args:
            time_indices: [B, L] Time indices

        Returns:
            encoding: [B, L, D] Temporal positional encoding
        """
        B, L = time_indices.shape
        device = time_indices.device

        # 1D Sinusoidal positional encoding
        # d_model = embed_dim
        position = time_indices.unsqueeze(-1)  # [B, L, 1]

        # Create frequency base
        div_term = torch.exp(
            torch.arange(0, self.embed_dim, 2, device=device).float() *
            (-torch.log(torch.tensor(10000.0)) / self.embed_dim)
        )

        # Sin and cos encoding
        pe = torch.zeros(B, L, self.embed_dim, device=device)
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)

        return pe

    def forward(self, x):
        """
        Forward pass: Enhance temporal structure awareness

        Args:
            x: [B, L, D] Patch embeddings

        Returns:
            x_enhanced: [B, L, D] Enhanced features
        """
        B, L, D = x.shape

        # patch_indices = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)

        # Step 1: Get time indices
        # time_indices = self.get_time_index(patch_indices)  # [B, L]
        time_indices = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)

        # Step 2: Generate temporal encoding
        temporal_enc = self.temporal_encoding(time_indices)  # [B, L, D]

        # Step 3: Project to feature space
        temporal_enc = self.temporal_proj(temporal_enc)  # [B, L, D]

        # Step 4: Fuse
        gate = torch.sigmoid(self.fusion_gate)
        x_enhanced = x + gate * temporal_enc

        return x_enhanced


class SGLoRA(nn.Module):
    """
    Temporal LoRA (TLoRA) - Temporal Structure Enhancement Module

    Combines LoRA with time series structure preservation module, mapping image spatial positions to original temporal positions,
    using temporal positional encoding to enhance features, solving the structural misalignment problem after converting time series to images.
    """

    def __init__(self,
                 base_model,
                 periodicity=24,
                 seq_len=768,
                 lora_config=None,
                 target_modules=['qkv', 'proj']):
        super().__init__()

        self.periodicity = periodicity
        self.seq_len = seq_len

        if lora_config is None:
            lora_config = {'r': 4, 'alpha': 16, 'dropout': 0.1}

        # Get model dimension
        embed_dim = base_model.pos_embed.shape[-1]
        num_patches = base_model.pos_embed.shape[1] - 1  # Subtract cls token

        # ========== Core Innovation Module ==========

        # TSA: Temporal Structure Alignment (map spatial positions to temporal positions)
        self.structure_alignment = TemporalGroundingAdapter(
            num_patches, periodicity, embed_dim
        )

        # Save base model and apply LoRA
        self.lora_model = base_model
        self.lora_model = apply_lora_to_model(
            self.lora_model,
            target_modules=target_modules,
            r=lora_config['r'],
            lora_alpha=lora_config['alpha'],
            lora_dropout=lora_config['dropout']
        )

        self.gate = nn.Parameter(torch.tensor(0.3))

    def unpatchify(self, x):
        return self.lora_model.unpatchify(x)

    def inject_temporal_structure(self, x_patches):
        """
        Inject temporal structure information

        Args:
            x_patches: [B, L, D] Patch embeddings

        Returns:
            x_enhanced: [B, L, D] Features after injecting temporal structure
        """
        # TSA: Temporal Structure Alignment
        x_enhanced = self.structure_alignment(x_patches)

        return x_enhanced

    def forward_encoder(self, x, mask_ratio, noise=None):
        """
        Override encoder with temporal structure injection
        """
        # embed patches
        x = self.lora_model.patch_embed(x)  # [B, L, D]

        # ========== Core Innovation: Inject Temporal Structure ==========
        x = self.inject_temporal_structure(x)  # [B, L, D]

        # add pos embed w/o cls token
        x = x + self.lora_model.pos_embed[:, 1:, :]

        # masking
        x, mask, ids_restore = self.lora_model.random_masking(x, mask_ratio, noise)

        # append cls token
        cls_token = self.lora_model.cls_token + self.lora_model.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.lora_model.blocks:
            x = blk(x)
        x = self.lora_model.norm(x)

        return x, mask, ids_restore

    def forward(self, imgs, mask_ratio=0.75, noise=None):
        """Forward pass"""
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio, noise)
        pred = self.lora_model.forward_decoder(latent, ids_restore)

        return None, pred, mask
