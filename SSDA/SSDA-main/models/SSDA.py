import torch
import os
import einops
import torch.nn.functional as F
from torch import nn
from PIL import Image
import requests
from tqdm import tqdm
import os
import copy
from . import models_mae
from utils.util import download_file
from layers.lora_layers import apply_lora_to_model
from layers.temporal_structure_preservation import SGLoRA
from layers.spectral_enhancement import SpectralMagnitudeAligner

# 两条分支无对齐

MAE_ARCH = {
    "mae_base": [models_mae.mae_vit_base_patch16, "mae_visualize_vit_base.pth"],
    "mae_large": [models_mae.mae_vit_large_patch16, "mae_visualize_vit_large.pth"],
    "mae_huge": [models_mae.mae_vit_huge_patch14, "mae_visualize_vit_huge.pth"]
}

MAE_DOWNLOAD_URL = "https://dl.fbaipublicfiles.com/mae/visualize/"


class TensorResize(nn.Module):
    """
    A simple tensor Resize implementation, replacing torchvision.transforms.Resize.
    Only supports tensor input with shape [B, C, H, W].
    """

    def __init__(self, size, interpolation: str = "bilinear"):
        super().__init__()
        # size: (new_H, new_W)
        self.size = size
        # interpolation: "bilinear" / "nearest" / "bicubic"
        self.interpolation = interpolation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mode = self.interpolation
        if mode in ("bilinear", "bicubic"):
            return F.interpolate(x, size=self.size, mode=mode, align_corners=False)
        else:
            return F.interpolate(x, size=self.size, mode=mode)


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        self.use_norm = config.use_norm
        self.norm_const = config.norm_const
        self.task_name = config.task_name
        self.nvars = config.enc_in
        self.pred_len = config.pred_len
        self.seq_len = config.seq_len
        self.context_len = config.seq_len
        self.periodicity = config.periodicity
        self.align_const = config.align_const
        self.interpolation = config.interpolation
        self.arch = config.vm_arch
        self.base_model_path = config.vision_model_path
        self.lora_rank = config.r
        self.lora_alpha = config.lora_alpha
        self.lora_dropout = config.lora_dropout
        self.fusion_alpha = nn.Parameter(torch.tensor(0.5))

        self._init_model()
                
        self._update_config(context_len=self.seq_len, 
                            pred_len=self.pred_len, 
                            periodicity=self.periodicity, 
                            norm_const=self.norm_const, 
                            align_const=self.align_const, 
                            interpolation=self.interpolation)
        
        # Spectral Magnitude Aligner
        self.spectral_enhancer = SpectralMagnitudeAligner(image_size=self.image_size, residual_weight=config.residual_weight)
        
    def _init_model(self):

        if self.arch not in MAE_ARCH:
            raise ValueError(f"Unknown arch: {self.arch}. Should be in {list(MAE_ARCH.keys())}")

        self.base_model = MAE_ARCH[self.arch][0]()

        if not os.path.isfile(self.base_model_path):
            remote_url = MAE_DOWNLOAD_URL + MAE_ARCH[self.arch][1]
            download_file(remote_url, self.base_model_path)
        try:
            checkpoint = torch.load(self.base_model_path, map_location='cpu')
            self.base_model.load_state_dict(checkpoint['model'], strict=True)
        except:
            print(f"Bad checkpoint file. Please delete {self.base_model_path} and redownload!")
        
        for name, param in self.base_model.named_parameters():
            param.requires_grad = 'norm' in name or 'pos_embed' in name

        self.structural_model = copy.deepcopy(self.base_model)
        
        # Structural Guided Low-Rank 
        self.structural_model = SGLoRA(
            base_model=self.structural_model, 
            periodicity=self.periodicity, 
            seq_len=self.seq_len,
            lora_config={
                'r': self.lora_rank, 
                'alpha': self.lora_alpha, 
                'dropout': self.lora_dropout
            }
        )

        self.spectral_model = self.base_model
    
    
    def _update_config(self, context_len, pred_len, periodicity=1, norm_const=0.4, align_const=0.4, interpolation='bilinear'):
        self.image_size = self.base_model.patch_embed.img_size[0]
        self.patch_size = self.base_model.patch_embed.patch_size[0]
        self.num_patch = self.image_size // self.patch_size

        self.context_len = context_len
        self.pred_len = pred_len
        self.periodicity = periodicity

        self.pad_left = 0
        self.pad_right = 0
        if self.context_len % self.periodicity != 0:
            self.pad_left = self.periodicity - self.context_len % self.periodicity

        if self.pred_len % self.periodicity != 0:
            self.pad_right = self.periodicity - self.pred_len % self.periodicity
        
        input_ratio = (self.pad_left + self.context_len) / (self.pad_left + self.context_len + self.pad_right + self.pred_len)
        self.num_patch_input = int(input_ratio * self.num_patch * align_const)
        if self.num_patch_input == 0:
            self.num_patch_input = 1
        self.num_patch_output = self.num_patch - self.num_patch_input
        self.adjust_input_ratio = self.num_patch_input / self.num_patch

        interp_mode = interpolation  # "bilinear" / "nearest" / "bicubic"       
        self.scale_x = ((self.pad_left + self.context_len) // self.periodicity) / (int(self.image_size * self.adjust_input_ratio))
        self.input_resize = TensorResize((self.image_size, int(self.image_size * self.adjust_input_ratio)), interpolation=self.interpolation)
        self.output_resize = TensorResize((self.periodicity, int(round(self.image_size * self.scale_x))),  interpolation=self.interpolation)

        self.norm_const = norm_const

        mask = torch.ones((self.num_patch, self.num_patch)).to(self.base_model.cls_token.device)
        mask[:, :self.num_patch_input] = torch.zeros((self.num_patch, self.num_patch_input))
        self.register_buffer("mask", mask.float().reshape((1, -1)))
        self.mask_ratio = torch.mean(mask).item()


    def _ts_to_vision_input(self, x: torch.Tensor, use_enhancement=False) -> torch.Tensor:
        x_enc = einops.rearrange(x, 'b l n -> b n l')
        x_pad = F.pad(x_enc, (self.pad_left, 0), mode='replicate') 
        x_2d = einops.rearrange(x_pad, 'b n (p f) -> (b n) 1 f p', f=self.periodicity)
        x_resize = self.input_resize(x_2d)
        if use_enhancement:
            x_resize = self.spectral_enhancer(x_resize)
        else:
            x_resize = einops.repeat(x_resize, 'b 1 h w -> b c h w', c=3)
        masked = torch.zeros((x_2d.shape[0], 3, self.image_size, self.num_patch_output * self.patch_size), device=x_2d.device, dtype=x_2d.dtype)
        image_input = torch.cat([x_resize, masked], dim=-1)
        
        return image_input


    def _reconstruct(self, x, model):
        x_reconstructed = model.unpatchify(x)
        x_grey = torch.mean(x_reconstructed, 1, keepdim=True)
        x_segmentation = self.output_resize(x_grey)

        return x_reconstructed, x_segmentation
    

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None, export_image=False):
        
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, export_image=export_image)
            return dec_out
        return None

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, export_image=False, fp64=False):
        # Forecasting using visual model.
        # x_enc: look-back window, size: [bs x context_len x nvars]
        # fp64=True can avoid math overflow in some benchmark, like Bitcoin.
        # return: forecasting window, size: [bs x pred_len x nvars]

        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach() # [bs x 1 x nvars]
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc.to(torch.float64) if fp64 else x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5) # [bs x 1 x nvars]
            stdev /= self.norm_const
            x_enc /= stdev
            
        # ==============================================================
        # Spectral Aware Adaptation Branch
        # ==============================================================
        spectral_input = self._ts_to_vision_input(x_enc, use_enhancement=True)
        _, spectral, spectral_mask = self.spectral_model(spectral_input, mask_ratio=self.mask_ratio, noise=einops.repeat(self.mask, '1 l -> n l', n=spectral_input.shape[0]))
        spectral_reconstruct, spectral_segmentation = self._reconstruct(spectral, self.spectral_model)
        spectral_flatten = einops.rearrange(
            spectral_segmentation, 
            '(b n) 1 f p -> b (p f) n', 
            b=x_enc.shape[0], f=self.periodicity) # flatten
        spectral_y = spectral_flatten[:, self.pad_left + self.context_len: self.pad_left + self.context_len + self.pred_len, :] # extract the forecasting window
        
        
        # ==============================================================
        # Structural Guided Low-Rank Adaptation Branch
        # ==============================================================
        structural_input = self._ts_to_vision_input(x_enc, use_enhancement=False)
        _, structural, structural_mask = self.structural_model(structural_input, mask_ratio=self.mask_ratio, noise=einops.repeat(self.mask, '1 l -> n l', n=structural_input.shape[0]))
        structural_reconstruct, structural_segmentation = self._reconstruct(structural, self.structural_model)
        structural_flatten = einops.rearrange(
            structural_segmentation, 
            '(b n) 1 f p -> b (p f) n', 
            b=x_enc.shape[0], f=self.periodicity
        ) # flatten
        structural_y = structural_flatten[:, self.pad_left + self.context_len: self.pad_left + self.context_len + self.pred_len, :] # extract the forecasting window
        

        # learnable weighted average of structural & spectral branches
        alpha = torch.clamp(self.fusion_alpha, 0.0, 1.0)
        # adaptive fusion
        y = alpha * structural_y + (1.0 - alpha) * spectral_y

        if self.use_norm:
            y = y * (stdev.repeat(1, self.pred_len, 1))
            y = y + (means.repeat(1, self.pred_len, 1))

        if export_image:
            structural_mask = structural_mask.detach()
            structural_mask = structural_mask.unsqueeze(-1).repeat(1, 1, self.structural_model.lora_model.patch_embed.patch_size[0]**2 *3)
            structural_mask = self.structural_model.unpatchify(structural_mask)
            structural_image_reconstructed = structural_input * (1 - structural_mask) + structural_reconstruct * structural_mask

            spectral_mask = spectral_mask.detach()
            spectral_mask = spectral_mask.unsqueeze(-1).repeat(1, 1, self.spectral_model.patch_embed.patch_size[0]**2 *3)
            spectral_mask = self.spectral_model.unpatchify(spectral_mask)
            spectral_image_reconstructed = spectral_input * (1 - spectral_mask) + spectral_reconstruct * spectral_mask

            structural_image_input = einops.rearrange(structural_input, '(b n) c h w -> b n h w c', b=x_enc.shape[0])
            spectral_image_input = einops.rearrange(spectral_input, '(b n) c h w -> b n h w c', b=x_enc.shape[0])
            structural_image_reconstructed = einops.rearrange(structural_image_reconstructed, '(b n) c h w -> b n h w c', b=x_enc.shape[0])
            spectral_image_reconstructed = einops.rearrange(spectral_image_reconstructed, '(b n) c h w -> b n h w c', b=x_enc.shape[0])

            return y[:, -self.pred_len:, :], structural_image_input, structural_image_reconstructed, spectral_image_input, spectral_image_reconstructed
        return y[:, -self.pred_len:, :]
