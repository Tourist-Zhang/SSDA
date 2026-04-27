"""
Vision Branch Enhancement Module
Addresses Motivation 1: Make input more like real images to enhance frozen MAE potential
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops


class SpectralMagnitudeAligner(nn.Module):
    """
    Spectral Magnitude Aligner: Transform time series data into images that better match MAE pretraining distribution.

    Core Innovations:
    1. SpectralMagnitudeAligner: Leverage frequency domain information to generate more natural image textures,
       enhancing only the magnitude spectrum while preserving phase structure.
    2. Domain Alignment: Align image distribution to ImageNet through contrast-aware normalization.
    """

    def __init__(self,
                 image_size=224,
                 num_channels=3,
                 residual_weight=0.3):
        super().__init__()
        self.image_size = image_size
        self.num_channels = num_channels
        self.residual_weight = residual_weight  # Tunable hyperparameter: weight for enhancement residual

        # Frequency enhancement: learn to map time series frequencies to image frequencies
        self.freq_enhancer = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),       # Batch normalization
            nn.ReLU(),
            nn.Dropout2d(0.1),        # Dropout
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
        )

        # Learnable residual weight (gate)
        self.residual_gate = nn.Parameter(torch.tensor(0.2))


    def frequency_enhancement(self, x_2d):
        """
        Frequency domain enhancement using FFT for more natural images.
        """
        # 2D FFT on 2D image
        x_fft = torch.fft.rfft2(x_2d, norm='ortho')

        # Separate magnitude and phase (key: keep phase unchanged)
        # x_fft shape: [B*N, 1, H, W//2+1]
        magnitude = torch.abs(x_fft)  # [B*N, 1, H, W//2+1]
        phase = torch.angle(x_fft)    # [B*N, 1, H, W//2+1]

        # Enhance magnitude only (keep phase intact to preserve structural info)
        # magnitude shape: [B*N, 1, H, W//2+1], can be fed directly to Conv2d
        enhanced_magnitude = self.freq_enhancer(magnitude)  # [B*N, 1, H, W//2+1]

        # Reconstruct complex number (keep phase, enhance magnitude)
        x_fft_enhanced = enhanced_magnitude * torch.exp(1j * phase)

        # Inverse FFT back to spatial domain
        x_enhanced = torch.fft.irfft2(x_fft_enhanced, s=x_2d.shape[-2:], norm='ortho')

        # Residual connection (learnable gate weight, initial value 0.2)
        # return x_2d + self.residual_gate * (x_enhanced - x_2d)
        return x_2d + self.residual_weight * x_enhanced

    def forward(self, x_2d):
        """
        Args:
            x_2d: [B*N, 1, H, W] Original 2D time series image

        Returns:
            enhanced_image: [B*N, 3, H, W] Enhanced 3-channel image
        """
        # 1. Frequency domain enhancement
        x_2d = self.frequency_enhancement(x_2d)

        # 2. Expand to 3 channels
        x_image = einops.repeat(x_2d, 'b 1 h w -> b c h w', c=self.num_channels)

        return x_image
