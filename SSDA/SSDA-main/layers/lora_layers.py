import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """
    LoRA-wrapped Linear layer
    """
    def __init__(self, linear_layer, r=4, lora_alpha=16, lora_dropout=0.1):
        super().__init__()
        self.linear = linear_layer
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r

        # Save original weights
        in_features = linear_layer.in_features
        out_features = linear_layer.out_features

        # LoRA parameters: A and B
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        # Dropout
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()

        # Initialization
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        # Freeze original layer
        for param in self.linear.parameters():
            param.requires_grad = False

    def forward(self, x):
        # Original output
        output = self.linear(x)

        # LoRA output
        x_dropout = self.lora_dropout(x)
        lora_output = F.linear(x_dropout, self.lora_B @ self.lora_A) * self.scaling

        return output + lora_output


class LoRAConv2d(nn.Module):
    """
    LoRA-wrapped Conv2d layer
    Uses a simpler approach: treating Conv2d as multiple independent convolution kernels
    """
    def __init__(self, conv_layer, r=4, lora_alpha=16, lora_dropout=0.1):
        super().__init__()
        self.conv = conv_layer
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r

        # Save original weights
        in_channels = conv_layer.in_channels
        out_channels = conv_layer.out_channels
        kernel_size = conv_layer.kernel_size

        # LoRA parameters: A and B
        # For Conv2d, we use a simpler approach:
        # A: [r, in_channels, kernel_h, kernel_w] - low-rank convolution kernels
        # B: [out_channels, r] - output projection
        self.lora_A = nn.Parameter(torch.zeros(r, in_channels, kernel_size[0], kernel_size[1]))
        self.lora_B = nn.Parameter(torch.zeros(out_channels, r))

        # Dropout
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()

        # Initialization
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        # Freeze original layer
        for param in self.conv.parameters():
            param.requires_grad = False

    def forward(self, x):
        # Original output
        output = self.conv(x)

        # LoRA output
        x_dropout = self.lora_dropout(x)

        # Step 1: Convolve input with A, get [B, r, H', W']
        lora_intermediate = F.conv2d(
            x_dropout,
            self.lora_A,
            bias=None,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=1  # LoRA does not use groups
        )

        # Step 2: Reshape [B, r, H', W'] to [B*H'*W', r], then project through B to [B*H'*W', out_channels]
        B, r, H, W = lora_intermediate.shape
        lora_intermediate = lora_intermediate.permute(0, 2, 3, 1).reshape(-1, r)  # [B*H'*W', r]
        lora_output = F.linear(lora_intermediate, self.lora_B)  # [B*H'*W', out_channels]
        lora_output = lora_output.reshape(B, H, W, -1).permute(0, 3, 1, 2)  # [B, out_channels, H', W']

        lora_output = lora_output * self.scaling

        return output + lora_output


def apply_lora_to_model(model, target_modules, r=4, lora_alpha=16, lora_dropout=0.1):
    """
    Replace specified layers in the model with LoRA layers

    Args:
        model: Model to apply LoRA to
        target_modules: List of module names to replace (e.g., ["qkv", "proj"])
        r: LoRA rank
        lora_alpha: LoRA alpha parameter
        lora_dropout: LoRA dropout rate

    Returns:
        Modified model
    """
    def replace_module(parent, name, module, target_modules, r, lora_alpha, lora_dropout):
        """Recursively replace modules"""
        for child_name, child_module in list(module.named_children()):
            full_name = f"{name}.{child_name}" if name else child_name

            # Check if module name matches target (supports partial matching, e.g., "qkv" matches "attn.qkv")
            if any(target in full_name or target in child_name for target in target_modules):
                # Check module type
                if isinstance(child_module, nn.Linear):
                    # Replace with LoRA Linear
                    lora_module = LoRALinear(child_module, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
                    setattr(module, child_name, lora_module)
                    # print(f"  Replaced {full_name} (Linear) with LoRA")
                elif isinstance(child_module, nn.Conv2d):
                    # Replace with LoRA Conv2d
                    lora_module = LoRAConv2d(child_module, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
                    setattr(module, child_name, lora_module)
                    # print(f"  Replaced {full_name} (Conv2d) with LoRA")
                else:
                    # Recursively process submodules
                    replace_module(module, full_name, child_module, target_modules, r, lora_alpha, lora_dropout)
            else:
                # Recursively process submodules
                replace_module(module, full_name, child_module, target_modules, r, lora_alpha, lora_dropout)

    # print(f"Applying LoRA to model with target_modules={target_modules}, r={r}, alpha={lora_alpha}")
    replace_module(model, "", model, target_modules, r, lora_alpha, lora_dropout)
    return model
