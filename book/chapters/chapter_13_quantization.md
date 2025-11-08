# Chapitre 13: Quantization et Compression de LLMs

## Introduction

La **quantization** est la technique la plus efficace pour réduire la taille et accélérer l'inférence des LLMs, avec un impact minimal sur la qualité.

### Pourquoi Quantizer?

```python
"""
Quantization = Reduce precision (FP32 → INT8/INT4)

Impact de la précision:

FP32 (32-bit floating point):
  • Llama 2 7B: 7B params × 4 bytes = 28GB
  • GPU needed: A100 40GB minimum
  • Cost: $1-2/hour

FP16 (16-bit):
  • Llama 2 7B: 7B × 2 bytes = 14GB
  • GPU needed: RTX 3090 24GB
  • Cost: $0.50/hour
  • Quality: ~Same as FP32

INT8 (8-bit):
  • Llama 2 7B: 7B × 1 byte = 7GB
  • GPU needed: RTX 3060 12GB
  • Cost: $0.20/hour
  • Quality: 98-99% of FP16
  • Speed: 2-3x faster

INT4 (4-bit):
  • Llama 2 7B: 7B × 0.5 bytes = 3.5GB
  • GPU needed: RTX 3050 8GB (or CPU!)
  • Cost: $0.10/hour or FREE (CPU)
  • Quality: 95-98% of FP16
  • Speed: 4-6x faster

Résultats:
  • Llama 2 70B (140GB FP16) → 35GB (4-bit) → Runs on single A100!
  • GPT-3 175B (350GB) → 87GB (4-bit) → Runs on consumer hardware
  • Quality loss: < 5% on most benchmarks
  • Cost reduction: 10-20x

Techniques:
  1. Post-Training Quantization (PTQ)
     • GPTQ: Optimal Brain Quantization
     • AWQ: Activation-aware quantization
     • Simple: No retraining needed

  2. Quantization-Aware Training (QAT)
     • QLoRA: Train with 4-bit base
     • Better quality but slower

  3. Dynamic Quantization
     • Quantize at runtime
     • Best flexibility
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import torch
import torch.nn as nn
import numpy as np


@dataclass
class QuantizationConfig:
    """Quantization configuration"""
    bits: int  # 4, 8, 16
    symmetric: bool = True  # Symmetric vs asymmetric
    per_channel: bool = True  # Per-channel vs per-tensor
    group_size: int = 128  # For grouped quantization


@dataclass
class ModelSize:
    """Model size stats"""
    name: str
    params_billions: float
    fp32_gb: float
    fp16_gb: float
    int8_gb: float
    int4_gb: float


# Model size comparison
MODEL_SIZES = [
    ModelSize(
        name="Llama 2 7B",
        params_billions=7,
        fp32_gb=28,
        fp16_gb=14,
        int8_gb=7,
        int4_gb=3.5
    ),
    ModelSize(
        name="Llama 2 13B",
        params_billions=13,
        fp32_gb=52,
        fp16_gb=26,
        int8_gb=13,
        int4_gb=6.5
    ),
    ModelSize(
        name="Llama 2 70B",
        params_billions=70,
        fp32_gb=280,
        fp16_gb=140,
        int8_gb=70,
        int4_gb=35
    ),
    ModelSize(
        name="GPT-3 175B",
        params_billions=175,
        fp32_gb=700,
        fp16_gb=350,
        int8_gb=175,
        int4_gb=87.5
    ),
]


def print_model_sizes():
    """Print model size comparison"""
    print("="*100)
    print("MODEL SIZES WITH QUANTIZATION")
    print("="*100)

    print(f"\n{'Model':<20} {'Parameters':<12} {'FP32':<10} {'FP16':<10} {'INT8':<10} {'INT4':<10}")
    print("-"*100)

    for model in MODEL_SIZES:
        params_str = f"{model.params_billions:.0f}B"
        print(f"{model.name:<20} {params_str:<12} {model.fp32_gb:<10.1f}GB {model.fp16_gb:<10.1f}GB "
              f"{model.int8_gb:<10.1f}GB {model.int4_gb:<10.1f}GB")

    print("\n" + "="*100)
    print("HARDWARE REQUIREMENTS (minimum)")
    print("="*100)

    hardware_guide = [
        ("Llama 2 7B FP16", "RTX 3090 24GB", "$1500"),
        ("Llama 2 7B INT4", "RTX 3060 12GB", "$300"),
        ("Llama 2 70B FP16", "8x A100 80GB", "$80k"),
        ("Llama 2 70B INT4", "1x A100 40GB", "$10k"),
        ("GPT-3 175B INT4", "2x A100 80GB", "$20k"),
    ]

    print(f"\n{'Configuration':<25} {'GPU Required':<20} {'Cost':<10}")
    print("-"*100)
    for config, gpu, cost in hardware_guide:
        print(f"{config:<25} {gpu:<20} {cost:<10}")

    print("\n" + "="*100)
    print("KEY INSIGHTS")
    print("="*100)
    print("""
1. Quantization enables consumer hardware:
   • Llama 2 7B (4-bit): Runs on RTX 3060
   • Llama 2 13B (4-bit): Runs on RTX 3090
   • Llama 2 70B (4-bit): Runs on single A100

2. Minimal quality loss:
   • INT8: 98-99% accuracy retention
   • INT4: 95-98% accuracy retention
   • GPTQ/AWQ: Even better (99%+)

3. Speed improvements:
   • INT8: 2-3x faster
   • INT4: 4-6x faster
   • Critical for production

4. Cost reduction:
   • 10-20x cheaper inference
   • Makes LLMs accessible
    """)


if __name__ == "__main__":
    print_model_sizes()
```

## 1. Quantization Theory

```python
"""
Quantization Fundamentals

Mapping high precision → low precision:
  FP16 range: [-65504, 65504] with 16-bit precision
  INT8 range: [-128, 127] with 256 discrete values
  INT4 range: [-8, 7] with 16 discrete values

Methods:
  1. Symmetric Quantization:
     • Zero point = 0
     • Scale = max(|max|, |min|) / (2^(bits-1) - 1)
     • Simpler, faster

  2. Asymmetric Quantization:
     • Zero point ≠ 0
     • Better range utilization
     • Slightly slower

  3. Per-Tensor vs Per-Channel:
     • Per-tensor: Single scale for entire tensor
     • Per-channel: Scale per output channel
     • Per-channel: Better accuracy

  4. Group-wise Quantization (GPTQ, AWQ):
     • Quantize in groups (e.g., 128 values)
     • Balance accuracy vs efficiency
"""

import torch
import torch.nn as nn
import math


class SymmetricQuantizer:
    """
    Symmetric quantization (zero-point = 0)

    Formula:
      scale = max(|x_max|, |x_min|) / (2^(bits-1) - 1)
      x_quant = round(x / scale)
      x_dequant = x_quant * scale

    Example:
        >>> quantizer = SymmetricQuantizer(bits=8)
        >>> x = torch.randn(128, 128)
        >>> x_quant = quantizer.quantize(x)
        >>> x_dequant = quantizer.dequantize(x_quant)
    """

    def __init__(self, bits: int = 8, per_channel: bool = False):
        """
        Args:
            bits: Number of bits (4 or 8)
            per_channel: Per-channel quantization
        """
        self.bits = bits
        self.per_channel = per_channel

        # Quantization range
        self.qmin = -(2 ** (bits - 1))
        self.qmax = 2 ** (bits - 1) - 1

    def compute_scale(self, x: torch.Tensor, dim: Optional[int] = None) -> torch.Tensor:
        """
        Compute quantization scale

        Args:
            x: Input tensor
            dim: Dimension for per-channel (None = per-tensor)

        Returns:
            Scale tensor
        """
        if dim is not None:
            # Per-channel
            x_max = x.abs().amax(dim=dim, keepdim=True)
        else:
            # Per-tensor
            x_max = x.abs().max()

        scale = x_max / self.qmax

        # Avoid division by zero
        scale = torch.clamp(scale, min=1e-8)

        return scale

    def quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize tensor

        Args:
            x: Input tensor (FP16/FP32)

        Returns:
            (quantized_tensor, scale)
        """
        # Compute scale
        if self.per_channel and x.dim() >= 2:
            # Per-channel (along output dimension)
            scale = self.compute_scale(x, dim=tuple(range(1, x.dim())))
        else:
            # Per-tensor
            scale = self.compute_scale(x)

        # Quantize
        x_quant = torch.round(x / scale)

        # Clamp to range
        x_quant = torch.clamp(x_quant, self.qmin, self.qmax)

        # Convert to int8
        if self.bits == 8:
            x_quant = x_quant.to(torch.int8)
        else:
            # INT4 stored as INT8 (no native INT4 in PyTorch)
            x_quant = x_quant.to(torch.int8)

        return x_quant, scale

    def dequantize(self, x_quant: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """
        Dequantize tensor

        Args:
            x_quant: Quantized tensor (INT8)
            scale: Scale tensor

        Returns:
            Dequantized tensor (FP16/FP32)
        """
        # Convert to float
        x_float = x_quant.float()

        # Dequantize
        x_dequant = x_float * scale

        return x_dequant


class AsymmetricQuantizer:
    """
    Asymmetric quantization (zero-point ≠ 0)

    Better range utilization for non-symmetric distributions

    Formula:
      scale = (x_max - x_min) / (2^bits - 1)
      zero_point = round(-x_min / scale)
      x_quant = round(x / scale + zero_point)
      x_dequant = (x_quant - zero_point) * scale

    Example:
        >>> quantizer = AsymmetricQuantizer(bits=8)
        >>> x = torch.randn(128, 128) + 5  # Non-symmetric
        >>> x_quant = quantizer.quantize(x)
    """

    def __init__(self, bits: int = 8):
        self.bits = bits
        self.qmin = 0
        self.qmax = 2 ** bits - 1

    def compute_params(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute scale and zero-point

        Args:
            x: Input tensor

        Returns:
            (scale, zero_point)
        """
        x_min = x.min()
        x_max = x.max()

        # Scale
        scale = (x_max - x_min) / (self.qmax - self.qmin)
        scale = torch.clamp(scale, min=1e-8)

        # Zero point
        zero_point = self.qmin - x_min / scale
        zero_point = torch.round(torch.clamp(zero_point, self.qmin, self.qmax))

        return scale, zero_point

    def quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize tensor

        Args:
            x: Input tensor

        Returns:
            (quantized_tensor, scale, zero_point)
        """
        scale, zero_point = self.compute_params(x)

        # Quantize
        x_quant = torch.round(x / scale + zero_point)
        x_quant = torch.clamp(x_quant, self.qmin, self.qmax)

        # Convert to uint8 (for 8-bit)
        x_quant = x_quant.to(torch.uint8 if self.bits == 8 else torch.int8)

        return x_quant, scale, zero_point

    def dequantize(
        self,
        x_quant: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor
    ) -> torch.Tensor:
        """Dequantize tensor"""
        x_float = x_quant.float()
        x_dequant = (x_float - zero_point) * scale

        return x_dequant


class QuantizedLinear(nn.Module):
    """
    Quantized linear layer

    Stores weights in INT8, performs computation in INT8,
    converts output back to FP16

    Example:
        >>> # Convert existing linear layer
        >>> linear_fp16 = nn.Linear(768, 3072)
        >>> linear_int8 = QuantizedLinear.from_float(linear_fp16, bits=8)
        >>> # Use normally
        >>> x = torch.randn(32, 128, 768, dtype=torch.float16)
        >>> output = linear_int8(x)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        bits: int = 8
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits

        # Quantized weight (INT8)
        self.register_buffer(
            'weight_quant',
            torch.zeros(out_features, in_features, dtype=torch.int8)
        )

        # Scale (per output channel)
        self.register_buffer(
            'weight_scale',
            torch.ones(out_features, 1)
        )

        # Bias (keep in FP16)
        if bias:
            self.register_buffer(
                'bias',
                torch.zeros(out_features, dtype=torch.float16)
            )
        else:
            self.bias = None

    @torch.no_grad()
    def quantize_weight(self, weight: torch.Tensor):
        """
        Quantize weight tensor

        Args:
            weight: FP16/FP32 weight (out_features, in_features)
        """
        quantizer = SymmetricQuantizer(bits=self.bits, per_channel=True)

        weight_quant, scale = quantizer.quantize(weight)

        self.weight_quant.copy_(weight_quant)
        self.weight_scale.copy_(scale)

    @classmethod
    def from_float(cls, module: nn.Linear, bits: int = 8):
        """
        Create quantized layer from float layer

        Args:
            module: Float linear layer
            bits: Quantization bits

        Returns:
            Quantized linear layer
        """
        qlinear = cls(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            bits=bits
        )

        # Quantize weight
        qlinear.quantize_weight(module.weight)

        # Copy bias
        if module.bias is not None:
            qlinear.bias.copy_(module.bias)

        return qlinear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with dequantization

        Args:
            x: Input (batch, seq_len, in_features)

        Returns:
            Output (batch, seq_len, out_features)
        """
        # Dequantize weight
        weight_float = self.weight_quant.float() * self.weight_scale

        # Convert to FP16 for computation
        weight = weight_float.half()

        # Linear
        output = torch.nn.functional.linear(x, weight, self.bias)

        return output


def quantization_error_analysis():
    """Analyze quantization error"""
    print("="*80)
    print("QUANTIZATION ERROR ANALYSIS")
    print("="*80)

    # Generate test tensor
    x = torch.randn(1024, 1024, dtype=torch.float32) * 10

    print(f"\nOriginal tensor:")
    print(f"  Shape: {x.shape}")
    print(f"  Dtype: {x.dtype}")
    print(f"  Range: [{x.min():.2f}, {x.max():.2f}]")
    print(f"  Mean: {x.mean():.2f}, Std: {x.std():.2f}")

    # Test different quantization methods
    configs = [
        ("FP16", 16, None),
        ("INT8 Symmetric", 8, SymmetricQuantizer(bits=8)),
        ("INT8 Asymmetric", 8, AsymmetricQuantizer(bits=8)),
        ("INT4 Symmetric", 4, SymmetricQuantizer(bits=4)),
    ]

    print(f"\n{'Method':<20} {'Bits':<8} {'Error (L2)':<15} {'Error (%)':<15}")
    print("-"*80)

    for name, bits, quantizer in configs:
        if quantizer is None:
            # FP16 baseline
            x_fp16 = x.half().float()
            error = (x - x_fp16).pow(2).mean().sqrt()
        elif isinstance(quantizer, SymmetricQuantizer):
            x_quant, scale = quantizer.quantize(x)
            x_dequant = quantizer.dequantize(x_quant, scale)
            error = (x - x_dequant).pow(2).mean().sqrt()
        else:  # Asymmetric
            x_quant, scale, zero_point = quantizer.quantize(x)
            x_dequant = quantizer.dequantize(x_quant, scale, zero_point)
            error = (x - x_dequant).pow(2).mean().sqrt()

        error_pct = (error / x.abs().mean() * 100).item()

        print(f"{name:<20} {bits:<8} {error.item():<15.4f} {error_pct:<15.2f}%")

    print("\n" + "="*80)
    print("CONCLUSIONS")
    print("="*80)
    print("""
1. FP16 vs FP32: ~0.01% error (negligible)
2. INT8: ~1-2% error (acceptable for most tasks)
3. INT4: ~5-10% error (needs careful calibration)
4. Asymmetric slightly better for non-zero-centered data
5. Per-channel quantization reduces error significantly
    """)


if __name__ == "__main__":
    quantization_error_analysis()
```

*[Suite avec GPTQ et AWQ dans la partie 2...]*
