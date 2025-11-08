# Chapitre 8 (Suite 2): QLoRA et Quantization

## 3. QLoRA: Quantized Low-Rank Adaptation

### 3.1 Introduction à la Quantization

```python
"""
Quantization: Réduire la précision des poids pour économiser la mémoire

Paper: "QLoRA: Efficient Finetuning of Quantized LLMs"
Authors: Dettmers et al. (University of Washington), 2023

Révolution: Fine-tune Llama 2 70B sur 1x RTX 3090 24GB!
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from enum import Enum


class QuantizationType(Enum):
    """Types de quantization"""
    FP32 = "fp32"  # 32-bit float (4 bytes)
    FP16 = "fp16"  # 16-bit float (2 bytes)
    BF16 = "bf16"  # Brain Float 16 (2 bytes)
    INT8 = "int8"  # 8-bit integer (1 byte)
    INT4 = "int4"  # 4-bit integer (0.5 bytes)
    NF4 = "nf4"    # 4-bit NormalFloat (0.5 bytes) - QLoRA


class QuantizationExplainer:
    """Explique les différents types de quantization"""

    @staticmethod
    def explain_precision_types():
        """Explique les types de précision"""

        return """
        === Types de Précision ===

        1. FP32 (Float 32-bit) - Standard
           - Range: ±3.4 × 10^38
           - Precision: ~7 decimal digits
           - Size: 4 bytes
           - Use: Training from scratch, highest precision

        2. FP16 (Float 16-bit) - Half Precision
           - Range: ±65,504
           - Precision: ~3 decimal digits
           - Size: 2 bytes
           - Use: Mixed precision training, inference
           - Issue: Peut overflow/underflow

        3. BF16 (BFloat16) - Brain Float
           - Range: ±3.4 × 10^38 (same as FP32!)
           - Precision: ~2 decimal digits
           - Size: 2 bytes
           - Use: Training, preferred over FP16
           - Advantage: Même range que FP32, moins d'overflow

        4. INT8 (Integer 8-bit)
           - Range: -128 to 127 (signed) or 0 to 255 (unsigned)
           - Size: 1 byte
           - Use: Inference optimization
           - Requires: Careful calibration

        5. INT4 / NF4 (4-bit)
           - Range: 16 unique values (INT4) or quantiles (NF4)
           - Size: 0.5 bytes
           - Use: Extreme compression (QLoRA)
           - Requires: Very careful quantization scheme

        Memory savings:
        - FP32 → FP16/BF16: 2x reduction
        - FP32 → INT8: 4x reduction
        - FP32 → INT4/NF4: 8x reduction 🚀
        """

    @staticmethod
    def calculate_memory_savings(
        num_parameters: int,
        original_dtype: str = "fp32",
        quantized_dtype: str = "nf4"
    ) -> dict:
        """Calcule les économies de mémoire"""

        bytes_per_param = {
            "fp32": 4,
            "fp16": 2,
            "bf16": 2,
            "int8": 1,
            "int4": 0.5,
            "nf4": 0.5
        }

        original_bytes = bytes_per_param[original_dtype]
        quantized_bytes = bytes_per_param[quantized_dtype]

        original_memory_gb = (num_parameters * original_bytes) / (1024**3)
        quantized_memory_gb = (num_parameters * quantized_bytes) / (1024**3)

        savings_gb = original_memory_gb - quantized_memory_gb
        savings_pct = (savings_gb / original_memory_gb) * 100

        return {
            "original_memory_gb": round(original_memory_gb, 2),
            "quantized_memory_gb": round(quantized_memory_gb, 2),
            "savings_gb": round(savings_gb, 2),
            "savings_pct": round(savings_pct, 1),
            "compression_ratio": f"{int(original_bytes / quantized_bytes)}x"
        }


# Exemple
if __name__ == "__main__":
    print("="*60)
    print("QUANTIZATION EXPLAINER")
    print("="*60)
    print()

    print(QuantizationExplainer.explain_precision_types())

    print("\n" + "="*60)
    print("MEMORY SAVINGS")
    print("="*60)
    print()

    # Exemples pour différents modèles
    models = {
        "Llama 2 7B": 7_000_000_000,
        "Llama 2 13B": 13_000_000_000,
        "Llama 2 70B": 70_000_000_000,
    }

    for model_name, num_params in models.items():
        print(f"\n### {model_name} ({num_params/1e9:.0f}B parameters)")

        # FP32 → FP16
        fp16_savings = QuantizationExplainer.calculate_memory_savings(
            num_params, "fp32", "fp16"
        )

        # FP32 → INT8
        int8_savings = QuantizationExplainer.calculate_memory_savings(
            num_params, "fp32", "int8"
        )

        # FP32 → NF4
        nf4_savings = QuantizationExplainer.calculate_memory_savings(
            num_params, "fp32", "nf4"
        )

        print(f"\nFP32 baseline: {fp16_savings['original_memory_gb']} GB")
        print(f"\nFP16:  {fp16_savings['quantized_memory_gb']} GB "
              f"({fp16_savings['compression_ratio']} compression, "
              f"{fp16_savings['savings_pct']:.0f}% savings)")
        print(f"INT8:  {int8_savings['quantized_memory_gb']} GB "
              f"({int8_savings['compression_ratio']} compression, "
              f"{int8_savings['savings_pct']:.0f}% savings)")
        print(f"NF4:   {nf4_savings['quantized_memory_gb']} GB "
              f"({nf4_savings['compression_ratio']} compression, "
              f"{nf4_savings['savings_pct']:.0f}% savings) 🚀")
```

### 3.2 NF4: 4-bit NormalFloat

```python
"""
NF4 (4-bit NormalFloat): Innovation clé de QLoRA

Observation: Les poids de neural networks suivent souvent
une distribution normale centrée.

NF4: Quantize de manière optimale pour une distribution normale!
"""

import torch
import numpy as np


class NF4Quantizer:
    """
    NF4 Quantizer

    Principe:
    - Diviser la distribution normale en 16 buckets de probabilité égale
    - Chaque bucket représente une valeur NF4
    - Optimise pour zero-mean normal distribution
    """

    # Les 16 valeurs NF4 (calculées pour N(0,1))
    # Ces valeurs sont les quantiles de la distribution normale
    NF4_QUANT_LEVELS = torch.tensor([
        -1.0,
        -0.6961928009986877,
        -0.5250730514526367,
        -0.39491748809814453,
        -0.28444138169288635,
        -0.18477343022823334,
        -0.09105003625154495,
        0.0,
        0.07958029955625534,
        0.16093020141124725,
        0.24611230194568634,
        0.33791524171829224,
        0.44070982933044434,
        0.5626170039176941,
        0.7229568362236023,
        1.0,
    ])

    @staticmethod
    def quantize(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize tensor en NF4

        Args:
            tensor: Tensor FP32/FP16 à quantizer

        Returns:
            quantized: Indices NF4 (4-bit, stockés en int8)
            scale: Scale factor pour dequantization
        """

        # Compute scale (max absolute value)
        scale = tensor.abs().max()

        # Normalize to [-1, 1]
        normalized = tensor / (scale + 1e-8)

        # Find nearest NF4 level for each value
        # Broadcasting: (N,) vs (16,) → (N, 16)
        distances = torch.abs(
            normalized.unsqueeze(-1) - NF4Quantizer.NF4_QUANT_LEVELS
        )

        # Get index of nearest level
        quantized = torch.argmin(distances, dim=-1).to(torch.uint8)

        return quantized, scale

    @staticmethod
    def dequantize(
        quantized: torch.Tensor,
        scale: torch.Tensor
    ) -> torch.Tensor:
        """
        Dequantize NF4 vers FP16/FP32

        Args:
            quantized: Indices NF4 (int8)
            scale: Scale factor

        Returns:
            Dequantized tensor
        """

        # Map indices to NF4 values
        nf4_values = NF4Quantizer.NF4_QUANT_LEVELS[quantized]

        # Scale back
        dequantized = nf4_values * scale

        return dequantized

    @staticmethod
    def quantize_per_block(
        tensor: torch.Tensor,
        block_size: int = 64
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize avec blocage (plus précis)

        Au lieu d'un seul scale pour tout le tensor,
        utilise un scale par block

        Args:
            tensor: Tensor à quantizer
            block_size: Taille des blocks

        Returns:
            quantized: Indices NF4
            scales: Scales per block
        """

        original_shape = tensor.shape
        numel = tensor.numel()

        # Flatten et pad si nécessaire
        flat = tensor.flatten()
        pad_size = (block_size - numel % block_size) % block_size
        if pad_size > 0:
            flat = torch.cat([flat, torch.zeros(pad_size, device=flat.device)])

        # Reshape en blocks
        blocks = flat.view(-1, block_size)

        # Quantize chaque block
        quantized_blocks = []
        scales = []

        for block in blocks:
            q, s = NF4Quantizer.quantize(block)
            quantized_blocks.append(q)
            scales.append(s)

        quantized = torch.cat(quantized_blocks)
        scales = torch.tensor(scales)

        # Remove padding
        if pad_size > 0:
            quantized = quantized[:-pad_size]

        return quantized.view(original_shape), scales

    @staticmethod
    def dequantize_per_block(
        quantized: torch.Tensor,
        scales: torch.Tensor,
        block_size: int = 64
    ) -> torch.Tensor:
        """Dequantize block-wise"""

        original_shape = quantized.shape
        flat = quantized.flatten()

        # Reconstruct blocks
        pad_size = (block_size - flat.numel() % block_size) % block_size
        if pad_size > 0:
            flat = torch.cat([flat, torch.zeros(pad_size, dtype=flat.dtype, device=flat.device)])

        blocks = flat.view(-1, block_size)

        # Dequantize each block
        dequantized_blocks = []
        for block, scale in zip(blocks, scales):
            deq = NF4Quantizer.dequantize(block, scale)
            dequantized_blocks.append(deq)

        dequantized = torch.cat(dequantized_blocks)

        # Remove padding
        if pad_size > 0:
            dequantized = dequantized[:-pad_size]

        return dequantized.view(original_shape)


# Exemple
if __name__ == "__main__":
    print("\n=== NF4 Quantization Demo ===\n")

    # Create sample weight matrix (simulating neural network weights)
    torch.manual_seed(42)
    weights = torch.randn(256, 256) * 0.1  # Normal distribution

    print(f"Original weights shape: {weights.shape}")
    print(f"Original dtype: {weights.dtype}")
    print(f"Original memory: {weights.numel() * 4 / 1024:.2f} KB (FP32)")

    # Quantize
    quantized, scales = NF4Quantizer.quantize_per_block(weights, block_size=64)

    print(f"\nQuantized dtype: {quantized.dtype}")
    quant_memory = quantized.numel() * 0.5 + scales.numel() * 4  # 0.5 bytes per NF4 + FP32 scales
    print(f"Quantized memory: {quant_memory / 1024:.2f} KB")
    print(f"Compression ratio: {(weights.numel() * 4) / quant_memory:.1f}x")

    # Dequantize
    dequantized = NF4Quantizer.dequantize_per_block(quantized, scales, block_size=64)

    # Measure error
    error = (weights - dequantized).abs().mean()
    relative_error = error / weights.abs().mean()

    print(f"\nReconstruction error:")
    print(f"  Absolute: {error:.6f}")
    print(f"  Relative: {relative_error:.4f} ({relative_error*100:.2f}%)")

    # Visualize distribution
    print(f"\nOriginal stats:")
    print(f"  Mean: {weights.mean():.4f}")
    print(f"  Std:  {weights.std():.4f}")
    print(f"  Min:  {weights.min():.4f}")
    print(f"  Max:  {weights.max():.4f}")

    print(f"\nDequantized stats:")
    print(f"  Mean: {dequantized.mean():.4f}")
    print(f"  Std:  {dequantized.std():.4f}")
    print(f"  Min:  {dequantized.min():.4f}")
    print(f"  Max:  {dequantized.max():.4f}")
```

### 3.3 Double Quantization

```python
"""
Double Quantization: Quantize les scales aussi!

Innovation de QLoRA: Les scales eux-mêmes peuvent être quantizés

Gains additionnels de mémoire de ~0.4 bits par paramètre
"""


class DoubleQuantization:
    """
    Double Quantization pour QLoRA

    Quantize non seulement les poids, mais aussi les scale factors!
    """

    @staticmethod
    def double_quantize(
        tensor: torch.Tensor,
        block_size: int = 64,
        scale_block_size: int = 256
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply double quantization

        1. Quantize weights → get scales
        2. Quantize scales → get second-level scales

        Args:
            tensor: Weight tensor
            block_size: Block size for weight quantization
            scale_block_size: Block size for scale quantization

        Returns:
            quantized_weights: NF4 quantized weights
            quantized_scales: INT8 quantized scales
            second_level_scales: FP32 scales for scales
        """

        # First level: Quantize weights
        quantized_weights, scales = NF4Quantizer.quantize_per_block(
            tensor, block_size=block_size
        )

        # Second level: Quantize scales
        # Scales are already much smaller, can use INT8
        scale_max = scales.abs().max()
        normalized_scales = scales / (scale_max + 1e-8)

        # Quantize to INT8 [-127, 127]
        quantized_scales = (normalized_scales * 127).round().to(torch.int8)

        second_level_scales = scale_max

        return quantized_weights, quantized_scales, second_level_scales

    @staticmethod
    def double_dequantize(
        quantized_weights: torch.Tensor,
        quantized_scales: torch.Tensor,
        second_level_scales: torch.Tensor,
        block_size: int = 64
    ) -> torch.Tensor:
        """
        Dequantize with double quantization

        1. Dequantize scales
        2. Dequantize weights using dequantized scales
        """

        # Dequantize scales
        scales = (quantized_scales.float() / 127.0) * second_level_scales

        # Dequantize weights
        weights = NF4Quantizer.dequantize_per_block(
            quantized_weights, scales, block_size=block_size
        )

        return weights

    @staticmethod
    def calculate_memory(
        num_params: int,
        block_size: int = 64
    ) -> dict:
        """
        Calcule la mémoire requise avec double quantization

        Args:
            num_params: Nombre de paramètres
            block_size: Taille des blocks

        Returns:
            Dict avec breakdown mémoire
        """

        num_blocks = (num_params + block_size - 1) // block_size

        # Weights: 4 bits (0.5 bytes) per param
        weights_bytes = num_params * 0.5

        # First-level scales: INT8 (1 byte) per block
        scales_bytes = num_blocks * 1

        # Second-level scales: FP32 (4 bytes)
        # One scale per group of blocks (e.g., 256 blocks)
        scale_block_size = 256
        num_scale_blocks = (num_blocks + scale_block_size - 1) // scale_block_size
        second_scales_bytes = num_scale_blocks * 4

        total_bytes = weights_bytes + scales_bytes + second_scales_bytes
        total_gb = total_bytes / (1024**3)

        # Compare avec FP32
        fp32_gb = (num_params * 4) / (1024**3)

        return {
            "weights_gb": round(weights_bytes / (1024**3), 3),
            "scales_gb": round(scales_bytes / (1024**3), 3),
            "second_scales_gb": round(second_scales_bytes / (1024**3), 3),
            "total_gb": round(total_gb, 3),
            "fp32_gb": round(fp32_gb, 2),
            "compression_ratio": round(fp32_gb / total_gb, 2),
            "bits_per_param": round((total_bytes * 8) / num_params, 2)
        }


# Exemple
if __name__ == "__main__":
    print("\n=== Double Quantization ===\n")

    # Test sur Llama 2 7B
    num_params = 7_000_000_000

    memory = DoubleQuantization.calculate_memory(num_params, block_size=64)

    print(f"Llama 2 7B ({num_params/1e9:.0f}B parameters)")
    print(f"\nMemory breakdown:")
    print(f"  Weights (NF4):           {memory['weights_gb']:.2f} GB")
    print(f"  Scales (INT8):           {memory['scales_gb']:.3f} GB")
    print(f"  Second-level scales:     {memory['second_scales_gb']:.4f} GB")
    print(f"  Total:                   {memory['total_gb']:.2f} GB")
    print(f"\nComparison:")
    print(f"  FP32:                    {memory['fp32_gb']:.2f} GB")
    print(f"  Compression ratio:       {memory['compression_ratio']:.1f}x")
    print(f"  Effective bits/param:    {memory['bits_per_param']:.2f} bits")

    print("\n🚀 Result: ~4.2 bits per parameter (vs 32 bits FP32)!")
    print("   → 7.6x compression with minimal quality loss")

    # Demo on small tensor
    print("\n" + "="*60)
    print("Demo on sample weights")
    print("="*60 + "\n")

    torch.manual_seed(42)
    weights = torch.randn(1024, 1024) * 0.1

    # Double quantize
    q_weights, q_scales, s_scales = DoubleQuantization.double_quantize(
        weights, block_size=64
    )

    # Calculate memory
    original_mem = weights.numel() * 4
    quantized_mem = (
        q_weights.numel() * 0.5 +  # NF4
        q_scales.numel() * 1 +      # INT8
        s_scales.numel() * 4        # FP32
    )

    print(f"Original memory: {original_mem / 1024:.2f} KB")
    print(f"Quantized memory: {quantized_mem / 1024:.2f} KB")
    print(f"Compression: {original_mem / quantized_mem:.2f}x")

    # Dequantize
    reconstructed = DoubleQuantization.double_dequantize(
        q_weights, q_scales, s_scales, block_size=64
    )

    # Error
    error = (weights - reconstructed).abs().mean()
    rel_error = error / weights.abs().mean()

    print(f"\nReconstruction error: {rel_error*100:.3f}%")
    print("✅ Très faible perte de qualité!")
```

*[Suite avec le projet pratique QLoRA dans la partie 4...]*
