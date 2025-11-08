# Chapitre 12 (Partie 2): TensorRT-LLM et CUDA Optimizations

## 2. TensorRT-LLM

```python
"""
TensorRT-LLM = NVIDIA's optimized inference engine

Key features:
  • Optimized CUDA kernels
  • Fused operations
  • INT8/FP8 quantization
  • Multi-GPU tensor parallelism
  • In-flight batching

Performance:
  • 2-8x faster than PyTorch
  • 10-20x faster than HuggingFace
  • Best for single-request latency

Installation:
  # Requires NVIDIA GPU with compute capability >= 8.0 (A100, H100)
  pip install tensorrt_llm
  # Or use NGC container

Architecture differences vs vLLM:
  vLLM:
    • Focus: Throughput optimization
    • Best for: Batch serving
    • PagedAttention: Memory efficiency

  TensorRT-LLM:
    • Focus: Latency optimization
    • Best for: Single-request speed
    • Fused kernels: Compute efficiency

When to use:
  ✅ TensorRT-LLM: Low latency critical (chatbots, real-time)
  ✅ vLLM: High throughput critical (batch processing)
  ✅ Both: Use vLLM with TensorRT backend (best of both!)
"""

from dataclasses import dataclass
from typing import List, Optional, Dict
import torch
import torch.nn as nn
import time


@dataclass
class KernelBenchmark:
    """Benchmark result for a kernel"""
    name: str
    operation: str
    baseline_ms: float
    optimized_ms: float
    speedup: float


# Kernel optimization benchmarks
KERNEL_BENCHMARKS = [
    KernelBenchmark(
        name="Attention (standard)",
        operation="Q @ K^T @ V",
        baseline_ms=12.5,
        optimized_ms=1.2,
        speedup=10.4
    ),
    KernelBenchmark(
        name="LayerNorm + Residual",
        operation="Fused LN + Add",
        baseline_ms=0.8,
        optimized_ms=0.15,
        speedup=5.3
    ),
    KernelBenchmark(
        name="GELU activation",
        operation="Fused GELU",
        baseline_ms=0.5,
        optimized_ms=0.08,
        speedup=6.25
    ),
    KernelBenchmark(
        name="Attention + Projection",
        operation="Fused Attn + Linear",
        baseline_ms=15.2,
        optimized_ms=2.1,
        speedup=7.2
    ),
]


def print_kernel_benchmarks():
    """Print kernel optimization benchmarks"""
    print("="*90)
    print("CUDA KERNEL OPTIMIZATIONS (Llama 2 7B, single layer)")
    print("="*90)

    print(f"\n{'Kernel':<25} {'Operation':<25} {'Baseline':<12} {'Optimized':<12} {'Speedup':<10}")
    print("-"*90)

    for bench in KERNEL_BENCHMARKS:
        print(f"{bench.name:<25} {bench.operation:<25} {bench.baseline_ms:<12.2f}ms "
              f"{bench.optimized_ms:<12.2f}ms {bench.speedup:<10.1f}x")

    total_baseline = sum(b.baseline_ms for b in KERNEL_BENCHMARKS)
    total_optimized = sum(b.optimized_ms for b in KERNEL_BENCHMARKS)
    overall_speedup = total_baseline / total_optimized

    print(f"\n{'Total per layer':<25} {'':<25} {total_baseline:<12.2f}ms "
          f"{total_optimized:<12.2f}ms {overall_speedup:<10.1f}x")

    print("\n" + "="*90)
    print("KEY TECHNIQUES")
    print("="*90)
    print("""
1. Kernel Fusion
   • Combine multiple ops into single kernel
   • Reduce memory transfers
   • Example: LayerNorm + Residual + Dropout → 1 kernel

2. Flash Attention
   • Tiled computation (fits in SRAM)
   • O(N) memory instead of O(N²)
   • 2-4x faster, exact same output

3. Memory Coalescing
   • Align memory accesses
   • Use shared memory efficiently
   • 2-3x bandwidth improvement

4. FP16/BF16 Tensor Cores
   • Use specialized hardware
   • 8x throughput vs FP32
   • Automatic mixed precision
    """)


if __name__ == "__main__":
    print_kernel_benchmarks()
```

## 2.1 Flash Attention

```python
"""
Flash Attention = Memory-efficient attention

Problem avec attention standard:
  • Attention matrix: O(N²) memory
  • Llama 2 (32 heads, 4096 context):
    32 × 4096 × 4096 × 2 bytes = 1GB per layer!
  • Ne rentre pas en SRAM (quelques MB)
  • Constant HBM ↔ SRAM transfers

Flash Attention solution:
  • Tiled computation
  • Compute attention in blocks
  • Fits in SRAM (fast memory)
  • Results: Same output, 2-4x faster, O(N) memory

Papers:
  • Flash Attention (Dao et al., 2022)
  • Flash Attention 2 (Dao, 2023)

Used by:
  • PyTorch 2.0+ (scaled_dot_product_attention)
  • vLLM
  • TensorRT-LLM
  • Megatron-LM
"""

import torch
import torch.nn.functional as F
import math
from typing import Optional


class StandardAttention(nn.Module):
    """
    Standard attention implementation (naive)

    Memory: O(N²) for attention matrix
    Speed: Slower due to HBM transfers

    Example:
        >>> attn = StandardAttention(d_model=768, num_heads=12)
        >>> x = torch.randn(2, 512, 768)  # (batch, seq_len, d_model)
        >>> output = attn(x)
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass (standard implementation)

        Args:
            x: Input (batch, seq_len, d_model)
            mask: Attention mask (optional)

        Returns:
            Output (batch, seq_len, d_model)
        """
        B, N, D = x.shape

        # Project Q, K, V
        q = self.q_proj(x)  # (B, N, D)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, d)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores: (B, H, N, N)
        # THIS IS THE MEMORY BOTTLENECK!
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum: (B, H, N, d)
        output = torch.matmul(attn_weights, v)

        # Concatenate heads
        output = output.transpose(1, 2).contiguous().view(B, N, D)

        # Output projection
        output = self.out_proj(output)

        return output


class FlashAttention(nn.Module):
    """
    Flash Attention implementation

    Uses PyTorch's scaled_dot_product_attention (PyTorch 2.0+)
    which implements Flash Attention automatically

    Example:
        >>> attn = FlashAttention(d_model=768, num_heads=12)
        >>> x = torch.randn(2, 512, 768)
        >>> output = attn(x)
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout_p = dropout

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = False
    ) -> torch.Tensor:
        """
        Forward pass (Flash Attention)

        Args:
            x: Input (batch, seq_len, d_model)
            mask: Attention mask (optional)
            is_causal: Use causal mask

        Returns:
            Output (batch, seq_len, d_model)
        """
        B, N, D = x.shape

        # Project Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Flash Attention (PyTorch 2.0+)
        # This automatically uses optimized kernels!
        output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=is_causal
        )

        # Concatenate heads
        output = output.transpose(1, 2).contiguous().view(B, N, D)

        # Output projection
        output = self.out_proj(output)

        return output


def benchmark_attention():
    """
    Benchmark standard vs Flash Attention

    Shows memory and speed differences
    """
    print("="*80)
    print("ATTENTION BENCHMARK")
    print("="*80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if device == 'cpu':
        print("⚠ Warning: CUDA not available, running on CPU (slow)")
        print("Flash Attention benefits only visible on GPU")

    # Config
    batch_size = 4
    seq_len = 2048
    d_model = 768
    num_heads = 12

    # Create models
    standard_attn = StandardAttention(d_model, num_heads).to(device)
    flash_attn = FlashAttention(d_model, num_heads).to(device)

    # Input
    x = torch.randn(batch_size, seq_len, d_model, device=device)

    # Warmup
    with torch.no_grad():
        _ = standard_attn(x)
        _ = flash_attn(x)

    if device == 'cuda':
        torch.cuda.synchronize()

    # Benchmark standard
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()

    num_iters = 10
    start = time.time()

    for _ in range(num_iters):
        with torch.no_grad():
            _ = standard_attn(x)

    if device == 'cuda':
        torch.cuda.synchronize()

    standard_time = (time.time() - start) / num_iters

    if device == 'cuda':
        standard_memory = torch.cuda.max_memory_allocated() / 1e9

    # Benchmark Flash
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()

    start = time.time()

    for _ in range(num_iters):
        with torch.no_grad():
            _ = flash_attn(x)

    if device == 'cuda':
        torch.cuda.synchronize()

    flash_time = (time.time() - start) / num_iters

    if device == 'cuda':
        flash_memory = torch.cuda.max_memory_allocated() / 1e9

    # Results
    print(f"\nConfiguration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Model dim: {d_model}")
    print(f"  Num heads: {num_heads}")

    print(f"\n{'Method':<20} {'Time (ms)':<15} {'Memory (GB)':<15} {'Speedup':<10}")
    print("-"*80)

    if device == 'cuda':
        print(f"{'Standard':<20} {standard_time*1000:<15.2f} {standard_memory:<15.2f} {'1.0x':<10}")
        print(f"{'Flash Attention':<20} {flash_time*1000:<15.2f} {flash_memory:<15.2f} "
              f"{standard_time/flash_time:<10.1f}x")

        print(f"\nMemory savings: {(1 - flash_memory/standard_memory)*100:.1f}%")
        print(f"Speed improvement: {standard_time/flash_time:.1f}x")
    else:
        print(f"{'Standard':<20} {standard_time*1000:<15.2f}")
        print(f"{'Flash Attention':<20} {flash_time*1000:<15.2f}")


if __name__ == "__main__":
    benchmark_attention()
```

## 2.2 Fused Kernels

```python
"""
Kernel Fusion = Combine operations into single kernel

Example: LayerNorm + Residual + Dropout

Standard (3 kernels):
  1. x_norm = LayerNorm(x)
  2. x_dropped = Dropout(x_norm)
  3. output = x + x_dropped

Fused (1 kernel):
  output = FusedLayerNormResidualDropout(x, residual)

Benefits:
  • 1 memory read instead of 3
  • 1 memory write instead of 3
  • 3-5x faster
  • Less memory bandwidth

Common fusions:
  • LayerNorm + Residual
  • GELU activation + Projection
  • Attention + Output projection
  • RMSNorm + Residual (Llama style)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class StandardLayerNorm(nn.Module):
    """
    Standard LayerNorm + Residual (unfused)

    3 separate operations:
      1. LayerNorm
      2. Dropout
      3. Residual add
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """
        Forward pass (unfused)

        Args:
            x: Input
            residual: Residual connection

        Returns:
            Output
        """
        # 3 separate kernel calls!
        x = self.norm(x)
        x = self.dropout(x)
        x = residual + x

        return x


class FusedLayerNormResidual(nn.Module):
    """
    Fused LayerNorm + Residual + Dropout

    In practice, use optimized libraries:
      • apex.normalization.FusedLayerNorm
      • torch.nn.functional.fused_norm (hypothetical)
      • Custom CUDA kernels

    This is a PyTorch approximation (still uses 3 ops)
    Real fusion requires CUDA kernel!

    Example:
        >>> fused = FusedLayerNormResidual(d_model=768)
        >>> x = torch.randn(2, 512, 768)
        >>> residual = torch.randn(2, 512, 768)
        >>> output = fused(x, residual)
    """

    def __init__(self, d_model: int, dropout: float = 0.1, eps: float = 1e-5):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.dropout_p = dropout
        self.eps = eps

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """
        Forward pass (conceptually fused)

        In real CUDA implementation, this would be single kernel

        Args:
            x: Input
            residual: Residual

        Returns:
            Output
        """
        # In real fused kernel, all of this happens in one pass:
        # 1. Compute mean/var
        # 2. Normalize
        # 3. Apply dropout
        # 4. Add residual
        # All in SRAM, no intermediate HBM writes!

        # PyTorch approximation (still separate ops):
        x = F.layer_norm(x, (x.size(-1),), self.weight, self.bias, self.eps)

        if self.training and self.dropout_p > 0:
            x = F.dropout(x, p=self.dropout_p, training=True)

        x = x + residual

        return x


class FusedGELU(nn.Module):
    """
    Fused GELU activation

    Standard GELU (slow):
      x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))

    Approximation (fast):
      x * sigmoid(1.702 * x)

    Fused with projection:
      output = GELU(x @ W)

    In single kernel instead of:
      1. Linear projection
      2. GELU activation
    """

    def __init__(self, approximate: str = "none"):
        super().__init__()
        self.approximate = approximate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """GELU activation"""
        if self.approximate == "tanh":
            # Fast approximation
            return 0.5 * x * (1 + torch.tanh(
                math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)
            ))
        else:
            # Exact (uses erf)
            return F.gelu(x)


class FusedMLPBlock(nn.Module):
    """
    Fused MLP block

    Fuses:
      • Linear projection
      • GELU activation
      • Linear projection
      • Dropout
      • Residual

    In optimized implementations (TensorRT-LLM), entire block
    is single fused kernel!

    Example:
        >>> mlp = FusedMLPBlock(d_model=768, d_ff=3072)
        >>> x = torch.randn(2, 512, 768)
        >>> output = mlp(x)
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        bias: bool = True
    ):
        super().__init__()

        self.fc1 = nn.Linear(d_model, d_ff, bias=bias)
        self.fc2 = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

        # Use fast GELU approximation
        self.act = FusedGELU(approximate="tanh")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        In real fused kernel:
          All operations happen in single CUDA kernel
          Intermediate results stay in SRAM
          Minimal HBM transfers

        Args:
            x: Input

        Returns:
            Output
        """
        residual = x

        # In fused kernel: all in one pass
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = x + residual

        return x


def benchmark_fused_kernels():
    """Benchmark fused vs unfused operations"""
    print("="*80)
    print("FUSED KERNELS BENCHMARK")
    print("="*80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Config
    batch_size = 32
    seq_len = 512
    d_model = 768

    # Create modules
    standard = StandardLayerNorm(d_model).to(device)
    fused = FusedLayerNormResidual(d_model).to(device)

    # Input
    x = torch.randn(batch_size, seq_len, d_model, device=device)
    residual = torch.randn(batch_size, seq_len, d_model, device=device)

    # Warmup
    with torch.no_grad():
        _ = standard(x, residual)
        _ = fused(x, residual)

    if device == 'cuda':
        torch.cuda.synchronize()

    # Benchmark
    num_iters = 100

    # Standard
    start = time.time()
    for _ in range(num_iters):
        with torch.no_grad():
            _ = standard(x, residual)

    if device == 'cuda':
        torch.cuda.synchronize()

    standard_time = (time.time() - start) / num_iters

    # Fused
    start = time.time()
    for _ in range(num_iters):
        with torch.no_grad():
            _ = fused(x, residual)

    if device == 'cuda':
        torch.cuda.synchronize()

    fused_time = (time.time() - start) / num_iters

    # Results
    print(f"\nConfiguration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Model dim: {d_model}")

    print(f"\n{'Method':<20} {'Time (ms)':<15} {'Speedup':<10}")
    print("-"*80)
    print(f"{'Standard':<20} {standard_time*1000:<15.3f} {'1.0x':<10}")
    print(f"{'Fused (PyTorch)':<20} {fused_time*1000:<15.3f} {standard_time/fused_time:<10.1f}x")

    print(f"\nNote: Real CUDA fused kernels (apex, TensorRT-LLM) are 3-5x faster!")
    print("This is just PyTorch approximation.")


if __name__ == "__main__":
    benchmark_fused_kernels()

    print("\n" + "="*80)
    print("REAL FUSED KERNELS")
    print("="*80)
    print("""
To use real fused kernels:

1. NVIDIA Apex:
   # Install
   git clone https://github.com/NVIDIA/apex
   cd apex
   pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

   # Use
   from apex.normalization import FusedLayerNorm
   norm = FusedLayerNorm(d_model)

2. TensorRT-LLM:
   # Provides fully fused transformer blocks
   # See TensorRT-LLM documentation

3. Custom CUDA:
   # Write custom fused kernels
   # Use torch.utils.cpp_extension
   # See examples in vLLM, Flash Attention repos

Benefits of real fused kernels:
  ✅ 3-5x faster than PyTorch
  ✅ Much less memory bandwidth
  ✅ Critical for production performance
    """)
```

*[Suite avec Speculative Decoding et Production System dans la partie 3...]*
