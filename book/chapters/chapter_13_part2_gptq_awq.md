# Chapitre 13 (Partie 2): GPTQ, AWQ et Formats Optimisés

## 2. GPTQ (Post-Training Quantization)

```python
"""
GPTQ = Optimal Brain Quantization for GPT

Paper: "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers"
        (Frantar et al., 2023)

Key idea:
  • Quantize weights layer-by-layer
  • Minimize reconstruction error
  • Use Hessian matrix (second-order)
  • No training needed!

Algorithm:
  For each layer:
    1. Collect activations (calibration data)
    2. Compute Hessian matrix H = 2 * X^T * X
    3. Quantize weights to minimize ||W_quant * X - W_orig * X||^2
    4. Use dynamic programming for optimal quantization

Results:
  • Llama 2 70B: 4-bit with 99% accuracy retention
  • 4x smaller, 3-4x faster
  • Works with any model
  • One-time process (~1 hour for 70B)

Libraries:
  • AutoGPTQ: https://github.com/PanQiWei/AutoGPTQ
  • GPTQ-for-LLaMa
  • Integrated in transformers

Used by:
  • TheBloke models on HuggingFace
  • Production systems
  • Most popular quantization method
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple
import math


class GPTQ:
    """
    GPTQ quantization algorithm (simplified)

    Real GPTQ is more complex, uses Cholesky decomposition,
    but this shows the core concept.

    Example:
        >>> gptq = GPTQ(bits=4, group_size=128)
        >>> # Calibrate with sample data
        >>> for batch in calibration_data:
        ...     gptq.add_batch(batch)
        >>> # Quantize layer
        >>> weight_quant, scale, zeros = gptq.quantize(layer.weight)
    """

    def __init__(
        self,
        bits: int = 4,
        group_size: int = 128,
        sym: bool = False,
        perchannel: bool = True
    ):
        """
        Args:
            bits: Number of bits (usually 4)
            group_size: Group size for quantization
            sym: Symmetric quantization
            perchannel: Per-channel quantization
        """
        self.bits = bits
        self.group_size = group_size
        self.sym = sym
        self.perchannel = perchannel

        self.maxq = 2 ** bits - 1

        # Accumulated Hessian
        self.H = None
        self.nsamples = 0

    def add_batch(self, inp: torch.Tensor):
        """
        Add batch for Hessian computation

        Args:
            inp: Input activations (batch, seq_len, hidden_dim)
        """
        if len(inp.shape) == 3:
            inp = inp.reshape(-1, inp.shape[-1])

        # Update Hessian: H = 2 * X^T * X
        # (In practice, done incrementally)
        tmp = inp.shape[0]

        if self.H is None:
            self.H = 2 * inp.t().matmul(inp)
        else:
            self.H += 2 * inp.t().matmul(inp)

        self.nsamples += tmp

    def quantize_groupwise(
        self,
        weight: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize weight matrix with groupwise quantization

        Args:
            weight: Weight matrix (out_features, in_features)

        Returns:
            (quantized_weight, scales, zeros)
        """
        out_features, in_features = weight.shape

        # Number of groups
        num_groups = (in_features + self.group_size - 1) // self.group_size

        # Initialize outputs
        weight_quant = torch.zeros_like(weight, dtype=torch.int32)
        scales = torch.zeros(out_features, num_groups, device=weight.device)
        zeros = torch.zeros(out_features, num_groups, device=weight.device)

        # Quantize group by group
        for i in range(num_groups):
            start_idx = i * self.group_size
            end_idx = min((i + 1) * self.group_size, in_features)

            # Get group
            w_group = weight[:, start_idx:end_idx]

            # Compute scale and zero point
            if self.sym:
                # Symmetric
                wmax = w_group.abs().amax(dim=1, keepdim=True)
                scale = wmax / (self.maxq / 2)
                scale = torch.clamp(scale, min=1e-8)
                zero = torch.zeros_like(scale)
            else:
                # Asymmetric
                wmin = w_group.amin(dim=1, keepdim=True)
                wmax = w_group.amax(dim=1, keepdim=True)
                scale = (wmax - wmin) / self.maxq
                scale = torch.clamp(scale, min=1e-8)
                zero = -wmin / scale

            # Quantize
            w_q = torch.round(w_group / scale + zero)
            w_q = torch.clamp(w_q, 0, self.maxq)

            # Store
            weight_quant[:, start_idx:end_idx] = w_q.int()
            scales[:, i] = scale.squeeze()
            zeros[:, i] = zero.squeeze()

        return weight_quant, scales, zeros

    def quantize(
        self,
        weight: torch.Tensor,
        use_hessian: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize weight matrix

        Args:
            weight: Weight to quantize
            use_hessian: Use Hessian for optimal quantization (slow)

        Returns:
            (quantized_weight, scales, zeros)
        """
        if use_hessian and self.H is not None:
            # Optimal quantization with Hessian
            # (Complex algorithm, see original paper)
            # For demo: use groupwise
            return self.quantize_groupwise(weight)
        else:
            # Simple groupwise quantization
            return self.quantize_groupwise(weight)


def demo_gptq():
    """Demo GPTQ usage with AutoGPTQ"""
    print("="*80)
    print("GPTQ QUANTIZATION")
    print("="*80)

    print("""
USING AutoGPTQ:

# 1. Install
pip install auto-gptq

# 2. Quantize model
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import AutoTokenizer

# Load model
model_name = "meta-llama/Llama-2-7b-hf"

# Quantization config
quantize_config = BaseQuantizeConfig(
    bits=4,                    # 4-bit quantization
    group_size=128,            # Group size
    desc_act=False,            # Activation ordering
    sym=True,                  # Symmetric quantization
    damp_percent=0.01,         # Damping
)

# Load calibration data
from datasets import load_dataset
data = load_dataset("c4", data_files="en/c4-train.00000-of-01024.json.gz", split="train")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize
examples = [
    tokenizer(data[i]["text"]) for i in range(128)
]

# Load and quantize
model = AutoGPTQForCausalLM.from_pretrained(
    model_name,
    quantize_config
)

model.quantize(examples)

# Save quantized model
model.save_quantized("./llama-2-7b-gptq-4bit")

# 3. Load and use quantized model
model = AutoGPTQForCausalLM.from_quantized(
    "./llama-2-7b-gptq-4bit",
    device="cuda:0"
)

# Generate
output = model.generate(**tokenizer("Hello", return_tensors="pt").to("cuda:0"))
print(tokenizer.decode(output[0]))


PERFORMANCE:

Model: Llama 2 7B
  • Original (FP16): 14GB, 50 tokens/sec
  • GPTQ 4-bit: 3.5GB, 120 tokens/sec
  • Speedup: 2.4x
  • Quality: 99% of original

Model: Llama 2 70B
  • Original (FP16): 140GB (doesn't fit on single GPU)
  • GPTQ 4-bit: 35GB (fits on A100!)
  • Perplexity: 5.10 → 5.15 (minimal degradation)


TIPS:
  ✅ Use group_size=128 (good tradeoff)
  ✅ More calibration data = better quality (but slower)
  ✅ sym=True usually works well
  ✅ Download pre-quantized models from TheBloke
    """)


if __name__ == "__main__":
    demo_gptq()
```

## 3. AWQ (Activation-aware Weight Quantization)

```python
"""
AWQ = Activation-aware Weight Quantization

Paper: "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration"
        (Lin et al., 2023)

Key insight:
  • Not all weights equally important!
  • 1% of weights handle 99% of activations
  • Protect important weights, quantize others

Algorithm:
  1. Analyze activation magnitudes
  2. Find "salient" weight channels
  3. Scale salient weights (keep precision)
  4. Quantize all weights
  5. Adjust scales back

Benefits vs GPTQ:
  • Faster quantization (no Hessian)
  • Better quality (activation-aware)
  • Hardware-friendly (no zero points)
  • 2x faster inference

Results:
  • Llama 2 70B: 4-bit with 99.5% accuracy
  • Better than GPTQ on most benchmarks
  • Faster to quantize (minutes vs hours)

Library:
  • AutoAWQ: https://github.com/casper-hansen/AutoAWQ
  • Integrated in vLLM, TensorRT-LLM
"""

import torch
import torch.nn as nn


class AWQ:
    """
    AWQ quantization (simplified)

    Example:
        >>> awq = AWQ(bits=4, group_size=128)
        >>> # Collect activation stats
        >>> for batch in calibration_data:
        ...     awq.add_batch(batch)
        >>> # Quantize
        >>> weight_quant, scale = awq.quantize(layer.weight)
    """

    def __init__(
        self,
        bits: int = 4,
        group_size: int = 128,
        alpha: float = 0.5
    ):
        """
        Args:
            bits: Quantization bits
            group_size: Group size
            alpha: Scaling factor for salient weights
        """
        self.bits = bits
        self.group_size = group_size
        self.alpha = alpha

        # Activation statistics
        self.activation_mag = None
        self.nsamples = 0

    def add_batch(self, inp: torch.Tensor):
        """
        Add batch for activation statistics

        Args:
            inp: Input activations
        """
        if len(inp.shape) == 3:
            inp = inp.reshape(-1, inp.shape[-1])

        # Track max activation magnitude per channel
        mag = inp.abs().amax(dim=0)

        if self.activation_mag is None:
            self.activation_mag = mag
        else:
            self.activation_mag = torch.max(self.activation_mag, mag)

        self.nsamples += inp.shape[0]

    def find_salient_channels(
        self,
        weight: torch.Tensor,
        percentile: float = 99
    ) -> torch.Tensor:
        """
        Find salient (important) channels based on activations

        Args:
            weight: Weight matrix
            percentile: Percentile threshold

        Returns:
            Scaling factors per channel
        """
        if self.activation_mag is None:
            # No calibration data, use uniform scaling
            return torch.ones(weight.shape[1], device=weight.device)

        # Compute importance: activation magnitude × weight magnitude
        weight_mag = weight.abs().mean(dim=0)
        importance = self.activation_mag * weight_mag

        # Find threshold
        threshold = torch.quantile(importance, percentile / 100)

        # Scale factor: s^alpha for salient, 1 for others
        scale = torch.ones_like(importance)
        salient = importance > threshold

        # Compute optimal scale for salient channels
        scale[salient] = (importance[salient] / importance.mean()).pow(self.alpha)

        return scale

    def quantize(
        self,
        weight: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize weight with AWQ

        Args:
            weight: Weight to quantize (out_features, in_features)

        Returns:
            (quantized_weight, scales, channel_scales)
        """
        # Find salient channels
        channel_scales = self.find_salient_channels(weight)

        # Apply scaling (protect salient weights)
        weight_scaled = weight * channel_scales.unsqueeze(0)

        # Group-wise quantization (similar to GPTQ)
        out_features, in_features = weight_scaled.shape
        num_groups = (in_features + self.group_size - 1) // self.group_size

        weight_quant = torch.zeros_like(weight_scaled, dtype=torch.int8)
        scales = torch.zeros(out_features, num_groups, device=weight.device)

        maxq = 2 ** self.bits - 1

        for i in range(num_groups):
            start_idx = i * self.group_size
            end_idx = min((i + 1) * self.group_size, in_features)

            w_group = weight_scaled[:, start_idx:end_idx]

            # Symmetric quantization
            wmax = w_group.abs().amax(dim=1, keepdim=True)
            scale = wmax / (maxq / 2)
            scale = torch.clamp(scale, min=1e-8)

            # Quantize
            w_q = torch.round(w_group / scale)
            w_q = torch.clamp(w_q, -(maxq // 2), maxq // 2)

            weight_quant[:, start_idx:end_idx] = w_q.char()
            scales[:, i] = scale.squeeze()

        return weight_quant, scales, channel_scales


def demo_awq():
    """Demo AWQ usage"""
    print("="*80)
    print("AWQ QUANTIZATION")
    print("="*80)

    print("""
USING AutoAWQ:

# 1. Install
pip install autoawq

# 2. Quantize model
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_name = "meta-llama/Llama-2-7b-hf"

# Load model
model = AutoAWQForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Quantization config
quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM"
}

# Quantize (automatic calibration)
model.quantize(
    tokenizer,
    quant_config=quant_config
)

# Save
model.save_quantized("./llama-2-7b-awq-4bit")

# 3. Load and use
model = AutoAWQForCausalLM.from_quantized(
    "./llama-2-7b-awq-4bit",
    fuse_layers=True  # Fuse layers for speed
)

# Generate
tokens = tokenizer("Hello", return_tensors="pt").to("cuda:0")
output = model.generate(**tokens, max_new_tokens=256)
print(tokenizer.decode(output[0]))


INTEGRATION WITH vLLM:

from vllm import LLM, SamplingParams

# Load AWQ model with vLLM
llm = LLM(
    model="./llama-2-7b-awq-4bit",
    quantization="awq",  # Specify AWQ
    dtype="float16"
)

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
outputs = llm.generate(["Write a poem"], sampling_params)


PERFORMANCE COMPARISON:

Llama 2 7B:
  Method          Size    Speed (tok/s)   Quality
  ────────────────────────────────────────────────
  FP16            14GB    50             100%
  GPTQ 4-bit      3.5GB   120            99.0%
  AWQ 4-bit       3.5GB   140            99.5%

Llama 2 70B:
  Method          Size    Speed (tok/s)   Quality
  ────────────────────────────────────────────────
  FP16            140GB   N/A            100%
  GPTQ 4-bit      35GB    25             98.8%
  AWQ 4-bit       35GB    30             99.2%


AWQ vs GPTQ:
  ✅ AWQ: Faster quantization (10min vs 1hr)
  ✅ AWQ: Better quality (activation-aware)
  ✅ AWQ: Faster inference (2x vs 1.5x)
  ✅ GPTQ: More mature, more models available
  ✅ Both: Production-ready
    """)


if __name__ == "__main__":
    demo_awq()
```

## 4. GGUF/GGML (llama.cpp)

```python
"""
GGUF/GGML = Georgi Gerganov's format for llama.cpp

llama.cpp:
  • C++ inference engine
  • CPU-optimized (no GPU needed!)
  • Extremely efficient
  • Cross-platform (Mac, Windows, Linux)

GGML (legacy) → GGUF (current):
  • Binary format for quantized models
  • Multiple quantization types
  • Optimized for CPU inference

Quantization types:
  • Q4_0: 4-bit, fastest
  • Q4_1: 4-bit, better quality
  • Q5_0, Q5_1: 5-bit
  • Q8_0: 8-bit
  • Q4_K_M, Q5_K_M: Mixed (K-quants, best quality)

Use cases:
  • Running LLMs on CPU
  • MacBooks (M1/M2/M3)
  • Edge devices
  • No GPU required!

Performance:
  • M2 Max: Llama 2 7B at 30-50 tokens/sec
  • M3 Max: Llama 2 13B at 25-40 tokens/sec
  • High-end CPU: Llama 2 7B at 15-30 tokens/sec

Tools:
  • llama.cpp: C++ engine
  • llama-cpp-python: Python bindings
  • LM Studio: GUI for GGUF models
  • Ollama: Easy CLI for running models
"""

def demo_gguf():
    """Demo GGUF/llama.cpp usage"""
    print("="*80)
    print("GGUF/GGML (llama.cpp)")
    print("="*80)

    print("""
QUANTIZATION TYPES:

Format      Bits    Size (7B)   Quality   Speed
────────────────────────────────────────────────
Q2_K        2.5     2.3GB       Poor      Fastest
Q3_K_M      3.5     3.0GB       Fair      Very Fast
Q4_0        4.0     3.5GB       Good      Fast
Q4_K_M      4.5     4.0GB       Very Good Fast
Q5_K_M      5.5     4.7GB       Excellent Medium
Q6_K        6.0     5.5GB       Near FP16 Slower
Q8_0        8.0     7.0GB       ~= FP16   Slow

Recommendation:
  ✅ Best quality/speed: Q4_K_M or Q5_K_M
  ✅ Best speed: Q4_0
  ✅ Best quality: Q6_K or Q8_0


USAGE WITH llama.cpp:

# 1. Install
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make

# 2. Download GGUF model (from HuggingFace)
# Search for "GGUF" on HuggingFace, e.g., TheBloke's models
# Example: https://huggingface.co/TheBloke/Llama-2-7B-GGUF

# 3. Run inference
./main -m llama-2-7b-q4_k_m.gguf \\
       -n 256 \\
       -p "Write a poem about AI"

# 4. Interactive mode
./main -m llama-2-7b-q4_k_m.gguf \\
       -n 256 \\
       --interactive


PYTHON BINDINGS (llama-cpp-python):

# Install
pip install llama-cpp-python

# Use
from llama_cpp import Llama

# Load model
llm = Llama(
    model_path="./llama-2-7b-q4_k_m.gguf",
    n_ctx=2048,          # Context window
    n_threads=8,         # CPU threads
    n_gpu_layers=0       # CPU only (set > 0 for GPU offloading)
)

# Generate
output = llm(
    "Write a poem about AI",
    max_tokens=256,
    temperature=0.8,
    top_p=0.95,
    echo=False
)

print(output['choices'][0]['text'])


USING OLLAMA (easiest):

# Install
curl https://ollama.ai/install.sh | sh

# Run model (automatically downloads GGUF)
ollama run llama2

# Or specific size
ollama run llama2:7b-q4_K_M
ollama run llama2:13b-q4_K_M
ollama run llama2:70b-q4_K_M

# Python API
import ollama

response = ollama.chat(model='llama2', messages=[
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
])
print(response['message']['content'])


PERFORMANCE (MacBook M2 Max, 32GB):

Model             Quant      Size    Speed (tok/s)   RAM
──────────────────────────────────────────────────────────
Llama 2 7B        Q4_K_M     4GB     45              12GB
Llama 2 13B       Q4_K_M     7GB     28              18GB
Llama 2 70B       Q4_K_M     35GB    N/A (too big)   N/A
CodeLlama 7B      Q4_K_M     4GB     42              12GB
Mistral 7B        Q4_K_M     4GB     50              12GB


TIPS:
  ✅ Use Q4_K_M or Q5_K_M for best balance
  ✅ Ollama is easiest for beginners
  ✅ llama.cpp for production/control
  ✅ Works great on Apple Silicon
  ✅ Can offload layers to GPU with n_gpu_layers
  ✅ Download from TheBloke on HuggingFace
    """)


if __name__ == "__main__":
    demo_gguf()

    print("\n" + "="*80)
    print("QUANTIZATION METHOD COMPARISON")
    print("="*80)

    comparison = """
Method      Format    Hardware    Quality    Speed      Ease
──────────────────────────────────────────────────────────────
GPTQ        GPU       CUDA        ★★★★☆     ★★★★☆     ★★★☆☆
AWQ         GPU       CUDA        ★★★★★     ★★★★★     ★★★☆☆
GGUF        CPU/GPU   Any         ★★★★☆     ★★★☆☆     ★★★★★

Use cases:
  • GPTQ: Production GPU inference (mature, stable)
  • AWQ: Production GPU inference (best quality/speed)
  • GGUF: CPU inference, edge devices, development

Recommendation:
  1. Production (GPU): Use AWQ with vLLM
  2. Development (local): Use GGUF with Ollama
  3. Edge/CPU: Use GGUF with llama.cpp
    """

    print(comparison)
```

*[Suite avec QLoRA et Production System dans la partie 3...]*
