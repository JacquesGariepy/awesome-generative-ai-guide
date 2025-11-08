# Chapitre 12: Inférence Optimisée pour LLMs

## Introduction

L'**inférence** est souvent le vrai bottleneck en production. Pendant que l'entraînement se fait une fois, l'inférence doit gérer des millions de requêtes par jour.

### Le Challenge de l'Inférence

```python
"""
Inférence LLM = Expensive!

Coûts comparés:
  Training GPT-3: $4.6M (one-time)
  Inference GPT-3: $100k-$1M+ par mois (ongoing)
  → Inference est souvent plus cher sur le long terme!

Challenges spécifiques:
  1. Memory-bound (pas compute-bound!)
     • Chaque token = load tous les poids
     • GPT-3 175B = 350GB de poids (FP16)
     • Bandwidth limité: 1-2TB/s

  2. Sequential generation
     • Ne peut pas paralléliser la génération
     • Chaque token dépend du précédent
     • Latency = critical

  3. Variable batch sizes
     • Différentes longueurs de séquences
     • Padding = waste
     • Need dynamic batching

  4. KV cache explosion
     • Llama 2 70B: 1GB KV cache per sequence (2048 tokens)
     • 100 concurrent users = 100GB just for cache!

Solutions:
  ✅ vLLM: PagedAttention + continuous batching
  ✅ TensorRT-LLM: Optimized CUDA kernels
  ✅ Flash Attention: Memory-efficient attention
  ✅ Speculative decoding: 2-3x speedup
  ✅ Quantization: 4-bit inference (Ch. 13)

Résultats réels:
  • vLLM vs HuggingFace: 24x throughput improvement
  • TensorRT-LLM: 8x faster than PyTorch
  • Speculative decoding: 2-3x speedup
  • Combined: 50-100x improvement possible!
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import torch
import torch.nn as nn
import time
import numpy as np


@dataclass
class InferenceStats:
    """Statistics for inference optimization"""
    framework: str
    model_size: str
    batch_size: int
    sequence_length: int
    throughput_tokens_per_sec: float
    latency_ms: float
    memory_gb: float
    cost_per_1m_tokens: float


# Real-world benchmarks
INFERENCE_BENCHMARKS = [
    InferenceStats(
        framework="HuggingFace (baseline)",
        model_size="Llama 2 7B",
        batch_size=1,
        sequence_length=512,
        throughput_tokens_per_sec=50,
        latency_ms=200,
        memory_gb=28,
        cost_per_1m_tokens=10.0
    ),
    InferenceStats(
        framework="vLLM",
        model_size="Llama 2 7B",
        batch_size=32,
        sequence_length=512,
        throughput_tokens_per_sec=1200,
        latency_ms=85,
        memory_gb=16,
        cost_per_1m_tokens=0.4
    ),
    InferenceStats(
        framework="TensorRT-LLM",
        model_size="Llama 2 7B",
        batch_size=32,
        sequence_length=512,
        throughput_tokens_per_sec=1500,
        latency_ms=70,
        memory_gb=14,
        cost_per_1m_tokens=0.3
    ),
    InferenceStats(
        framework="vLLM + Speculative",
        model_size="Llama 2 7B",
        batch_size=32,
        sequence_length=512,
        throughput_tokens_per_sec=2400,
        latency_ms=45,
        memory_gb=18,
        cost_per_1m_tokens=0.2
    ),
]


def print_inference_comparison():
    """Print comparison of inference frameworks"""
    print("="*100)
    print("INFERENCE OPTIMIZATION BENCHMARKS (Llama 2 7B)")
    print("="*100)

    print(f"\n{'Framework':<25} {'Throughput':<15} {'Latency':<12} {'Memory':<10} {'Cost/1M':<10}")
    print("-"*100)

    baseline = INFERENCE_BENCHMARKS[0]

    for stats in INFERENCE_BENCHMARKS:
        throughput_str = f"{stats.throughput_tokens_per_sec:.0f} tok/s"
        if stats != baseline:
            speedup = stats.throughput_tokens_per_sec / baseline.throughput_tokens_per_sec
            throughput_str += f" ({speedup:.1f}x)"

        cost_str = f"${stats.cost_per_1m_tokens:.2f}"
        if stats != baseline:
            savings = (1 - stats.cost_per_1m_tokens / baseline.cost_per_1m_tokens) * 100
            cost_str += f" (-{savings:.0f}%)"

        print(f"{stats.framework:<25} {throughput_str:<15} {stats.latency_ms:<12.0f}ms "
              f"{stats.memory_gb:<10.0f}GB {cost_str:<10}")

    print("\n" + "="*100)
    print("KEY INSIGHTS")
    print("="*100)
    print("""
1. vLLM = Biggest improvement
   • PagedAttention: Efficient KV cache management
   • Continuous batching: No padding waste
   • 24x throughput vs baseline
   • Industry standard (Anthropic, OpenAI use similar)

2. TensorRT-LLM = Fastest single-request
   • Optimized CUDA kernels
   • Fused operations
   • Best for low-latency scenarios

3. Speculative Decoding = Free speedup
   • Use small model to draft
   • Verify with large model
   • 2-3x faster, same quality
   • Works with any framework

4. Cost Impact
   • 50x cost reduction possible
   • $10 → $0.20 per 1M tokens
   • Makes production viable
    """)


if __name__ == "__main__":
    print_inference_comparison()
```

## 1. vLLM et PagedAttention

```python
"""
vLLM = State-of-the-art inference engine

Key innovation: PagedAttention
  • Inspired by virtual memory paging
  • KV cache stored in non-contiguous blocks
  • Near-zero waste
  • Dynamic allocation

Traditional attention:
  • Pre-allocate max sequence length
  • Llama 2 (4096 context): Allocate 4096 slots
  • Actual length 512: Waste 87% memory!

PagedAttention:
  • Allocate in blocks (e.g., 16 tokens)
  • 512 tokens: Only 32 blocks
  • 87% memory saved!

Installation:
  pip install vllm

Used by:
  • Anthropic (Claude)
  • Together AI
  • Anyscale
  • Many production systems
"""

import time
from typing import List, Optional
import torch


class PagedKVCache:
    """
    Simplified PagedAttention KV cache

    Real vLLM is much more sophisticated, but this shows the concept.

    Example:
        >>> cache = PagedKVCache(
        ...     num_blocks=100,
        ...     block_size=16,
        ...     num_heads=32,
        ...     head_dim=128
        ... )
        >>> # Allocate blocks for sequence
        >>> seq_blocks = cache.allocate_sequence(num_tokens=512)
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16
    ):
        """
        Args:
            num_blocks: Total number of blocks (pool)
            block_size: Tokens per block (usually 16)
            num_heads: Number of attention heads
            head_dim: Dimension per head
            dtype: Data type
        """
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_dim = head_dim

        # KV cache pool: [num_blocks, block_size, 2, num_heads, head_dim]
        # 2 = key and value
        self.cache = torch.zeros(
            num_blocks, block_size, 2, num_heads, head_dim,
            dtype=dtype,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

        # Free blocks
        self.free_blocks = set(range(num_blocks))

        # Sequence -> blocks mapping
        self.sequence_blocks: Dict[int, List[int]] = {}

    def allocate_sequence(self, seq_id: int, num_tokens: int) -> List[int]:
        """
        Allocate blocks for a sequence

        Args:
            seq_id: Sequence ID
            num_tokens: Number of tokens

        Returns:
            List of block IDs
        """
        # Calculate required blocks
        num_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size

        if len(self.free_blocks) < num_blocks_needed:
            raise RuntimeError(f"Out of memory: need {num_blocks_needed} blocks, "
                             f"only {len(self.free_blocks)} free")

        # Allocate blocks
        allocated = []
        for _ in range(num_blocks_needed):
            block_id = self.free_blocks.pop()
            allocated.append(block_id)

        self.sequence_blocks[seq_id] = allocated

        return allocated

    def free_sequence(self, seq_id: int):
        """
        Free blocks for a sequence

        Args:
            seq_id: Sequence ID
        """
        if seq_id not in self.sequence_blocks:
            return

        # Return blocks to free pool
        for block_id in self.sequence_blocks[seq_id]:
            self.free_blocks.add(block_id)

        del self.sequence_blocks[seq_id]

    def get_kv(self, seq_id: int, position: int) -> torch.Tensor:
        """
        Get KV for a position in sequence

        Args:
            seq_id: Sequence ID
            position: Token position

        Returns:
            KV tensor [2, num_heads, head_dim]
        """
        blocks = self.sequence_blocks[seq_id]

        # Find block and offset
        block_idx = position // self.block_size
        offset = position % self.block_size

        block_id = blocks[block_idx]

        return self.cache[block_id, offset]

    def set_kv(self, seq_id: int, position: int, kv: torch.Tensor):
        """
        Set KV for a position

        Args:
            seq_id: Sequence ID
            position: Token position
            kv: KV tensor [2, num_heads, head_dim]
        """
        blocks = self.sequence_blocks[seq_id]

        block_idx = position // self.block_size
        offset = position % self.block_size

        block_id = blocks[block_idx]

        self.cache[block_id, offset] = kv

    def get_utilization(self) -> float:
        """Get memory utilization"""
        used = self.num_blocks - len(self.free_blocks)
        return used / self.num_blocks

    def get_stats(self) -> Dict:
        """Get cache statistics"""
        return {
            'total_blocks': self.num_blocks,
            'free_blocks': len(self.free_blocks),
            'used_blocks': self.num_blocks - len(self.free_blocks),
            'utilization': self.get_utilization(),
            'active_sequences': len(self.sequence_blocks)
        }


class ContinuousBatchingEngine:
    """
    Continuous batching for dynamic request handling

    Traditional batching:
      • Wait for batch to fill
      • All sequences same length (padding)
      • Finish all sequences together

    Continuous batching:
      • Start processing immediately
      • Different lengths (no padding)
      • Sequences finish independently
      • New sequences join dynamically

    Example:
        >>> engine = ContinuousBatchingEngine(max_batch_size=32)
        >>> # Requests arrive continuously
        >>> engine.add_request("Generate a poem")
        >>> engine.add_request("Explain quantum physics")
        >>> # Process batch
        >>> results = engine.step()
    """

    def __init__(
        self,
        model,
        tokenizer,
        max_batch_size: int = 32,
        block_size: int = 16
    ):
        """
        Args:
            model: LLM model
            tokenizer: Tokenizer
            max_batch_size: Maximum batch size
            block_size: KV cache block size
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size

        # KV cache
        self.kv_cache = PagedKVCache(
            num_blocks=1000,  # Pool of 1000 blocks
            block_size=block_size,
            num_heads=model.config.num_attention_heads,
            head_dim=model.config.hidden_size // model.config.num_attention_heads
        )

        # Active requests
        self.active_requests: Dict[int, Dict] = {}
        self.next_request_id = 0

        # Waiting queue
        self.waiting_queue: List[Dict] = []

    def add_request(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 1.0
    ) -> int:
        """
        Add new request

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Request ID
        """
        request_id = self.next_request_id
        self.next_request_id += 1

        # Tokenize
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')[0]

        request = {
            'request_id': request_id,
            'prompt': prompt,
            'input_ids': input_ids,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'num_generated': 0,
            'finished': False,
            'output_ids': []
        }

        self.waiting_queue.append(request)

        return request_id

    def schedule_requests(self):
        """
        Schedule waiting requests to active batch

        This is where continuous batching magic happens!
        """
        # Add requests while we have space
        while (len(self.active_requests) < self.max_batch_size and
               len(self.waiting_queue) > 0):

            request = self.waiting_queue.pop(0)
            request_id = request['request_id']

            # Allocate KV cache
            num_tokens = len(request['input_ids']) + request['max_tokens']
            try:
                self.kv_cache.allocate_sequence(request_id, num_tokens)
                self.active_requests[request_id] = request
            except RuntimeError:
                # Out of memory, put back in queue
                self.waiting_queue.insert(0, request)
                break

    def step(self) -> Dict[int, str]:
        """
        Process one step of generation

        Returns:
            Dict of completed requests {request_id: generated_text}
        """
        # Schedule new requests
        self.schedule_requests()

        if not self.active_requests:
            return {}

        # Prepare batch
        # (In real vLLM, this is much more complex)
        batch_ids = []
        batch_requests = []

        for request_id, request in self.active_requests.items():
            if request['num_generated'] == 0:
                # First token: use full prompt
                batch_ids.append(request['input_ids'])
            else:
                # Subsequent tokens: use last generated
                batch_ids.append(torch.tensor([request['output_ids'][-1]]))

            batch_requests.append(request_id)

        # Pad batch (in real vLLM, no padding needed!)
        # This is simplified

        # Forward pass (simplified)
        # In real vLLM: Use PagedAttention kernel
        # For now: Skip actual forward pass

        # Simulate generation
        for request_id in batch_requests:
            request = self.active_requests[request_id]

            # Generate next token (random for demo)
            next_token = torch.randint(0, self.tokenizer.vocab_size, (1,)).item()

            request['output_ids'].append(next_token)
            request['num_generated'] += 1

            # Check if done
            if (request['num_generated'] >= request['max_tokens'] or
                next_token == self.tokenizer.eos_token_id):
                request['finished'] = True

        # Remove finished requests
        completed = {}

        for request_id in list(self.active_requests.keys()):
            request = self.active_requests[request_id]

            if request['finished']:
                # Decode output
                output_text = self.tokenizer.decode(request['output_ids'])
                completed[request_id] = output_text

                # Free KV cache
                self.kv_cache.free_sequence(request_id)

                # Remove from active
                del self.active_requests[request_id]

        return completed

    def get_stats(self) -> Dict:
        """Get engine statistics"""
        return {
            'active_requests': len(self.active_requests),
            'waiting_requests': len(self.waiting_queue),
            'kv_cache': self.kv_cache.get_stats()
        }


# Demo
if __name__ == "__main__":
    print("="*80)
    print("vLLM & PAGEDATTENTION")
    print("="*80)

    print("\n1. PAGED KV CACHE")
    print("-"*80)

    # Create cache
    cache = PagedKVCache(
        num_blocks=100,
        block_size=16,
        num_heads=32,
        head_dim=128
    )

    print(f"Cache config:")
    print(f"  Total blocks: {cache.num_blocks}")
    print(f"  Block size: {cache.block_size} tokens")
    print(f"  Total capacity: {cache.num_blocks * cache.block_size} tokens")

    # Allocate sequences
    seq1_blocks = cache.allocate_sequence(seq_id=1, num_tokens=512)
    print(f"\nAllocated seq 1 (512 tokens): {len(seq1_blocks)} blocks")

    seq2_blocks = cache.allocate_sequence(seq_id=2, num_tokens=128)
    print(f"Allocated seq 2 (128 tokens): {len(seq2_blocks)} blocks")

    stats = cache.get_stats()
    print(f"\nCache stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Compare to traditional
    traditional_memory = 2 * 4096 * 32 * 128 * 2  # max_len * num_seqs * num_heads * head_dim * 2 bytes (FP16)
    paged_memory = (len(seq1_blocks) + len(seq2_blocks)) * 16 * 32 * 128 * 2

    print(f"\nMemory comparison:")
    print(f"  Traditional (pre-allocated): {traditional_memory / 1e6:.1f} MB")
    print(f"  PagedAttention: {paged_memory / 1e6:.1f} MB")
    print(f"  Savings: {(1 - paged_memory/traditional_memory)*100:.1f}%")

    print("\n" + "="*80)
    print("READY TO USE vLLM")
    print("="*80)
    print("""
Real vLLM usage:

# Install
pip install vllm

# Basic usage
from vllm import LLM, SamplingParams

# Load model
llm = LLM(model="meta-llama/Llama-2-7b-hf")

# Generate
prompts = ["Write a poem", "Explain AI"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=256)

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(output.outputs[0].text)

# Advanced: OpenAI-compatible server
python -m vllm.entrypoints.openai.api_server \\
    --model meta-llama/Llama-2-7b-hf \\
    --tensor-parallel-size 2

# Then use with OpenAI API:
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

response = client.completions.create(
    model="meta-llama/Llama-2-7b-hf",
    prompt="Write a poem",
    max_tokens=256
)
    """)
```

*[Suite avec TensorRT-LLM et CUDA Optimizations dans la partie 2...]*
