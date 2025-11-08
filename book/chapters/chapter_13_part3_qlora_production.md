# Chapitre 13 (Partie 3): QLoRA et Système de Production

## 5. QLoRA (Quantized LoRA)

```python
"""
QLoRA = Train LoRA adapters on quantized base model

Paper: "QLoRA: Efficient Finetuning of Quantized LLMs" (Dettmers et al., 2023)

Key innovation:
  • Load base model in 4-bit
  • Train LoRA adapters in FP16
  • Backprop through quantized weights
  • Memory: 10x reduction vs full fine-tuning

Requirements:
  • 4-bit base model (NF4 quantization)
  • LoRA adapters (trainable)
  • Double quantization (quantize scales too!)
  • Paged optimizers (for long sequences)

Results:
  • Llama 2 70B fine-tuning: 48GB GPU (was 480GB!)
  • Quality: Same as full fine-tuning
  • Speed: Slightly slower but acceptable
  • Enables fine-tuning on consumer GPUs

Example:
  • Fine-tune Llama 2 70B on single A100
  • Fine-tune Llama 2 13B on RTX 3090
  • Fine-tune Llama 2 7B on RTX 3060
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional


@dataclass
class QLoRAConfig:
    """QLoRA configuration"""
    # Quantization
    bits: int = 4
    quant_type: str = "nf4"  # nf4 or fp4
    double_quant: bool = True  # Double quantization

    # LoRA
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: list = None

    # Training
    learning_rate: float = 2e-4
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 2048


def demo_qlora_training():
    """Demo QLoRA training setup"""
    print("="*80)
    print("QLORA TRAINING")
    print("="*80)

    print("""
SETUP WITH TRANSFORMERS + PEFT + BITSANDBYTES:

# 1. Install dependencies
pip install transformers peft bitsandbytes accelerate

# 2. Load model in 4-bit
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

model_name = "meta-llama/Llama-2-7b-hf"

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # NF4 quantization
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,      # Double quantization
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

# 3. Prepare for training
model = prepare_model_for_kbit_training(model)

# 4. Add LoRA adapters
lora_config = LoraConfig(
    r=64,                          # LoRA rank
    lora_alpha=16,                 # LoRA alpha
    target_modules=[               # Target modules
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# Print trainable parameters
model.print_trainable_parameters()
# Output: trainable params: 41,943,040 || all params: 6,738,415,616 || trainable%: 0.62

# 5. Train
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

# Load dataset
dataset = load_dataset("timdettmers/openassistant-guanaco")

training_args = TrainingArguments(
    output_dir="./qlora-llama2-7b",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    num_train_epochs=3,
    save_strategy="epoch",
    fp16=True,                     # Mixed precision
    optim="paged_adamw_8bit",     # Paged optimizer
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
)

# Train!
trainer.train()

# 6. Save adapter
model.save_pretrained("./qlora-adapter")

# 7. Load and use
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

model = PeftModel.from_pretrained(base_model, "./qlora-adapter")

# Generate
inputs = tokenizer("Hello", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0]))


MEMORY COMPARISON (Llama 2 7B):

Method                      GPU Memory    Trainable Params
────────────────────────────────────────────────────────────
Full Fine-tuning (FP16)     28GB          7B (100%)
LoRA (FP16)                 16GB          40M (0.6%)
QLoRA (4-bit + LoRA)        6GB           40M (0.6%)

Method                      GPU Memory    Trainable Params
────────────────────────────────────────────────────────────
Full Fine-tuning (FP16)     280GB         70B (100%)
LoRA (FP16)                 160GB         400M (0.6%)
QLoRA (4-bit + LoRA)        48GB          400M (0.6%)

Impact:
  • 70B model trainable on single A100!
  • 7B model trainable on consumer GPU (RTX 3090)
  • Same quality as full fine-tuning
  • 4-5x slower training (acceptable)


TIPS:
  ✅ Use nf4 quantization (better than fp4)
  ✅ Enable double quantization
  ✅ Use paged_adamw_8bit optimizer
  ✅ Start with r=64, lora_alpha=16
  ✅ Target all attention layers
  ✅ Use gradient checkpointing for longer sequences
    """)


if __name__ == "__main__":
    demo_qlora_training()
```

## 6. Production Quantization System

```python
"""
AUTOMATIC QUANTIZATION SYSTEM

Features:
  • Auto-detect optimal quantization method
  • Benchmark quality vs speed
  • Deploy quantized models
  • Monitor degradation
"""

import torch
from typing import Dict, List, Optional
from dataclasses import dataclass
import time
import json
from pathlib import Path


@dataclass
class QuantizationBenchmark:
    """Benchmark results for quantization"""
    method: str
    bits: int
    model_size_gb: float
    latency_ms: float
    throughput_tokens_per_sec: float
    memory_gb: float
    perplexity: float
    quality_score: float  # 0-1 (vs FP16 baseline)


class AutoQuantizer:
    """
    Automatic quantization system

    Analyzes model and selects best quantization method

    Example:
        >>> quantizer = AutoQuantizer(model_name="llama-2-7b")
        >>> results = quantizer.benchmark_all()
        >>> best = quantizer.recommend(target="throughput")
    """

    def __init__(
        self,
        model_name: str,
        calibration_samples: int = 128
    ):
        """
        Args:
            model_name: Model name or path
            calibration_samples: Number of calibration samples
        """
        self.model_name = model_name
        self.calibration_samples = calibration_samples

        self.benchmarks: List[QuantizationBenchmark] = []

    def quantize_gptq(self, bits: int = 4) -> QuantizationBenchmark:
        """Quantize with GPTQ and benchmark"""
        print(f"Quantizing with GPTQ ({bits}-bit)...")

        # Simulate quantization (in practice, use AutoGPTQ)
        # This is just a demo structure

        return QuantizationBenchmark(
            method="GPTQ",
            bits=bits,
            model_size_gb=3.5 if bits == 4 else 7.0,
            latency_ms=45.0 if bits == 4 else 60.0,
            throughput_tokens_per_sec=120 if bits == 4 else 90,
            memory_gb=6.0 if bits == 4 else 10.0,
            perplexity=5.15 if bits == 4 else 5.05,
            quality_score=0.99 if bits == 4 else 0.995
        )

    def quantize_awq(self, bits: int = 4) -> QuantizationBenchmark:
        """Quantize with AWQ and benchmark"""
        print(f"Quantizing with AWQ ({bits}-bit)...")

        return QuantizationBenchmark(
            method="AWQ",
            bits=bits,
            model_size_gb=3.5 if bits == 4 else 7.0,
            latency_ms=40.0 if bits == 4 else 55.0,
            throughput_tokens_per_sec=140 if bits == 4 else 100,
            memory_gb=5.5 if bits == 4 else 9.5,
            perplexity=5.10 if bits == 4 else 5.02,
            quality_score=0.995 if bits == 4 else 0.998
        )

    def quantize_gguf(self, quant_type: str = "Q4_K_M") -> QuantizationBenchmark:
        """Quantize to GGUF format and benchmark"""
        print(f"Converting to GGUF ({quant_type})...")

        bits_map = {
            "Q4_0": 4.0,
            "Q4_K_M": 4.5,
            "Q5_K_M": 5.5,
            "Q8_0": 8.0,
        }

        bits = bits_map.get(quant_type, 4.0)

        return QuantizationBenchmark(
            method="GGUF",
            bits=int(bits),
            model_size_gb=4.0,
            latency_ms=80.0,  # CPU inference
            throughput_tokens_per_sec=35,
            memory_gb=8.0,
            perplexity=5.20,
            quality_score=0.98
        )

    def benchmark_all(self) -> List[QuantizationBenchmark]:
        """
        Benchmark all quantization methods

        Returns:
            List of benchmark results
        """
        print("="*80)
        print(f"BENCHMARKING QUANTIZATION METHODS: {self.model_name}")
        print("="*80)

        # Benchmark different methods
        methods = [
            ("gptq_4bit", lambda: self.quantize_gptq(bits=4)),
            ("gptq_8bit", lambda: self.quantize_gptq(bits=8)),
            ("awq_4bit", lambda: self.quantize_awq(bits=4)),
            ("gguf_q4_k_m", lambda: self.quantize_gguf("Q4_K_M")),
            ("gguf_q5_k_m", lambda: self.quantize_gguf("Q5_K_M")),
        ]

        self.benchmarks = []

        for name, method in methods:
            try:
                result = method()
                self.benchmarks.append(result)
                print(f"  ✅ {name}: {result.quality_score:.1%} quality, "
                      f"{result.throughput_tokens_per_sec:.0f} tok/s")
            except Exception as e:
                print(f"  ❌ {name}: Failed - {e}")

        return self.benchmarks

    def recommend(
        self,
        target: str = "balanced",
        min_quality: float = 0.95
    ) -> QuantizationBenchmark:
        """
        Recommend best quantization method

        Args:
            target: Optimization target ("quality", "throughput", "latency", "balanced")
            min_quality: Minimum acceptable quality score

        Returns:
            Recommended quantization config
        """
        if not self.benchmarks:
            raise ValueError("Run benchmark_all() first")

        # Filter by minimum quality
        candidates = [b for b in self.benchmarks if b.quality_score >= min_quality]

        if not candidates:
            raise ValueError(f"No method meets quality threshold {min_quality}")

        # Select based on target
        if target == "quality":
            best = max(candidates, key=lambda b: b.quality_score)
        elif target == "throughput":
            best = max(candidates, key=lambda b: b.throughput_tokens_per_sec)
        elif target == "latency":
            best = min(candidates, key=lambda b: b.latency_ms)
        elif target == "memory":
            best = min(candidates, key=lambda b: b.memory_gb)
        else:  # balanced
            # Composite score: quality × throughput / latency
            best = max(candidates, key=lambda b:
                      b.quality_score * b.throughput_tokens_per_sec / b.latency_ms)

        return best

    def print_report(self):
        """Print comprehensive benchmark report"""
        if not self.benchmarks:
            print("No benchmarks available. Run benchmark_all() first.")
            return

        print("\n" + "="*100)
        print("QUANTIZATION BENCHMARK REPORT")
        print("="*100)

        print(f"\n{'Method':<15} {'Bits':<6} {'Size':<8} {'Latency':<10} {'Throughput':<12} "
              f"{'Memory':<10} {'Quality':<10}")
        print("-"*100)

        for b in sorted(self.benchmarks, key=lambda x: -x.quality_score):
            print(f"{b.method:<15} {b.bits:<6} {b.model_size_gb:<8.1f}GB "
                  f"{b.latency_ms:<10.1f}ms {b.throughput_tokens_per_sec:<12.0f}tok/s "
                  f"{b.memory_gb:<10.1f}GB {b.quality_score:<10.1%}")

        # Recommendations
        print("\n" + "="*100)
        print("RECOMMENDATIONS")
        print("="*100)

        targets = ["quality", "throughput", "latency", "balanced"]

        for target in targets:
            try:
                best = self.recommend(target=target, min_quality=0.95)
                print(f"\n{target.capitalize():12}: {best.method} {best.bits}-bit")
                print(f"  Quality: {best.quality_score:.1%}")
                print(f"  Throughput: {best.throughput_tokens_per_sec:.0f} tokens/sec")
                print(f"  Latency: {best.latency_ms:.1f}ms")
                print(f"  Memory: {best.memory_gb:.1f}GB")
            except ValueError as e:
                print(f"\n{target.capitalize():12}: {e}")


class ProductionQuantizationPipeline:
    """
    Complete production quantization pipeline

    Example:
        >>> pipeline = ProductionQuantizationPipeline(
        ...     model_name="llama-2-7b",
        ...     output_dir="./quantized"
        ... )
        >>> pipeline.run(method="awq", bits=4)
    """

    def __init__(
        self,
        model_name: str,
        output_dir: str = "./quantized_models"
    ):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def run(
        self,
        method: str = "awq",
        bits: int = 4,
        validate: bool = True
    ):
        """
        Run quantization pipeline

        Args:
            method: Quantization method ("gptq", "awq", "gguf")
            bits: Number of bits
            validate: Run validation after quantization
        """
        print("="*80)
        print(f"QUANTIZATION PIPELINE: {self.model_name}")
        print("="*80)

        # Step 1: Quantize
        print(f"\n1. Quantizing with {method.upper()} ({bits}-bit)...")
        quantized_path = self.output_dir / f"{self.model_name}-{method}-{bits}bit"

        # Simulate quantization
        # In practice, call actual quantization libraries
        quantized_path.mkdir(exist_ok=True)

        print(f"   ✅ Saved to: {quantized_path}")

        # Step 2: Validate
        if validate:
            print(f"\n2. Validating quantized model...")
            quality_score = self._validate_model(quantized_path)
            print(f"   Quality score: {quality_score:.1%}")

        # Step 3: Benchmark
        print(f"\n3. Benchmarking performance...")
        benchmark = self._benchmark_model(quantized_path)
        print(f"   Throughput: {benchmark['throughput']:.0f} tokens/sec")
        print(f"   Latency: {benchmark['latency']:.1f}ms")

        # Step 4: Save metadata
        print(f"\n4. Saving metadata...")
        metadata = {
            "model_name": self.model_name,
            "method": method,
            "bits": bits,
            "quality_score": quality_score if validate else None,
            "benchmark": benchmark,
            "quantized_path": str(quantized_path)
        }

        with open(self.output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"\n✅ Quantization complete!")
        print(f"   Model: {quantized_path}")
        print(f"   Metadata: {self.output_dir / 'metadata.json'}")

    def _validate_model(self, model_path: Path) -> float:
        """Validate quantized model quality"""
        # In practice, run perplexity evaluation
        # This is simulation
        return 0.995

    def _benchmark_model(self, model_path: Path) -> Dict:
        """Benchmark quantized model"""
        # In practice, run actual benchmarks
        return {
            "throughput": 140,
            "latency": 40.0,
            "memory_gb": 5.5
        }


# Demo
if __name__ == "__main__":
    print("="*80)
    print("AUTOMATIC QUANTIZATION SYSTEM")
    print("="*80)

    # Demo auto-quantizer
    quantizer = AutoQuantizer(model_name="llama-2-7b")
    results = quantizer.benchmark_all()
    quantizer.print_report()

    print("\n" + "="*80)
    print("PRODUCTION PIPELINE EXAMPLE")
    print("="*80)

    print("""
Complete quantization workflow:

# 1. Auto-select best method
from auto_quantizer import AutoQuantizer, ProductionQuantizationPipeline

quantizer = AutoQuantizer("meta-llama/Llama-2-7b-hf")
results = quantizer.benchmark_all()

# Get recommendation
best = quantizer.recommend(target="balanced", min_quality=0.95)
print(f"Recommended: {best.method} {best.bits}-bit")

# 2. Quantize with selected method
pipeline = ProductionQuantizationPipeline(
    model_name="meta-llama/Llama-2-7b-hf",
    output_dir="./quantized"
)

pipeline.run(
    method=best.method.lower(),
    bits=best.bits,
    validate=True
)

# 3. Deploy
# Deploy quantized model to production
# See Chapter 14-15 for deployment


DEPLOYMENT CHECKLIST:

✅ Quantization:
   • Select method (AWQ for best quality/speed)
   • Validate quality (> 95% retention)
   • Benchmark performance

✅ Testing:
   • Unit tests (output correctness)
   • Integration tests (end-to-end)
   • Load tests (throughput, latency)

✅ Monitoring:
   • Track quality metrics
   • Monitor latency/throughput
   • Alert on degradation

✅ Rollout:
   • Canary deployment (1% traffic)
   • Gradual rollout (10% → 50% → 100%)
   • Rollback plan ready
    """)

    print("\n✅ CHAPITRE 13 TERMINÉ!")
    print("="*80)
    print("""
Vous maîtrisez maintenant:

1. Quantization Theory
   ✅ Symmetric vs asymmetric
   ✅ Per-tensor vs per-channel
   ✅ INT8, INT4, FP8 formats
   ✅ Calibration techniques

2. GPTQ
   ✅ Optimal brain quantization
   ✅ Layer-by-layer quantization
   ✅ Group-wise quantization
   ✅ 99% quality retention

3. AWQ
   ✅ Activation-aware quantization
   ✅ Salient weight protection
   ✅ Better quality than GPTQ
   ✅ Faster inference

4. GGUF/llama.cpp
   ✅ CPU inference
   ✅ Cross-platform
   ✅ Multiple quant types
   ✅ Works on Mac/Windows/Linux

5. QLoRA
   ✅ 4-bit base + LoRA adapters
   ✅ 10x memory reduction
   ✅ Train 70B on single GPU
   ✅ Same quality as full FT

6. Production System
   ✅ Auto-quantization
   ✅ Benchmarking pipeline
   ✅ Quality validation
   ✅ Deployment strategies

Results achieved:
  • 4-10x smaller models
  • 2-6x faster inference
  • < 5% quality loss
  • Consumer hardware viable

Ready for production quantized models!

Next: Chapter 14 → Cloud Deployment (AWS, GCP, Azure, Kubernetes)
    """)
```
