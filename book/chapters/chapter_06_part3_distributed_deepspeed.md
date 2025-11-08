# Chapitre 6 (Partie 3): Distributed Training et DeepSpeed

## 4. Distributed Training

```python
"""
Distributed Training = Entraîner sur plusieurs GPUs

Pourquoi distribuer?
  • Single GPU: GPT-2 small (117M params) = ~400MB
  • Single GPU: GPT-3 (175B params) = ~700GB → NE RENTRE PAS!
  • Solution: Split across GPUs

Stratégies:

1. Data Parallel (DP)
   • Each GPU has FULL model copy
   • Split data across GPUs
   • Good for: Small models (< 1B params)
   • Memory: N × model_size

2. Distributed Data Parallel (DDP)
   • Like DP but more efficient
   • Standard PyTorch approach
   • Good for: Medium models (< 10B params)

3. Fully Sharded Data Parallel (FSDP)
   • Split model AND data across GPUs
   • Each GPU has fraction of model
   • Good for: Large models (10B-100B params)
   • Memory: model_size / N

4. Pipeline Parallel
   • Split model by layers
   • Layer 1-10 on GPU1, 11-20 on GPU2, etc.
   • Good for: Very deep models

5. Tensor Parallel
   • Split individual layers across GPUs
   • Used by Megatron-LM
   • Good for: Huge models (100B+ params)
"""

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
import os


class DistributedSetup:
    """
    Setup for distributed training
    """

    @staticmethod
    def setup_ddp(rank: int, world_size: int):
        """
        Setup DDP

        Args:
            rank: Process rank (0, 1, 2, ...)
            world_size: Total number of processes
        """
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'

        # Initialize process group
        dist.init_process_group(
            backend='nccl',  # Use 'nccl' for GPU, 'gloo' for CPU
            rank=rank,
            world_size=world_size
        )

        # Set device
        torch.cuda.set_device(rank)

    @staticmethod
    def cleanup_ddp():
        """Cleanup DDP"""
        dist.destroy_process_group()

    @staticmethod
    def is_main_process() -> bool:
        """Check if main process"""
        return dist.get_rank() == 0


class DDPTrainer:
    """
    Trainer with DDP support
    """

    def __init__(
        self,
        model: GPT,
        config: ModelConfig,
        train_dataset: TextDataset,
        val_dataset: TextDataset,
        rank: int,
        world_size: int
    ):
        self.rank = rank
        self.world_size = world_size
        self.config = config

        # Setup DDP
        DistributedSetup.setup_ddp(rank, world_size)

        # Move model to device
        device = torch.device(f'cuda:{rank}')
        model = model.to(device)

        # Wrap with DDP
        self.model = DDP(model, device_ids=[rank])

        # Datasets (each rank gets subset)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        # Optimizer
        self.optimizer = self._configure_optimizer()

        if self.is_main():
            print(f"Initialized DDP with {world_size} GPUs")

    def is_main(self) -> bool:
        """Check if main process"""
        return self.rank == 0

    def _configure_optimizer(self):
        """Configure optimizer"""
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
            weight_decay=self.config.weight_decay
        )

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """
        Single training step

        Args:
            x: Input tokens
            y: Target tokens

        Returns:
            Loss
        """
        self.optimizer.zero_grad()

        # Forward
        logits, loss = self.model(x, y)

        # Backward
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.grad_clip
        )

        # Optimizer step
        self.optimizer.step()

        return loss.item()

    def train(self):
        """Training loop"""
        if self.is_main():
            print("Starting DDP training...")

        for iter in range(self.config.max_iters):
            # Get batch (each rank gets different data)
            x, y = DataModule.get_batch(
                self.train_dataset,
                self.config.batch_size,
                f'cuda:{self.rank}'
            )

            # Train step
            loss = self.train_step(x, y)

            # Log (only main process)
            if self.is_main() and iter % 100 == 0:
                print(f"Iter {iter}: Loss {loss:.4f}")

        # Cleanup
        DistributedSetup.cleanup_ddp()


def launch_ddp_example():
    """
    Example of launching DDP training

    In practice, use torch.distributed.launch or torchrun:

    # Single node, 4 GPUs
    torchrun --nproc_per_node=4 train_ddp.py

    # Multi-node (2 nodes, 4 GPUs each = 8 total)
    # Node 0:
    torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
             --master_addr=192.168.1.1 --master_port=12355 train_ddp.py

    # Node 1:
    torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 \
             --master_addr=192.168.1.1 --master_port=12355 train_ddp.py
    """

    print("""
Example DDP launch:

# train_ddp.py
import torch.multiprocessing as mp

def main():
    world_size = torch.cuda.device_count()

    # Spawn processes
    mp.spawn(
        train_worker,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )

def train_worker(rank, world_size):
    # Setup
    setup_ddp(rank, world_size)

    # Create model, trainer, etc.
    # ...

    # Train
    trainer.train()

    # Cleanup
    cleanup_ddp()

if __name__ == "__main__":
    main()

# Or use torchrun (recommended):
# torchrun --nproc_per_node=4 train_ddp.py
    """)


## 5. DeepSpeed

"""
DeepSpeed = Microsoft's library for efficient large-scale training

Features:
  • ZeRO (Zero Redundancy Optimizer)
    - Stage 1: Shard optimizer states (4x memory reduction)
    - Stage 2: Shard gradients (8x reduction)
    - Stage 3: Shard parameters (linear scaling)
  • Activation checkpointing
  • Mixed precision (FP16)
  • Gradient accumulation
  • 3D parallelism (data + pipeline + tensor)

Used by:
  • GPT-3, GPT-NeoX, BLOOM, Llama
  • Megatron-DeepSpeed

Installation:
  pip install deepspeed
"""

class DeepSpeedTrainer:
    """
    Trainer with DeepSpeed

    Requires deepspeed config file
    """

    @staticmethod
    def create_deepspeed_config() -> dict:
        """
        Create DeepSpeed configuration

        Returns:
            Config dict
        """
        config = {
            "train_batch_size": 32,
            "train_micro_batch_size_per_gpu": 4,  # Gradient accumulation = 32/4 = 8 steps
            "gradient_accumulation_steps": 8,

            # Optimizer
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": 3e-4,
                    "betas": [0.9, 0.95],
                    "eps": 1e-8,
                    "weight_decay": 0.1
                }
            },

            # Learning rate schedule
            "scheduler": {
                "type": "WarmupDecayLR",
                "params": {
                    "warmup_min_lr": 0,
                    "warmup_max_lr": 3e-4,
                    "warmup_num_steps": 1000,
                    "total_num_steps": 10000
                }
            },

            # Mixed precision (FP16)
            "fp16": {
                "enabled": True,
                "loss_scale": 0,
                "loss_scale_window": 1000,
                "hysteresis": 2,
                "min_loss_scale": 1
            },

            # ZeRO optimization
            "zero_optimization": {
                "stage": 2,  # Stage 0, 1, 2, or 3
                "offload_optimizer": {
                    "device": "cpu",  # Offload to CPU RAM
                    "pin_memory": True
                },
                "allgather_partitions": True,
                "allgather_bucket_size": 5e8,
                "reduce_scatter": True,
                "reduce_bucket_size": 5e8,
                "overlap_comm": True,
                "contiguous_gradients": True
            },

            # Gradient clipping
            "gradient_clipping": 1.0,

            # Logging
            "steps_per_print": 100,
            "wall_clock_breakdown": False
        }

        return config

    @staticmethod
    def print_zero_stages():
        """Explain ZeRO stages"""
        print("="*80)
        print("DEEPSPEED ZERO OPTIMIZATION STAGES")
        print("="*80)

        stages = [
            {
                "stage": 0,
                "name": "Disabled",
                "what": "Standard DDP",
                "memory": "N × model_size",
                "speedup": "1x (baseline)"
            },
            {
                "stage": 1,
                "name": "Optimizer State Partitioning",
                "what": "Shard optimizer states across GPUs",
                "memory": "0.75 × baseline",
                "speedup": "1.0x"
            },
            {
                "stage": 2,
                "name": "+ Gradient Partitioning",
                "what": "Shard gradients + optimizer states",
                "memory": "0.5 × baseline",
                "speedup": "1.0x"
            },
            {
                "stage": 3,
                "name": "+ Parameter Partitioning",
                "what": "Shard everything (params, grads, optim)",
                "memory": "model_size / N",
                "speedup": "0.9x (slight overhead)"
            }
        ]

        for stage_info in stages:
            print(f"\nZeRO Stage {stage_info['stage']}: {stage_info['name']}")
            print(f"  What: {stage_info['what']}")
            print(f"  Memory: {stage_info['memory']}")
            print(f"  Speedup: {stage_info['speedup']}")

        print("\n" + "="*80)
        print("EXAMPLE: Training 7B Model")
        print("="*80)
        print("""
Model: 7B parameters = ~28GB (FP32) = ~14GB (FP16)

Without ZeRO (DDP):
  • 8x A100 (40GB each)
  • Each GPU: 14GB (model) + 14GB (gradients) + 28GB (optimizer) = 56GB
  • ❌ OOM! Need 56GB but only have 40GB

With ZeRO Stage 2:
  • Model: 14GB (replicated on each GPU)
  • Gradients: 14GB / 8 = 1.75GB per GPU
  • Optimizer: 28GB / 8 = 3.5GB per GPU
  • Total: 14 + 1.75 + 3.5 = 19.25GB per GPU
  • ✅ Fits in 40GB!

With ZeRO Stage 3:
  • Everything sharded: (14 + 14 + 28) / 8 = 7GB per GPU
  • ✅ Can even fit on 8GB GPUs!
  • Or train 50B model on 8x A100s
        """)

    @staticmethod
    def example_usage():
        """Example of using DeepSpeed"""
        print("\n" + "="*80)
        print("DEEPSPEED USAGE EXAMPLE")
        print("="*80)

        print("""
# train_deepspeed.py

import deepspeed
from deepspeed.ops.adam import FusedAdam

def main():
    # Create model
    model = GPT(config)

    # Create optimizer (optional, can use config)
    optimizer = FusedAdam(
        model.parameters(),
        lr=3e-4,
        betas=(0.9, 0.95)
    )

    # Initialize DeepSpeed
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config="ds_config.json"
    )

    # Training loop
    for iter in range(max_iters):
        # Get batch
        x, y = get_batch(...)

        # Forward
        logits, loss = model_engine(x, y)

        # Backward (DeepSpeed handles everything!)
        model_engine.backward(loss)

        # Step
        model_engine.step()

    # Save checkpoint
    model_engine.save_checkpoint("checkpoint")

# Launch with DeepSpeed
# deepspeed --num_gpus=8 train_deepspeed.py

# Or with hostfile (multi-node)
# deepspeed --hostfile=hostfile train_deepspeed.py
        """)


# Demo
if __name__ == "__main__":
    print("="*80)
    print("DISTRIBUTED TRAINING & DEEPSPEED")
    print("="*80)

    # DDP example
    launch_ddp_example()

    # DeepSpeed
    deepspeed_trainer = DeepSpeedTrainer()
    deepspeed_trainer.print_zero_stages()
    deepspeed_trainer.example_usage()

    print("\n" + "="*80)
    print("KEY TAKEAWAYS")
    print("="*80)
    print("""
1. Single GPU: < 1B parameters
   • Use standard training
   • GPT-2 small (117M) fits on 1x 16GB GPU

2. Multi-GPU (same node): 1B-10B parameters
   • Use DDP (PyTorch built-in)
   • 4x 24GB GPUs → GPT-2 XL (1.5B)

3. Multi-GPU (ZeRO): 10B-100B parameters
   • Use DeepSpeed ZeRO Stage 2/3
   • 8x 40GB A100s → Llama 2 7B comfortably

4. Multi-node: 100B+ parameters
   • Use DeepSpeed + pipeline parallel
   • 100s-1000s of GPUs
   • GPT-3 (175B), Llama 2 70B

Best practices:
  ✅ Start with single GPU (prototyping)
  ✅ Scale to DDP (multi-GPU, same node)
  ✅ Use DeepSpeed for large models (> 10B)
  ✅ Use ZeRO Stage 2 by default
  ✅ Use ZeRO Stage 3 for very large models
  ✅ Monitor GPU utilization (should be 90-100%)

Tools:
  • PyTorch DDP: Built-in, simple
  • DeepSpeed: Advanced, efficient
  • FSDP: PyTorch's ZeRO alternative
  • Megatron-LM: Nvidia's framework (tensor parallel)
  • Accelerate (HuggingFace): Unified API
    """)

    print("\n✅ CHAPITRE 6 TERMINÉ!")
    print("="*80)
    print("""
Vous maîtrisez maintenant:

1. Architecture GPT complète
   ✅ Causal attention
   ✅ Transformer blocks
   ✅ Generation

2. Data Pipeline
   ✅ Character/token-level datasets
   ✅ Batch generation
   ✅ Train/val splits

3. Training Loop
   ✅ AdamW optimizer
   ✅ Gradient clipping
   ✅ Checkpointing
   ✅ Evaluation

4. Distributed Training
   ✅ DDP (multi-GPU)
   ✅ DeepSpeed ZeRO
   ✅ Scaling strategies

Ready to train your own LLM!

Next: Chapter 7 → Fine-tuning (already done!)
    """)
```
