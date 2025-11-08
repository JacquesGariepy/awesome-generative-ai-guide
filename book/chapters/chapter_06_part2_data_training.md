# Chapitre 6 (Partie 2): Data Pipeline et Training Loop

## 2. Préparation des Données

```python
"""
Data Pipeline pour Pré-entraînement

Sources de données:
  • Common Crawl: Web data (billions of pages)
  • Books: Project Gutenberg, Books3
  • Wikipedia: Clean, factual
  • GitHub: Code (for code models)
  • Papers: ArXiv (scientific)

Notre exemple: Shakespeare (simple, educational)
  • ~1MB of text
  • ~300k characters
  • ~100k tokens
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List, Tuple
import os


class TextDataset(Dataset):
    """
    Dataset for character-level or token-level language modeling
    """

    def __init__(
        self,
        text: str,
        block_size: int,
        tokenizer=None,
        char_level: bool = True
    ):
        """
        Args:
            text: Raw text data
            block_size: Context window size
            tokenizer: Tokenizer (if token-level)
            char_level: Use character-level (True) or token-level (False)
        """
        self.block_size = block_size
        self.char_level = char_level

        if char_level:
            # Character-level
            chars = sorted(list(set(text)))
            self.vocab_size = len(chars)

            print(f"Data has {len(text)} characters, {self.vocab_size} unique")

            # Create mappings
            self.stoi = {ch: i for i, ch in enumerate(chars)}
            self.itos = {i: ch for i, ch in enumerate(chars)}

            # Encode entire text
            self.data = torch.tensor(
                [self.stoi[c] for c in text],
                dtype=torch.long
            )

        else:
            # Token-level (requires tokenizer)
            assert tokenizer is not None, "Tokenizer required for token-level"
            self.tokenizer = tokenizer
            self.vocab_size = tokenizer.vocab_size

            # Encode text
            tokens = tokenizer.encode(text)
            self.data = torch.tensor(tokens, dtype=torch.long)

            print(f"Data has {len(text)} characters, {len(tokens)} tokens")

    def __len__(self):
        """Number of samples"""
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get one training example

        Returns:
            x: Input tokens (block_size,)
            y: Target tokens (block_size,) - shifted by 1
        """
        # Get chunk
        chunk = self.data[idx:idx + self.block_size + 1]

        # Input and target
        x = chunk[:-1]
        y = chunk[1:]

        return x, y

    def decode(self, tokens: torch.Tensor) -> str:
        """
        Decode tokens to text

        Args:
            tokens: Token IDs

        Returns:
            Decoded text
        """
        if self.char_level:
            return ''.join([self.itos[int(i)] for i in tokens])
        else:
            return self.tokenizer.decode(tokens.tolist())


class DataModule:
    """
    Complete data module for pretraining
    """

    @staticmethod
    def load_shakespeare() -> str:
        """
        Load Shakespeare dataset

        Returns:
            Raw text
        """
        # In practice, download from:
        # https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

        # For demo, use placeholder
        text = """
First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?

All:
Resolved. resolved.

First Citizen:
First, you know Caius Marcius is chief enemy to the people.
""" * 100  # Repeat for more data

        return text

    @staticmethod
    def create_splits(
        text: str,
        train_frac: float = 0.9
    ) -> Tuple[str, str]:
        """
        Split text into train and validation

        Args:
            text: Full text
            train_frac: Fraction for training

        Returns:
            (train_text, val_text)
        """
        n = len(text)
        train_data = text[:int(n * train_frac)]
        val_data = text[int(n * train_frac):]

        print(f"Train: {len(train_data)} chars")
        print(f"Val: {len(val_data)} chars")

        return train_data, val_data

    @staticmethod
    def get_batch(
        dataset: TextDataset,
        batch_size: int,
        device: str = "cpu"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get random batch

        Args:
            dataset: Dataset
            batch_size: Batch size
            device: Device to put tensors on

        Returns:
            (x, y) batch
        """
        # Random indices
        ix = torch.randint(len(dataset), (batch_size,))

        # Get examples
        x_list = []
        y_list = []

        for i in ix:
            x, y = dataset[i]
            x_list.append(x)
            y_list.append(y)

        # Stack into batch
        x = torch.stack(x_list).to(device)
        y = torch.stack(y_list).to(device)

        return x, y


# Demo
if __name__ == "__main__":
    print("="*80)
    print("DATA PIPELINE")
    print("="*80)

    # Load data
    text = DataModule.load_shakespeare()
    print(f"\nLoaded text: {len(text)} characters")
    print(f"Sample:\n{text[:200]}...")

    # Split
    train_text, val_text = DataModule.create_splits(text)

    # Create dataset
    block_size = 64
    train_dataset = TextDataset(train_text, block_size, char_level=True)
    val_dataset = TextDataset(val_text, block_size, char_level=True)

    print(f"\nDataset info:")
    print(f"  Vocab size: {train_dataset.vocab_size}")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")

    # Get batch
    x, y = DataModule.get_batch(train_dataset, batch_size=4)

    print(f"\nBatch shapes:")
    print(f"  x: {x.shape}")
    print(f"  y: {y.shape}")

    print(f"\nFirst training example:")
    print(f"  Input:  {train_dataset.decode(x[0])}")
    print(f"  Target: {train_dataset.decode(y[0])}")
```

## 3. Training Loop Complet

```python
"""
Training Loop pour Pré-entraînement

Composants:
  • Forward pass
  • Loss computation
  • Backward pass
  • Optimizer step
  • Logging
  • Checkpointing
  • Evaluation
"""

import time
from contextlib import nullcontext
from pathlib import Path
import json


class Trainer:
    """
    Trainer for pretraining GPT
    """

    def __init__(
        self,
        model: GPT,
        config: ModelConfig,
        train_dataset: TextDataset,
        val_dataset: TextDataset
    ):
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        # Move model to device
        self.model.to(config.device)

        # Optimizer
        self.optimizer = self._configure_optimizer()

        # For logging
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []

    def _configure_optimizer(self) -> torch.optim.Optimizer:
        """
        Configure optimizer (AdamW with weight decay)
        """
        # Separate parameters into decay and no-decay groups
        decay = set()
        no_decay = set()

        whitelist_weight_modules = (nn.Linear,)
        blacklist_weight_modules = (nn.LayerNorm, nn.Embedding)

        for mn, m in self.model.named_modules():
            for pn, p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn

                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        # Validate
        param_dict = {pn: p for pn, p in self.model.named_parameters()}

        inter_params = decay & no_decay
        union_params = decay | no_decay

        assert len(inter_params) == 0, "Parameters in both decay and no_decay"
        assert len(param_dict.keys() - union_params) == 0, "Parameters not categorized"

        # Create optimizer groups
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": self.config.weight_decay
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0
            }
        ]

        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2)
        )

        return optimizer

    @torch.no_grad()
    def estimate_loss(self) -> dict:
        """
        Estimate loss on train and val sets

        Returns:
            Dict with 'train' and 'val' losses
        """
        self.model.eval()

        out = {}
        for split in ['train', 'val']:
            losses = torch.zeros(self.config.eval_iters)

            dataset = self.train_dataset if split == 'train' else self.val_dataset

            for k in range(self.config.eval_iters):
                x, y = DataModule.get_batch(
                    dataset,
                    self.config.batch_size,
                    self.config.device
                )

                with nullcontext():
                    logits, loss = self.model(x, y)

                losses[k] = loss.item()

            out[split] = losses.mean()

        self.model.train()
        return out

    def train(self):
        """
        Complete training loop
        """
        print("="*80)
        print("TRAINING")
        print("="*80)

        print(f"\nConfiguration:")
        print(f"  Device: {self.config.device}")
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  Learning rate: {self.config.learning_rate}")
        print(f"  Max iterations: {self.config.max_iters}")
        print(f"  Eval interval: {self.config.eval_interval}")

        # Compile model (PyTorch 2.0+)
        if self.config.compile:
            print("\nCompiling model...")
            self.model = torch.compile(self.model)

        # Training loop
        print("\n" + "="*80)
        print("Starting training...")
        print("="*80)

        start_time = time.time()

        for iter in range(self.config.max_iters):
            # Evaluate
            if iter % self.config.eval_interval == 0 or iter == self.config.max_iters - 1:
                losses = self.estimate_loss()
                train_loss = losses['train']
                val_loss = losses['val']

                print(f"\nIter {iter}/{self.config.max_iters}")
                print(f"  Train loss: {train_loss:.4f}")
                print(f"  Val loss: {val_loss:.4f}")

                self.train_losses.append(train_loss)
                self.val_losses.append(val_loss)

                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(f"best_model.pt")
                    print(f"  ✅ New best val loss: {val_loss:.4f}")

                # Sample generation
                if iter % (self.config.eval_interval * 5) == 0:
                    self.generate_sample()

            # Get batch
            x, y = DataModule.get_batch(
                self.train_dataset,
                self.config.batch_size,
                self.config.device
            )

            # Forward
            logits, loss = self.model(x, y)

            # Backward
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.grad_clip
            )

            # Optimizer step
            self.optimizer.step()

        # Training complete
        elapsed = time.time() - start_time

        print("\n" + "="*80)
        print("TRAINING COMPLETE")
        print("="*80)
        print(f"Time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
        print(f"Best val loss: {self.best_val_loss:.4f}")
        print(f"Final train loss: {self.train_losses[-1]:.4f}")

    def save_checkpoint(self, filename: str):
        """
        Save model checkpoint

        Args:
            filename: Checkpoint filename
        """
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.config,
            'best_val_loss': self.best_val_loss
        }

        torch.save(checkpoint, filename)

    def load_checkpoint(self, filename: str):
        """
        Load model checkpoint

        Args:
            filename: Checkpoint filename
        """
        checkpoint = torch.load(filename, map_location=self.config.device)

        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.best_val_loss = checkpoint['best_val_loss']

        print(f"Loaded checkpoint: {filename}")
        print(f"  Best val loss: {self.best_val_loss:.4f}")

    def generate_sample(self, num_tokens: int = 100):
        """
        Generate sample text

        Args:
            num_tokens: Number of tokens to generate
        """
        self.model.eval()

        # Start with newline
        context = torch.zeros((1, 1), dtype=torch.long, device=self.config.device)

        # Generate
        generated = self.model.generate(
            context,
            max_new_tokens=num_tokens,
            temperature=0.8,
            top_k=200
        )

        # Decode
        text = self.train_dataset.decode(generated[0])

        print(f"\n--- Generated Sample ---")
        print(text)
        print("--- End Sample ---\n")

        self.model.train()


# Demo / Main
if __name__ == "__main__":
    print("="*80)
    print("COMPLETE PRETRAINING PIPELINE")
    print("="*80)

    # Configuration
    config = ModelConfig(
        vocab_size=65,  # Will be updated by dataset
        n_layers=4,
        n_heads=4,
        n_embd=128,
        block_size=64,
        batch_size=16,
        learning_rate=1e-3,
        max_iters=1000,
        eval_interval=100,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"\nDevice: {config.device}")

    # Load data
    print("\n--- Loading Data ---")
    text = DataModule.load_shakespeare()
    train_text, val_text = DataModule.create_splits(text)

    # Create datasets
    train_dataset = TextDataset(train_text, config.block_size, char_level=True)
    val_dataset = TextDataset(val_text, config.block_size, char_level=True)

    # Update config with vocab size
    config.vocab_size = train_dataset.vocab_size

    # Create model
    print("\n--- Creating Model ---")
    model = GPT(config)

    # Create trainer
    trainer = Trainer(model, config, train_dataset, val_dataset)

    # Train
    trainer.train()

    # Save final model
    trainer.save_checkpoint("final_model.pt")

    # Final generation
    print("\n" + "="*80)
    print("FINAL GENERATION (after training)")
    print("="*80)
    trainer.generate_sample(num_tokens=200)

    print("\n✅ PRETRAINING COMPLETE!")
    print("="*80)
    print("""
What we built:
  ✅ Complete GPT architecture
  ✅ Character-level dataset
  ✅ Training loop with:
    • AdamW optimizer
    • Gradient clipping
    • Evaluation
    • Checkpointing
    • Generation sampling

Next steps:
  • Scale to larger models (100M+ params)
  • Use token-level (GPT-2 tokenizer)
  • Train on massive datasets (100GB+)
  • Distributed training (multi-GPU)
  • See Chapter 7 for fine-tuning!
    """)
```

*[Suite avec Distributed Training et Monitoring dans la partie 3...]*
