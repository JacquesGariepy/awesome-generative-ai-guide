# Chapitre 8 (Suite 3): Projet Pratique QLoRA

## 4. Projet Complet: Fine-tune Llama 2 7B avec QLoRA

```python
"""
Projet complet: Fine-tuning Llama 2 7B avec QLoRA

Ce projet montre comment:
1. Charger un modèle 7B en 4-bit (NF4)
2. Appliquer LoRA avec PEFT
3. Fine-tuner sur dataset personnalisé
4. Merge et export

Prérequis:
- GPU avec >=8GB VRAM (RTX 3060, RTX 4060, etc.)
- Libraries: transformers, peft, bitsandbytes, datasets

Installation:
pip install transformers>=4.30.0
pip install peft>=0.4.0
pip install bitsandbytes>=0.40.0
pip install datasets accelerate
"""

import os
from typing import Optional, Dict
from dataclasses import dataclass, field

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import load_dataset


# =====================
# 1. Configuration
# =====================

@dataclass
class QLoRAConfig:
    """Configuration complète pour QLoRA fine-tuning"""

    # Model
    model_name: str = "meta-llama/Llama-2-7b-hf"

    # Quantization config (BitsAndBytes)
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"  # Compute en BF16
    bnb_4bit_quant_type: str = "nf4"  # NormalFloat 4-bit
    bnb_4bit_use_double_quant: bool = True  # Double quantization

    # LoRA config
    lora_r: int = 64  # Rank (plus élevé = plus de capacité)
    lora_alpha: int = 16  # Scaling (typiquement rank/4 à rank/2)
    lora_dropout: float = 0.05
    lora_target_modules: list = field(default_factory=lambda: [
        "q_proj",  # Query projection
        "k_proj",  # Key projection
        "v_proj",  # Value projection
        "o_proj",  # Output projection
        "gate_proj",  # Gate projection (Llama 2)
        "up_proj",    # Up projection (Llama 2)
        "down_proj"   # Down projection (Llama 2)
    ])

    # Training
    output_dir: str = "./llama2-7b-qlora-finetuned"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4  # Effective batch = 16
    learning_rate: float = 2e-4  # Higher LR for LoRA
    warmup_ratio: float = 0.03
    weight_decay: float = 0.001
    max_seq_length: int = 512

    # Optimization
    optim: str = "paged_adamw_8bit"  # 8-bit Adam (from bitsandbytes)
    gradient_checkpointing: bool = True
    max_grad_norm: float = 0.3

    # Logging
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100


# =====================
# 2. Model Loading avec Quantization
# =====================

class QLoRAModelLoader:
    """Load modèle avec quantization 4-bit"""

    @staticmethod
    def create_bnb_config(config: QLoRAConfig) -> BitsAndBytesConfig:
        """
        Create BitsAndBytes quantization config

        Returns:
            BitsAndBytesConfig for 4-bit quantization
        """

        compute_dtype = getattr(torch, config.bnb_4bit_compute_dtype)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=config.load_in_4bit,
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
        )

        return bnb_config

    @staticmethod
    def load_model(config: QLoRAConfig):
        """
        Load model avec quantization 4-bit

        Returns:
            Tuple de (model, tokenizer)
        """

        print("="*60)
        print("Loading Model with 4-bit Quantization")
        print("="*60)
        print()

        # BitsAndBytes config
        bnb_config = QLoRAModelLoader.create_bnb_config(config)

        # Load tokenizer
        print(f"Loading tokenizer: {config.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            trust_remote_code=True
        )

        # Add pad token if missing
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # Load model with quantization
        print(f"Loading model: {config.model_name}")
        print(f"  Quantization: {config.bnb_4bit_quant_type}")
        print(f"  Compute dtype: {config.bnb_4bit_compute_dtype}")
        print(f"  Double quant: {config.bnb_4bit_use_double_quant}")

        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            quantization_config=bnb_config,
            device_map="auto",  # Automatically distribute
            trust_remote_code=True
        )

        # Prepare for k-bit training
        model = prepare_model_for_kbit_training(model)

        print()
        print(f"✅ Model loaded successfully!")
        print(f"   Total params: {model.num_parameters():,}")
        print(f"   Memory footprint: ~{model.get_memory_footprint() / 1e9:.2f} GB")
        print()

        return model, tokenizer


# =====================
# 3. LoRA Configuration
# =====================

class QLoRAPEFTConfig:
    """Configure PEFT avec LoRA"""

    @staticmethod
    def create_lora_config(config: QLoRAConfig) -> LoraConfig:
        """
        Create LoRA configuration

        Returns:
            LoraConfig pour PEFT
        """

        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=config.lora_target_modules,
            lora_dropout=config.lora_dropout,
            bias="none",  # Don't train biases
            task_type=TaskType.CAUSAL_LM
        )

        return lora_config

    @staticmethod
    def apply_lora(model, config: QLoRAConfig):
        """
        Apply LoRA to model

        Returns:
            PEFT model avec LoRA adapters
        """

        print("="*60)
        print("Applying LoRA")
        print("="*60)
        print()

        lora_config = QLoRAPEFTConfig.create_lora_config(config)

        # Apply LoRA
        model = get_peft_model(model, lora_config)

        # Print trainable parameters
        model.print_trainable_parameters()

        print()
        print("✅ LoRA applied successfully!")
        print()

        return model


# =====================
# 4. Dataset Preparation
# =====================

class DatasetPreparator:
    """Prepare dataset for training"""

    def __init__(self, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def format_instruction(self, example: Dict) -> str:
        """
        Format example en instruction format

        Modify selon votre dataset
        """

        # Example format pour Alpaca-style dataset
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        output = example.get("output", "")

        if input_text:
            prompt = f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output}"""
        else:
            prompt = f"""### Instruction:
{instruction}

### Response:
{output}"""

        return prompt

    def tokenize_function(self, examples):
        """Tokenize examples"""

        # Format all examples
        texts = [self.format_instruction(ex) for ex in examples]

        # Tokenize
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors=None
        )

        # Labels = input_ids for causal LM
        tokenized["labels"] = tokenized["input_ids"].copy()

        return tokenized

    def prepare_dataset(self, dataset_name: str = "tatsu-lab/alpaca"):
        """
        Load and prepare dataset

        Args:
            dataset_name: HuggingFace dataset name

        Returns:
            Processed dataset
        """

        print("="*60)
        print("Loading Dataset")
        print("="*60)
        print()

        # Load dataset
        dataset = load_dataset(dataset_name, split="train")

        # Take subset for demo (remove for full training)
        dataset = dataset.select(range(1000))  # First 1000 examples

        # Tokenize
        print(f"Tokenizing {len(dataset)} examples...")
        tokenized_dataset = dataset.map(
            lambda examples: self.tokenize_function(examples),
            batched=False,
            remove_columns=dataset.column_names
        )

        print(f"✅ Dataset prepared: {len(tokenized_dataset)} examples")
        print()

        return tokenized_dataset


# =====================
# 5. Training Pipeline
# =====================

class QLoRATrainer:
    """Complete QLoRA training pipeline"""

    def __init__(self, config: QLoRAConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None

    def setup(self):
        """Setup model, tokenizer, and LoRA"""

        # Load model with quantization
        self.model, self.tokenizer = QLoRAModelLoader.load_model(self.config)

        # Apply LoRA
        self.model = QLoRAPEFTConfig.apply_lora(self.model, self.config)

    def prepare_data(self):
        """Prepare training dataset"""

        preparator = DatasetPreparator(
            self.tokenizer,
            max_length=self.config.max_seq_length
        )

        train_dataset = preparator.prepare_dataset()

        return train_dataset

    def create_trainer(self, train_dataset):
        """Create HuggingFace Trainer"""

        print("="*60)
        print("Creating Trainer")
        print("="*60)
        print()

        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            optim=self.config.optim,
            fp16=False,
            bf16=True,  # Use BF16 for training
            gradient_checkpointing=self.config.gradient_checkpointing,
            max_grad_norm=self.config.max_grad_norm,
            report_to=["tensorboard"],
            save_total_limit=3,
        )

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # Causal LM, not masked LM
        )

        # Trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )

        effective_batch = (
            self.config.per_device_train_batch_size *
            self.config.gradient_accumulation_steps *
            torch.cuda.device_count()
        )

        print(f"Effective batch size: {effective_batch}")
        print(f"Total training steps: {len(train_dataset) // effective_batch * self.config.num_train_epochs}")
        print()
        print("✅ Trainer created!")
        print()

    def train(self):
        """Launch training"""

        print("="*60)
        print("Starting Training")
        print("="*60)
        print()

        # Train
        train_result = self.trainer.train()

        # Save final model
        print("\nSaving final model...")
        self.trainer.save_model(self.config.output_dir)

        print()
        print("="*60)
        print("Training Complete!")
        print("="*60)
        print(f"Final train loss: {train_result.training_loss:.4f}")
        print(f"Model saved to: {self.config.output_dir}")
        print()

        return train_result

    def run_full_pipeline(self):
        """Run complete pipeline"""

        # Setup
        self.setup()

        # Prepare data
        train_dataset = self.prepare_data()

        # Create trainer
        self.create_trainer(train_dataset)

        # Train
        result = self.train()

        return result


# =====================
# 6. Inference avec modèle QLoRA
# =====================

class QLoRAInference:
    """Inference avec modèle fine-tuné QLoRA"""

    def __init__(self, model_path: str, base_model_name: str):
        """
        Load fine-tuned LoRA model

        Args:
            model_path: Path vers LoRA adapter weights
            base_model_name: Nom du modèle de base
        """

        print(f"Loading fine-tuned model from {model_path}...")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Load base model avec quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )

        # Load LoRA adapters
        from peft import PeftModel
        self.model = PeftModel.from_pretrained(base_model, model_path)

        print("✅ Model loaded!")

    def generate(
        self,
        instruction: str,
        input_text: str = "",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """
        Generate response

        Args:
            instruction: Instruction
            input_text: Input (optional)
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling

        Returns:
            Generated text
        """

        # Format prompt
        if input_text:
            prompt = f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""
        else:
            prompt = f"""### Instruction:
{instruction}

### Response:
"""

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.model.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the response part
        if "### Response:" in response:
            response = response.split("### Response:")[-1].strip()

        return response


# =====================
# 7. Main
# =====================

def main():
    """Main entry point"""

    print("="*60)
    print("QLoRA Fine-tuning Pipeline for Llama 2 7B")
    print("="*60)
    print()

    # Configuration
    config = QLoRAConfig(
        num_train_epochs=1,  # 1 epoch for demo
        per_device_train_batch_size=2,  # Adjust based on VRAM
        gradient_accumulation_steps=8,  # Effective batch = 16
    )

    # Training pipeline
    trainer = QLoRATrainer(config)

    # Run (commented out to not actually train)
    # result = trainer.run_full_pipeline()

    print("Training pipeline ready!")
    print("\nTo train:")
    print("  trainer = QLoRATrainer(config)")
    print("  result = trainer.run_full_pipeline()")
    print("\nTo use the fine-tuned model:")
    print("  inference = QLoRAInference('./llama2-7b-qlora-finetuned', 'meta-llama/Llama-2-7b-hf')")
    print("  response = inference.generate('Explain quantum computing', '')")


if __name__ == "__main__":
    main()
```

## 5. Merge LoRA Adapters (Optional)

```python
"""
Merge LoRA adapters into base model pour déploiement

Avantages:
- Pas besoin de PEFT au runtime
- Inference plus rapide (pas d'overhead)
- Compatible avec toutes les libraries

Inconvénient:
- Perd la modularité (cannot swap adapters)
"""

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


class LoRAMerger:
    """Merge LoRA adapters dans base model"""

    @staticmethod
    def merge_and_save(
        base_model_name: str,
        lora_adapter_path: str,
        output_path: str
    ):
        """
        Merge LoRA adapters et save merged model

        Args:
            base_model_name: Nom du modèle de base
            lora_adapter_path: Path vers LoRA adapters
            output_path: Output path pour merged model
        """

        print("="*60)
        print("Merging LoRA Adapters")
        print("="*60)
        print()

        # Load base model (FP16, not quantized pour merge)
        print("Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        # Load LoRA adapters
        print("Loading LoRA adapters...")
        model = PeftModel.from_pretrained(base_model, lora_adapter_path)

        # Merge
        print("Merging adapters into base model...")
        merged_model = model.merge_and_unload()

        # Save
        print(f"Saving merged model to {output_path}...")
        merged_model.save_pretrained(output_path)

        # Save tokenizer
        tokenizer = AutoTokenizer.from_pretrained(lora_adapter_path)
        tokenizer.save_pretrained(output_path)

        print()
        print("✅ Merge complete!")
        print(f"   Merged model saved to: {output_path}")
        print()

        return merged_model


# Exemple
if __name__ == "__main__":
    # Merge LoRA adapters
    # LoRAMerger.merge_and_save(
    #     base_model_name="meta-llama/Llama-2-7b-hf",
    #     lora_adapter_path="./llama2-7b-qlora-finetuned",
    #     output_path="./llama2-7b-merged"
    # )

    print("LoRA Merger ready!")
    print("\nTo merge:")
    print("  LoRAMerger.merge_and_save(")
    print("      base_model_name='meta-llama/Llama-2-7b-hf',")
    print("      lora_adapter_path='./llama2-7b-qlora-finetuned',")
    print("      output_path='./llama2-7b-merged'")
    print("  )")
```

## 6. Best Practices et Recommandations

```python
"""
Best practices pour QLoRA fine-tuning
"""

QLORA_BEST_PRACTICES = {
    "Configuration LoRA": {
        "rank (r)": {
            "small tasks": "8-16",
            "medium tasks": "16-32",
            "complex tasks": "32-64",
            "very complex": "64-128",
            "note": "Plus le rank est élevé, plus de capacité mais plus de paramètres"
        },

        "alpha": {
            "rule of thumb": "rank // 2 à rank * 2",
            "common values": "16, 32, 64",
            "scaling": "alpha / rank détermine l'importance de LoRA"
        },

        "target_modules": {
            "minimal": ["q_proj", "v_proj"],
            "recommended": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "aggressive": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "note": "Plus de modules = meilleure performance mais plus de paramètres"
        },

        "dropout": {
            "small dataset": "0.1",
            "medium dataset": "0.05",
            "large dataset": "0.01",
            "note": "Prévient l'overfitting"
        }
    },

    "Training Hyperparameters": {
        "learning_rate": {
            "LoRA FP16": "1e-4 à 3e-4",
            "QLoRA": "2e-4 à 5e-4",
            "note": "LoRA peut supporter LR plus élevé que full FT"
        },

        "batch_size": {
            "8GB VRAM": "1-2 per device",
            "16GB VRAM": "2-4 per device",
            "24GB VRAM": "4-8 per device",
            "note": "Utiliser gradient accumulation pour effective batch ≥16"
        },

        "optimizer": {
            "recommended": "paged_adamw_8bit",
            "alternative": "adamw_torch",
            "note": "paged_adamw_8bit économise encore plus de mémoire"
        }
    },

    "Common Issues": {
        "OOM (Out of Memory)": [
            "Réduire batch size",
            "Augmenter gradient accumulation",
            "Activer gradient checkpointing",
            "Réduire max_seq_length",
            "Utiliser paged_adamw_8bit optimizer"
        ],

        "Loss ne descend pas": [
            "Augmenter learning rate",
            "Augmenter LoRA rank",
            "Ajouter plus de target modules",
            "Vérifier la qualité des données"
        ],

        "Overfitting": [
            "Augmenter dropout",
            "Réduire rank",
            "Augmenter weight decay",
            "Plus de données d'entraînement"
        ]
    }
}


def print_best_practices():
    """Print best practices"""

    print("="*60)
    print("QLORA BEST PRACTICES")
    print("="*60)
    print()

    for category, items in QLORA_BEST_PRACTICES.items():
        print(f"\n### {category}")
        print()

        for key, value in items.items():
            print(f"**{key}**:")
            if isinstance(value, dict):
                for k, v in value.items():
                    print(f"  {k}: {v}")
            elif isinstance(value, list):
                for item in value:
                    print(f"  • {item}")
            else:
                print(f"  {value}")
            print()


if __name__ == "__main__":
    print_best_practices()
```

---

## Conclusion du Chapitre 8

### Ce que vous avez appris

1. **LoRA Theory**: Théorie mathématique du low-rank adaptation
2. **PEFT Methods**: Adapters, Prefix Tuning, IA3, et comparaisons
3. **Quantization**: FP32 → FP16 → INT8 → NF4 (4-bit)
4. **QLoRA**: Combine quantization + LoRA pour fine-tuning accessible
5. **Practical Project**: Pipeline complet QLoRA avec HuggingFace

### Chiffres Clés

- **LoRA**: 0.1-1% des paramètres, 95-99% performance
- **QLoRA**: Fine-tune 70B sur 24GB VRAM (impossible avant!)
- **NF4**: 8x compression avec ~1% perte de qualité
- **Double Quantization**: ~4.2 bits/param (vs 32 bits FP32)

### Recommendations Finales

| Scenario | Solution |
|----------|----------|
| GPU 8GB, modèle 7B | QLoRA rank=16-32 |
| GPU 24GB, modèle 13B | LoRA FP16 rank=32-64 |
| GPU 40GB+, modèle 70B | QLoRA rank=64 |
| Multi-task | LoRA (swap adapters) |
| Production | Merge LoRA → déployer |

### Ressources

**Papers**:
- LoRA: https://arxiv.org/abs/2106.09685
- QLoRA: https://arxiv.org/abs/2305.14314
- Adapters: https://arxiv.org/abs/1902.00751

**Libraries**:
- PEFT: https://github.com/huggingface/peft
- bitsandbytes: https://github.com/TimDettmers/bitsandbytes

---

**Chapitre 8 terminé!** Vous maîtrisez maintenant LoRA, QLoRA, et toutes les techniques PEFT.

**Prochain chapitre**: Chapitre 9 - Instruction Tuning et Alignment
