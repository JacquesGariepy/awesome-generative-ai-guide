# Chapitre 9 (Suite 2): Projet Pratique Instruction Tuning

## 6. Projet Complet: Instruction-Tune Llama 2 7B

```python
"""
Projet complet: Instruction tuning de Llama 2 7B

Ce projet montre comment:
1. Préparer dataset d'instruction (Alpaca format)
2. Appliquer chat template approprié
3. Fine-tuner avec loss masking
4. Évaluer le modèle
5. Déployer

Utilise: QLoRA pour efficacité mémoire
"""

import os
from typing import List, Dict, Optional
from dataclasses import dataclass, field

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import numpy as np


# =====================
# 1. Configuration
# =====================

@dataclass
class InstructionTuningConfig:
    """Configuration pour instruction tuning"""

    # Model
    model_name: str = "meta-llama/Llama-2-7b-hf"

    # Dataset
    dataset_name: str = "tatsu-lab/alpaca"  # ou "vicgalle/alpaca-gpt4"
    num_train_samples: int = 10000  # -1 pour tout
    num_eval_samples: int = 500

    # Instruction format
    instruction_template: str = "### Instruction:"
    input_template: str = "### Input:"
    response_template: str = "### Response:"

    # Quantization (QLoRA)
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True

    # LoRA
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])

    # Training
    output_dir: str = "./llama2-7b-instruct"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    max_seq_length: int = 512
    warmup_ratio: float = 0.03

    # Logging
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 100


# =====================
# 2. Data Processing avec Loss Masking
# =====================

class InstructionDataCollator:
    """
    Data collator pour instruction tuning

    Key feature: Loss masking
    - Compute loss SEULEMENT sur response
    - Ignore instruction et input (labels = -100)
    """

    def __init__(
        self,
        tokenizer,
        response_template: str = "### Response:",
        instruction_template: str = "### Instruction:",
        mlm: bool = False
    ):
        """
        Args:
            tokenizer: Tokenizer
            response_template: Template marquant le début de la réponse
            instruction_template: Template marquant l'instruction
            mlm: Masked language modeling (False pour causal LM)
        """
        self.tokenizer = tokenizer
        self.response_template = response_template
        self.instruction_template = instruction_template
        self.mlm = mlm

        # Tokenize templates pour trouver les positions
        self.response_template_ids = tokenizer.encode(
            response_template,
            add_special_tokens=False
        )

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate batch avec loss masking

        Args:
            features: Liste de examples

        Returns:
            Batch dict
        """

        # Extract input_ids et labels
        input_ids = [f["input_ids"] for f in features]
        labels = [f["labels"] for f in features]

        # Pad
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(ids) for ids in input_ids],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )

        labels = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(lbls) for lbls in labels],
            batch_first=True,
            padding_value=-100  # Ignore index
        )

        # Attention mask
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


class InstructionDatasetProcessor:
    """Process dataset pour instruction tuning"""

    def __init__(
        self,
        tokenizer,
        config: InstructionTuningConfig
    ):
        self.tokenizer = tokenizer
        self.config = config

        # Template tokens
        self.response_template = config.response_template
        self.instruction_template = config.instruction_template
        self.input_template = config.input_template

    def format_instruction(self, example: Dict) -> str:
        """
        Format example en instruction format

        Args:
            example: Dict avec 'instruction', 'input', 'output'

        Returns:
            Formatted string
        """

        instruction = example.get('instruction', '')
        input_text = example.get('input', '')
        output = example.get('output', '')

        if input_text:
            formatted = f"""{self.instruction_template}
{instruction}

{self.input_template}
{input_text}

{self.response_template}
{output}"""
        else:
            formatted = f"""{self.instruction_template}
{instruction}

{self.response_template}
{output}"""

        return formatted

    def tokenize_with_masking(self, example: Dict) -> Dict:
        """
        Tokenize avec loss masking

        Stratégie:
        1. Tokenize le texte complet
        2. Find position du response template
        3. Mask tout avant response (labels = -100)
        4. Keep labels pour response

        Args:
            example: Example dict

        Returns:
            Dict avec input_ids et labels
        """

        # Format
        full_text = self.format_instruction(example)

        # Tokenize
        tokenized = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.config.max_seq_length,
            padding=False,
            return_tensors=None
        )

        input_ids = tokenized["input_ids"]

        # Create labels (copy of input_ids)
        labels = input_ids.copy()

        # Find response template position
        response_template_ids = self.tokenizer.encode(
            self.response_template,
            add_special_tokens=False
        )

        # Search for template
        response_start = None
        for i in range(len(input_ids) - len(response_template_ids)):
            if input_ids[i:i+len(response_template_ids)] == response_template_ids:
                response_start = i + len(response_template_ids)
                break

        # Mask everything before response
        if response_start is not None:
            labels[:response_start] = [-100] * response_start
        else:
            # Si template pas trouvé, mask tout (sécurité)
            labels = [-100] * len(labels)

        return {
            "input_ids": input_ids,
            "labels": labels
        }

    def prepare_dataset(self):
        """
        Load et prepare dataset

        Returns:
            (train_dataset, eval_dataset)
        """

        print("="*60)
        print("Loading and Processing Dataset")
        print("="*60)
        print()

        # Load dataset
        dataset = load_dataset(self.config.dataset_name, split="train")

        # Subset if needed
        if self.config.num_train_samples > 0:
            total_samples = self.config.num_train_samples + self.config.num_eval_samples
            dataset = dataset.select(range(min(total_samples, len(dataset))))

        # Split train/eval
        split = dataset.train_test_split(
            test_size=self.config.num_eval_samples,
            seed=42
        )

        train_dataset = split['train']
        eval_dataset = split['test']

        # Process
        print(f"Processing {len(train_dataset)} training examples...")
        train_dataset = train_dataset.map(
            self.tokenize_with_masking,
            remove_columns=train_dataset.column_names,
            desc="Processing train"
        )

        print(f"Processing {len(eval_dataset)} eval examples...")
        eval_dataset = eval_dataset.map(
            self.tokenize_with_masking,
            remove_columns=eval_dataset.column_names,
            desc="Processing eval"
        )

        print()
        print("✅ Dataset prepared!")
        print(f"  Train: {len(train_dataset)} examples")
        print(f"  Eval: {len(eval_dataset)} examples")
        print()

        return train_dataset, eval_dataset


# =====================
# 3. Training Pipeline
# =====================

class InstructionTuningTrainer:
    """Pipeline complet pour instruction tuning"""

    def __init__(self, config: InstructionTuningConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None

    def setup_model(self):
        """Setup model et tokenizer avec quantization"""

        print("="*60)
        print("Loading Model")
        print("="*60)
        print()

        # Quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.config.load_in_4bit,
            bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=getattr(torch, self.config.bnb_4bit_compute_dtype),
            bnb_4bit_use_double_quant=self.config.bnb_4bit_use_double_quant
        )

        # Tokenizer
        print(f"Loading tokenizer: {self.config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True
        )

        # Add pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Model
        print(f"Loading model: {self.config.model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )

        # Prepare for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)

        # LoRA
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.lora_target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )

        self.model = get_peft_model(self.model, lora_config)

        print()
        self.model.print_trainable_parameters()
        print()
        print("✅ Model loaded!")
        print()

    def prepare_data(self):
        """Prepare datasets"""

        processor = InstructionDatasetProcessor(self.tokenizer, self.config)
        train_dataset, eval_dataset = processor.prepare_dataset()

        return train_dataset, eval_dataset

    def create_trainer(self, train_dataset, eval_dataset):
        """Create Trainer"""

        print("="*60)
        print("Creating Trainer")
        print("="*60)
        print()

        # Training args
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            logging_steps=self.config.logging_steps,
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            bf16=True,
            optim="paged_adamw_8bit",
            gradient_checkpointing=True,
            report_to=["tensorboard"],
            save_total_limit=3
        )

        # Data collator avec loss masking
        data_collator = InstructionDataCollator(
            tokenizer=self.tokenizer,
            response_template=self.config.response_template
        )

        # Trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator
        )

        print("✅ Trainer created!")
        print()

    def train(self):
        """Launch training"""

        print("="*60)
        print("Starting Training")
        print("="*60)
        print()

        # Train
        result = self.trainer.train()

        # Save
        print("\nSaving model...")
        self.trainer.save_model(self.config.output_dir)
        self.tokenizer.save_pretrained(self.config.output_dir)

        print()
        print("="*60)
        print("Training Complete!")
        print("="*60)
        print(f"Final train loss: {result.training_loss:.4f}")

        # Eval
        metrics = self.trainer.evaluate()
        print(f"Final eval loss: {metrics['eval_loss']:.4f}")
        print(f"Final perplexity: {np.exp(metrics['eval_loss']):.2f}")
        print()

        return result, metrics

    def run_full_pipeline(self):
        """Run complete pipeline"""

        # Setup
        self.setup_model()

        # Data
        train_dataset, eval_dataset = self.prepare_data()

        # Trainer
        self.create_trainer(train_dataset, eval_dataset)

        # Train
        result, metrics = self.train()

        return result, metrics


# =====================
# 4. Inference et Évaluation
# =====================

class InstructionTunedInference:
    """Inference avec modèle instruction-tuned"""

    def __init__(self, model_path: str, base_model_name: str):
        """
        Load fine-tuned model

        Args:
            model_path: Path vers LoRA weights
            base_model_name: Base model name
        """

        print(f"Loading instruction-tuned model from {model_path}...")

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Base model
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

        # Load LoRA
        from peft import PeftModel
        self.model = PeftModel.from_pretrained(base_model, model_path)

        print("✅ Model loaded!")

    def generate_response(
        self,
        instruction: str,
        input_text: str = "",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """
        Generate response à une instruction

        Args:
            instruction: L'instruction
            input_text: Input optionnel
            max_new_tokens: Max tokens à générer
            temperature: Sampling temperature
            top_p: Nucleus sampling

        Returns:
            Generated response
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
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract response
        if "### Response:" in full_response:
            response = full_response.split("### Response:")[-1].strip()
        else:
            response = full_response

        return response


class InstructionTuningEvaluator:
    """Evaluate instruction-tuned model"""

    def __init__(self, model_inference: InstructionTunedInference):
        self.model = model_inference

    def evaluate_helpfulness(
        self,
        test_instructions: List[Dict[str, str]]
    ) -> Dict[str, float]:
        """
        Evaluate helpfulness sur test set

        Args:
            test_instructions: Liste de dicts avec 'instruction', 'input', 'expected_output'

        Returns:
            Metrics
        """

        results = {
            "num_examples": len(test_instructions),
            "responses": [],
            "avg_length": 0
        }

        total_length = 0

        for example in test_instructions:
            instruction = example['instruction']
            input_text = example.get('input', '')

            response = self.model.generate_response(instruction, input_text)

            results["responses"].append({
                "instruction": instruction,
                "input": input_text,
                "response": response,
                "expected": example.get('expected_output', '')
            })

            total_length += len(response)

        results["avg_length"] = total_length / len(test_instructions)

        return results


# =====================
# 5. Main
# =====================

def main():
    """Main entry point"""

    print("="*60)
    print("Instruction Tuning Pipeline for Llama 2 7B")
    print("="*60)
    print()

    # Config
    config = InstructionTuningConfig(
        num_train_samples=1000,  # Demo
        num_eval_samples=100,
        num_train_epochs=1
    )

    # Training
    trainer = InstructionTuningTrainer(config)

    # Run (commented to not actually train)
    # result, metrics = trainer.run_full_pipeline()

    print("Training pipeline ready!")
    print("\nTo train:")
    print("  trainer = InstructionTuningTrainer(config)")
    print("  result, metrics = trainer.run_full_pipeline()")
    print("\nTo use:")
    print("  model = InstructionTunedInference('./llama2-7b-instruct', 'meta-llama/Llama-2-7b-hf')")
    print("  response = model.generate_response('Explain quantum physics in simple terms')")


if __name__ == "__main__":
    main()
```

## 7. Conclusion et Ressources

### Résumé du Chapitre

Dans ce chapitre, nous avons couvert:

1. **Formats de Données**
   - Alpaca (Stanford) - simple et populaire
   - ShareGPT - conversations multi-turn
   - OpenAI - API compatible

2. **Création de Données**
   - Self-Instruct (génération automatique)
   - Distillation depuis GPT-4
   - Augmentation de données

3. **Chat Templates**
   - Llama 2, ChatML, Alpaca, Vicuna, Zephyr
   - Formatage multi-turn
   - Loss masking

4. **Best Practices**
   - Quality > Quantity
   - Diversité des tâches
   - Éviter overfitting sur format
   - Loss masking essentiel

5. **Projet Pratique**
   - Pipeline complet QLoRA
   - Loss masking automatique
   - Évaluation

### Chiffres Clés

- **Dataset minimum**: 1k-5k exemples (10k+ recommandé)
- **Epochs**: 1-3 (éviter overfitting)
- **Learning rate**: 2e-4 à 5e-4 (QLoRA)
- **Loss masking**: Essentiel (compute loss seulement sur response)

### Datasets Populaires

| Dataset | Taille | Format | Qualité |
|---------|--------|--------|---------|
| Alpaca | 52k | Alpaca | ⭐⭐⭐ |
| Alpaca GPT-4 | 52k | Alpaca | ⭐⭐⭐⭐⭐ |
| ShareGPT | 52k | Multi-turn | ⭐⭐⭐⭐ |
| Dolly 15k | 15k | Alpaca | ⭐⭐⭐⭐⭐ (humain) |
| FLAN | 1.8M | Diverse | ⭐⭐⭐⭐ |

### Ressources

**Papers**:
- Self-Instruct: https://arxiv.org/abs/2212.10560
- Alpaca: https://crfm.stanford.edu/2023/03/13/alpaca.html
- Vicuna: https://lmsys.org/blog/2023-03-30-vicuna/

**Datasets**:
- Alpaca: https://huggingface.co/datasets/tatsu-lab/alpaca
- ShareGPT: https://huggingface.co/datasets/RyokoAI/ShareGPT52K
- FLAN: https://huggingface.co/datasets/conceptofmind/FLAN_2022

---

**Chapitre 9 terminé!** Vous maîtrisez maintenant l'instruction tuning de A à Z.

**Prochain chapitre**: Chapitre 10 - Retrieval-Augmented Generation (RAG)
