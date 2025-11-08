# Chapitre 7 (Suite 2): Hyperparamètres et Prévention du Surapprentissage

## 3.3 Hyperparamètres Critiques

```python
"""
Configuration complète des hyperparamètres pour fine-tuning
"""

from typing import Optional, Dict, List
from dataclasses import dataclass, field
from enum import Enum
import math


class LRSchedulerType(Enum):
    """Types de learning rate schedulers"""
    CONSTANT = "constant"
    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    POLYNOMIAL = "polynomial"
    INVERSE_SQRT = "inverse_sqrt"


class OptimizerType(Enum):
    """Types d'optimizers"""
    ADAM = "adam"
    ADAMW = "adamw"
    ADAFACTOR = "adafactor"
    SGD = "sgd"
    ADAGRAD = "adagrad"


@dataclass
class HyperparameterConfig:
    """
    Configuration complète des hyperparamètres pour fine-tuning

    Basé sur les best practices de HuggingFace et recherche récente
    """

    # Learning Rate
    learning_rate: float = 2e-5
    """
    Learning rate initial

    Recommandations:
    - Full fine-tuning: 1e-5 à 5e-5
    - LoRA: 1e-4 à 3e-4
    - QLoRA: 2e-4 à 5e-4

    Rule of thumb: Plus le modèle est grand, plus le LR doit être petit
    """

    lr_scheduler_type: LRSchedulerType = LRSchedulerType.COSINE
    """Type de scheduler pour learning rate"""

    warmup_ratio: float = 0.03
    """
    Pourcentage des steps pour warmup

    Recommandations:
    - Petit dataset (<10k): 0.1 (10%)
    - Moyen dataset: 0.03-0.05 (3-5%)
    - Grand dataset (>100k): 0.01 (1%)
    """

    warmup_steps: Optional[int] = None
    """Nombre de steps pour warmup (alternative à warmup_ratio)"""

    # Batch Size
    per_device_train_batch_size: int = 4
    """
    Batch size par GPU/CPU

    Recommandations selon VRAM:
    - 8GB VRAM: 1-2
    - 16GB VRAM: 2-4
    - 24GB VRAM: 4-8
    - 40GB+ VRAM: 8-16
    """

    per_device_eval_batch_size: int = 8
    """Batch size pour évaluation (peut être 2x train)"""

    gradient_accumulation_steps: int = 4
    """
    Nombre d'étapes d'accumulation de gradient

    Effective batch size = per_device_batch_size * gradient_accumulation_steps * num_gpus

    Recommandations:
    - Petit modèle (<1B): effective batch 32-64
    - Moyen modèle (1-7B): effective batch 64-128
    - Grand modèle (7B+): effective batch 128-256
    """

    # Training Duration
    num_train_epochs: int = 3
    """
    Nombre d'époques

    Recommandations:
    - Petit dataset (<1k): 5-10 epochs
    - Moyen dataset (1k-10k): 3-5 epochs
    - Grand dataset (>10k): 1-3 epochs
    """

    max_steps: int = -1
    """
    Nombre max de steps (override num_train_epochs si > 0)

    Utile pour:
    - Contrôle précis du training
    - Comparaisons équitables
    - Budgets de compute fixes
    """

    # Optimizer
    optimizer: OptimizerType = OptimizerType.ADAMW
    """Type d'optimizer"""

    adam_beta1: float = 0.9
    """Beta1 pour Adam/AdamW"""

    adam_beta2: float = 0.999
    """Beta2 pour Adam/AdamW"""

    adam_epsilon: float = 1e-8
    """Epsilon pour Adam/AdamW"""

    # Regularization
    weight_decay: float = 0.01
    """
    Weight decay (L2 regularization)

    Recommandations:
    - Petit dataset: 0.1
    - Moyen dataset: 0.01
    - Grand dataset: 0.001
    - Pas de regularization: 0.0
    """

    max_grad_norm: float = 1.0
    """
    Gradient clipping

    Recommandations:
    - Standard: 1.0
    - Unstable training: 0.5
    - Pas de clipping: 0.0 (not recommended)
    """

    # Dropout (if model supports it)
    dropout: float = 0.1
    """Dropout rate for regularization"""

    attention_dropout: float = 0.1
    """Dropout in attention layers"""

    # Mixed Precision
    fp16: bool = False
    """Use FP16 mixed precision (older GPUs)"""

    bf16: bool = True
    """
    Use BF16 mixed precision (recommended for newer GPUs)

    Requires:
    - Ampere GPUs (A100, RTX 3090, etc.)
    - Or newer (H100, etc.)
    """

    # Logging and Evaluation
    logging_steps: int = 10
    """Log metrics every N steps"""

    eval_steps: int = 500
    """Evaluate every N steps"""

    save_steps: int = 500
    """Save checkpoint every N steps"""

    save_total_limit: int = 3
    """Keep only N best checkpoints"""

    # Advanced
    gradient_checkpointing: bool = True
    """
    Enable gradient checkpointing to save memory

    Trade-off:
    - Pro: ~40% less VRAM
    - Con: ~20% slower training
    """

    optim: str = "adamw_torch"
    """
    Optimizer implementation

    Options:
    - adamw_torch: PyTorch AdamW
    - adamw_hf: HuggingFace AdamW
    - adafactor: Memory-efficient optimizer
    - adamw_8bit: 8-bit Adam (from bitsandbytes)
    """

    # Data
    max_seq_length: int = 512
    """Maximum sequence length"""

    dataloader_num_workers: int = 4
    """Number of workers for data loading"""

    dataloader_pin_memory: bool = True
    """Pin memory for faster GPU transfer"""

    # Seed
    seed: int = 42
    """Random seed for reproducibility"""

    def to_training_args(self) -> Dict[str, any]:
        """Convert to HuggingFace TrainingArguments dict"""

        return {
            "learning_rate": self.learning_rate,
            "lr_scheduler_type": self.lr_scheduler_type.value,
            "warmup_ratio": self.warmup_ratio,
            "warmup_steps": self.warmup_steps,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "per_device_eval_batch_size": self.per_device_eval_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "num_train_epochs": self.num_train_epochs,
            "max_steps": self.max_steps,
            "weight_decay": self.weight_decay,
            "max_grad_norm": self.max_grad_norm,
            "fp16": self.fp16,
            "bf16": self.bf16,
            "logging_steps": self.logging_steps,
            "eval_steps": self.eval_steps,
            "save_steps": self.save_steps,
            "save_total_limit": self.save_total_limit,
            "gradient_checkpointing": self.gradient_checkpointing,
            "optim": self.optim,
            "seed": self.seed,
            "dataloader_num_workers": self.dataloader_num_workers,
            "dataloader_pin_memory": self.dataloader_pin_memory,
        }

    def compute_effective_batch_size(self, num_gpus: int = 1) -> int:
        """Compute effective batch size"""

        return (
            self.per_device_train_batch_size *
            self.gradient_accumulation_steps *
            num_gpus
        )

    def estimate_training_time(
        self,
        num_examples: int,
        num_gpus: int = 1,
        tokens_per_second_per_gpu: int = 1000
    ) -> Dict[str, any]:
        """
        Estimate training time

        Args:
            num_examples: Number of training examples
            num_gpus: Number of GPUs
            tokens_per_second_per_gpu: Throughput per GPU

        Returns:
            Dict with time estimates
        """

        effective_batch = self.compute_effective_batch_size(num_gpus)

        if self.max_steps > 0:
            total_steps = self.max_steps
        else:
            steps_per_epoch = math.ceil(num_examples / effective_batch)
            total_steps = steps_per_epoch * self.num_train_epochs

        # Estimate tokens processed per step
        avg_seq_length = self.max_seq_length * 0.7  # Assume 70% of max
        tokens_per_step = effective_batch * avg_seq_length

        # Estimate time
        total_tokens = tokens_per_step * total_steps
        tokens_per_second_total = tokens_per_second_per_gpu * num_gpus
        seconds = total_tokens / tokens_per_second_total

        hours = seconds / 3600

        return {
            "total_steps": total_steps,
            "total_tokens": int(total_tokens),
            "estimated_hours": round(hours, 2),
            "estimated_days": round(hours / 24, 2),
            "effective_batch_size": effective_batch,
            "tokens_per_second": tokens_per_second_total
        }


class HyperparameterPresets:
    """Presets pour différents cas d'usage"""

    @staticmethod
    def quick_experiment() -> HyperparameterConfig:
        """Configuration pour expérimentations rapides"""

        return HyperparameterConfig(
            learning_rate=5e-5,
            num_train_epochs=1,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=1,
            logging_steps=10,
            eval_steps=100,
            save_steps=100,
            warmup_ratio=0.0,
            gradient_checkpointing=False,  # Faster but more VRAM
        )

    @staticmethod
    def standard_full_finetuning() -> HyperparameterConfig:
        """Configuration standard pour full fine-tuning"""

        return HyperparameterConfig(
            learning_rate=2e-5,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            warmup_ratio=0.03,
            weight_decay=0.01,
            lr_scheduler_type=LRSchedulerType.COSINE,
            gradient_checkpointing=True,
            bf16=True,
            save_total_limit=3,
        )

    @staticmethod
    def lora_finetuning() -> HyperparameterConfig:
        """Configuration pour LoRA fine-tuning"""

        return HyperparameterConfig(
            learning_rate=1e-4,  # Higher LR for LoRA
            num_train_epochs=3,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=2,
            warmup_ratio=0.05,
            weight_decay=0.01,
            lr_scheduler_type=LRSchedulerType.LINEAR,
            gradient_checkpointing=True,
            bf16=True,
        )

    @staticmethod
    def qlora_finetuning() -> HyperparameterConfig:
        """Configuration pour QLoRA (4-bit)"""

        return HyperparameterConfig(
            learning_rate=2e-4,  # Higher LR for QLoRA
            num_train_epochs=3,
            per_device_train_batch_size=16,  # Can fit larger batches
            gradient_accumulation_steps=1,
            warmup_ratio=0.03,
            weight_decay=0.0,  # Often not needed with 4-bit
            lr_scheduler_type=LRSchedulerType.COSINE,
            gradient_checkpointing=True,
            bf16=True,
            optim="paged_adamw_8bit",  # Memory-efficient
        )

    @staticmethod
    def small_dataset() -> HyperparameterConfig:
        """Configuration pour petit dataset (<1000 exemples)"""

        return HyperparameterConfig(
            learning_rate=3e-5,
            num_train_epochs=10,  # More epochs
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,
            warmup_ratio=0.1,  # More warmup
            weight_decay=0.1,  # More regularization
            lr_scheduler_type=LRSchedulerType.COSINE,
            eval_steps=50,  # Evaluate more often
            save_steps=50,
        )

    @staticmethod
    def large_dataset() -> HyperparameterConfig:
        """Configuration pour grand dataset (>100k exemples)"""

        return HyperparameterConfig(
            learning_rate=1e-5,  # Lower LR
            num_train_epochs=1,  # Fewer epochs
            per_device_train_batch_size=8,
            gradient_accumulation_steps=4,
            warmup_ratio=0.01,  # Less warmup
            weight_decay=0.001,  # Less regularization
            lr_scheduler_type=LRSchedulerType.LINEAR,
            eval_steps=1000,  # Evaluate less often
            save_steps=5000,
        )


# Exemple d'utilisation
if __name__ == "__main__":
    print("=== Presets de Configuration ===\n")

    presets = {
        "Quick Experiment": HyperparameterPresets.quick_experiment(),
        "Standard Full Fine-tuning": HyperparameterPresets.standard_full_finetuning(),
        "LoRA": HyperparameterPresets.lora_finetuning(),
        "QLoRA": HyperparameterPresets.qlora_finetuning(),
        "Small Dataset": HyperparameterPresets.small_dataset(),
        "Large Dataset": HyperparameterPresets.large_dataset(),
    }

    for name, config in presets.items():
        print(f"### {name}")
        print(f"  Learning Rate: {config.learning_rate}")
        print(f"  Epochs: {config.num_train_epochs}")
        print(f"  Batch Size: {config.per_device_train_batch_size}")
        print(f"  Gradient Accumulation: {config.gradient_accumulation_steps}")
        print(f"  Effective Batch (1 GPU): {config.compute_effective_batch_size(1)}")
        print(f"  Weight Decay: {config.weight_decay}")
        print(f"  LR Scheduler: {config.lr_scheduler_type.value}")
        print()

    print("\n" + "="*60 + "\n")

    # Estimation de temps de training
    print("=== Estimation de Temps de Training ===\n")

    config = HyperparameterPresets.standard_full_finetuning()

    scenarios = [
        {"name": "Small Dataset", "examples": 1000, "gpus": 1},
        {"name": "Medium Dataset", "examples": 10000, "gpus": 1},
        {"name": "Large Dataset", "examples": 100000, "gpus": 4},
    ]

    for scenario in scenarios:
        estimate = config.estimate_training_time(
            num_examples=scenario["examples"],
            num_gpus=scenario["gpus"]
        )

        print(f"{scenario['name']}:")
        print(f"  Examples: {scenario['examples']:,}")
        print(f"  GPUs: {scenario['gpus']}")
        print(f"  Total Steps: {estimate['total_steps']:,}")
        print(f"  Effective Batch: {estimate['effective_batch_size']}")
        print(f"  Estimated Time: {estimate['estimated_hours']:.1f} hours ({estimate['estimated_days']:.1f} days)")
        print()
```

## 3.4 Prévention du Surapprentissage (Overfitting)

```python
"""
Techniques et stratégies pour prévenir le surapprentissage
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from enum import Enum


class OverfittingSignal(Enum):
    """Signaux d'overfitting"""
    TRAIN_EVAL_DIVERGENCE = "train_eval_divergence"
    DECREASING_EVAL_PERFORMANCE = "decreasing_eval_performance"
    PERFECT_TRAIN_METRICS = "perfect_train_metrics"
    HIGH_VARIANCE = "high_variance"
    MEMORIZATION = "memorization"


@dataclass
class TrainingMetrics:
    """Métriques de training à un point donné"""
    step: int
    epoch: float
    train_loss: float
    eval_loss: Optional[float] = None
    train_perplexity: Optional[float] = None
    eval_perplexity: Optional[float] = None
    learning_rate: Optional[float] = None


class OverfittingDetector:
    """
    Détecte les signaux d'overfitting pendant le training

    Utilise plusieurs heuristiques pour identifier l'overfitting:
    1. Divergence train/eval loss
    2. Augmentation de eval loss alors que train loss diminue
    3. Train loss proche de 0 mais eval loss élevé
    """

    def __init__(
        self,
        divergence_threshold: float = 0.5,
        patience: int = 3,
        min_delta: float = 0.01
    ):
        """
        Args:
            divergence_threshold: Seuil de divergence train/eval
            patience: Nombre d'évaluations sans amélioration
            min_delta: Amélioration minimale considérée significative
        """
        self.divergence_threshold = divergence_threshold
        self.patience = patience
        self.min_delta = min_delta

        self.history: List[TrainingMetrics] = []
        self.best_eval_loss = float('inf')
        self.patience_counter = 0
        self.signals: Dict[str, List[int]] = {
            signal.value: [] for signal in OverfittingSignal
        }

    def add_metrics(self, metrics: TrainingMetrics) -> Dict[str, any]:
        """
        Ajoute des métriques et détecte l'overfitting

        Returns:
            Dict avec signaux d'overfitting détectés
        """
        self.history.append(metrics)

        detected_signals = []

        if metrics.eval_loss is not None:
            # Check 1: Divergence train/eval
            if self._check_divergence(metrics):
                detected_signals.append(OverfittingSignal.TRAIN_EVAL_DIVERGENCE)
                self.signals[OverfittingSignal.TRAIN_EVAL_DIVERGENCE.value].append(
                    metrics.step
                )

            # Check 2: Eval performance decreasing
            if self._check_eval_degradation(metrics):
                detected_signals.append(OverfittingSignal.DECREASING_EVAL_PERFORMANCE)
                self.signals[OverfittingSignal.DECREASING_EVAL_PERFORMANCE.value].append(
                    metrics.step
                )

            # Check 3: Perfect train but poor eval
            if self._check_perfect_train(metrics):
                detected_signals.append(OverfittingSignal.PERFECT_TRAIN_METRICS)
                self.signals[OverfittingSignal.PERFECT_TRAIN_METRICS.value].append(
                    metrics.step
                )

        # Check 4: High variance in train loss
        if self._check_variance():
            detected_signals.append(OverfittingSignal.HIGH_VARIANCE)

        return {
            "step": metrics.step,
            "is_overfitting": len(detected_signals) > 0,
            "signals": [s.value for s in detected_signals],
            "should_stop": self.patience_counter >= self.patience,
            "patience_counter": self.patience_counter,
        }

    def _check_divergence(self, metrics: TrainingMetrics) -> bool:
        """Vérifie si train et eval divergent"""

        if metrics.eval_loss is None:
            return False

        # Divergence = (eval_loss - train_loss) / train_loss
        divergence = (metrics.eval_loss - metrics.train_loss) / metrics.train_loss

        return divergence > self.divergence_threshold

    def _check_eval_degradation(self, metrics: TrainingMetrics) -> bool:
        """Vérifie si eval loss se dégrade"""

        if metrics.eval_loss is None:
            return False

        # Update best eval loss
        if metrics.eval_loss < self.best_eval_loss - self.min_delta:
            self.best_eval_loss = metrics.eval_loss
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            return self.patience_counter >= self.patience

    def _check_perfect_train(self, metrics: TrainingMetrics) -> bool:
        """Vérifie si train loss est très bas mais pas eval"""

        if metrics.eval_loss is None:
            return False

        # Train loss < 0.1 mais eval loss > 0.5
        return metrics.train_loss < 0.1 and metrics.eval_loss > 0.5

    def _check_variance(self) -> bool:
        """Vérifie la variance de train loss"""

        if len(self.history) < 10:
            return False

        # Prendre les 10 dernières métriques
        recent_losses = [m.train_loss for m in self.history[-10:]]
        variance = np.var(recent_losses)

        # Variance élevée = instabilité = possible overfitting
        return variance > 0.1

    def get_recommendation(self) -> Dict[str, any]:
        """Génère des recommandations basées sur les signaux"""

        recommendations = []

        # Compter les signaux
        signal_counts = {
            signal: len(steps)
            for signal, steps in self.signals.items()
        }

        # Recommandations basées sur les signaux
        if signal_counts.get(OverfittingSignal.TRAIN_EVAL_DIVERGENCE.value, 0) > 2:
            recommendations.append({
                "issue": "Train/Eval divergence détectée plusieurs fois",
                "solutions": [
                    "Augmenter weight_decay (régularisation L2)",
                    "Réduire learning rate",
                    "Ajouter plus de données d'entraînement",
                    "Utiliser data augmentation"
                ]
            })

        if signal_counts.get(OverfittingSignal.DECREASING_EVAL_PERFORMANCE.value, 0) > 0:
            recommendations.append({
                "issue": "Performance eval se dégrade",
                "solutions": [
                    "Arrêter le training (early stopping)",
                    "Revenir au meilleur checkpoint",
                    "Réduire le nombre d'époques",
                    "Augmenter dropout"
                ]
            })

        if signal_counts.get(OverfittingSignal.PERFECT_TRAIN_METRICS.value, 0) > 0:
            recommendations.append({
                "issue": "Train parfait mais eval médiocre (overfitting sévère)",
                "solutions": [
                    "Réduire drastiquement la complexité du modèle",
                    "Utiliser PEFT (LoRA) au lieu de full fine-tuning",
                    "Augmenter significativement le dataset",
                    "Augmenter weight_decay à 0.1+"
                ]
            })

        if signal_counts.get(OverfittingSignal.HIGH_VARIANCE.value, 0) > 0:
            recommendations.append({
                "issue": "Haute variance dans train loss",
                "solutions": [
                    "Réduire learning rate",
                    "Augmenter batch size",
                    "Utiliser gradient clipping plus agressif",
                    "Vérifier la qualité des données"
                ]
            })

        return {
            "signals_detected": signal_counts,
            "total_signals": sum(signal_counts.values()),
            "recommendations": recommendations,
            "should_stop_training": self.patience_counter >= self.patience
        }


class RegularizationStrategies:
    """Stratégies de régularisation pour prévenir l'overfitting"""

    @staticmethod
    def get_strategies() -> List[Dict[str, any]]:
        """Retourne toutes les stratégies de régularisation"""

        return [
            {
                "name": "Weight Decay (L2 Regularization)",
                "description": "Pénalise les poids élevés",
                "implementation": "Ajuster weight_decay dans optimizer",
                "when_to_use": "Toujours recommandé",
                "typical_values": {
                    "aggressive": 0.1,
                    "standard": 0.01,
                    "light": 0.001,
                    "none": 0.0
                },
                "code_example": """
# Dans TrainingArguments
training_args = TrainingArguments(
    weight_decay=0.01,  # Standard
    ...
)
                """
            },

            {
                "name": "Dropout",
                "description": "Désactive aléatoirement des neurones pendant training",
                "implementation": "Modifier architecture du modèle",
                "when_to_use": "Petit dataset, overfitting sévère",
                "typical_values": {
                    "aggressive": 0.3,
                    "standard": 0.1,
                    "light": 0.05
                },
                "code_example": """
# Lors du chargement du modèle
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1
)
                """
            },

            {
                "name": "Early Stopping",
                "description": "Arrête le training quand eval ne s'améliore plus",
                "implementation": "EarlyStoppingCallback",
                "when_to_use": "Toujours recommandé",
                "typical_values": {
                    "patient": {"patience": 5, "threshold": 0.01},
                    "standard": {"patience": 3, "threshold": 0.01},
                    "aggressive": {"patience": 2, "threshold": 0.005}
                },
                "code_example": """
from transformers import EarlyStoppingCallback

trainer = Trainer(
    ...
    callbacks=[EarlyStoppingCallback(
        early_stopping_patience=3,
        early_stopping_threshold=0.01
    )]
)
                """
            },

            {
                "name": "Data Augmentation",
                "description": "Augmente artificiellement la taille du dataset",
                "implementation": "Paraphrasing, back-translation, etc.",
                "when_to_use": "Petit dataset (<5k exemples)",
                "techniques": [
                    "Paraphrasing avec LLM",
                    "Back-translation",
                    "Synonym replacement",
                    "Random insertion/deletion"
                ],
                "code_example": """
# Exemple: Paraphrasing
def augment_with_paraphrase(example, model):
    prompt = f"Paraphrase this: {example['text']}"
    paraphrase = model.generate(prompt)
    return [example, {"text": paraphrase, ...}]

augmented_dataset = []
for ex in original_dataset:
    augmented_dataset.extend(
        augment_with_paraphrase(ex, paraphrase_model)
    )
                """
            },

            {
                "name": "Gradient Clipping",
                "description": "Limite la magnitude des gradients",
                "implementation": "max_grad_norm parameter",
                "when_to_use": "Training instable, exploding gradients",
                "typical_values": {
                    "aggressive": 0.5,
                    "standard": 1.0,
                    "light": 5.0
                },
                "code_example": """
training_args = TrainingArguments(
    max_grad_norm=1.0,  # Clip gradients
    ...
)
                """
            },

            {
                "name": "Learning Rate Scheduling",
                "description": "Réduit progressivement le learning rate",
                "implementation": "lr_scheduler_type parameter",
                "when_to_use": "Toujours recommandé",
                "schedules": {
                    "cosine": "Décroissance smooth (recommandé)",
                    "linear": "Décroissance linéaire",
                    "cosine_with_restarts": "Cosine avec restarts",
                    "constant_with_warmup": "Constant après warmup"
                },
                "code_example": """
training_args = TrainingArguments(
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,  # 3% warmup
    ...
)
                """
            },

            {
                "name": "Label Smoothing",
                "description": "Empêche le modèle d'être trop confiant",
                "implementation": "label_smoothing_factor parameter",
                "when_to_use": "Classification, overfitting sur labels",
                "typical_values": {
                    "aggressive": 0.2,
                    "standard": 0.1,
                    "light": 0.05
                },
                "code_example": """
training_args = TrainingArguments(
    label_smoothing_factor=0.1,
    ...
)
                """
            },

            {
                "name": "Parameter-Efficient Fine-Tuning (PEFT)",
                "description": "Fine-tune seulement une fraction des paramètres",
                "implementation": "LoRA, Adapters, Prefix Tuning",
                "when_to_use": "Petit dataset, ressources limitées",
                "methods": {
                    "LoRA": "Matrices low-rank (recommandé)",
                    "Adapters": "Petits layers additionnels",
                    "Prefix Tuning": "Tuning de prompts"
                },
                "code_example": """
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,  # Rank
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05
)

model = get_peft_model(base_model, lora_config)
# Seuls ~0.1% des paramètres sont trainables!
                """
            }
        ]


# Exemple d'utilisation
if __name__ == "__main__":
    print("=== Détection d'Overfitting ===\n")

    detector = OverfittingDetector(
        divergence_threshold=0.5,
        patience=3,
        min_delta=0.01
    )

    # Simuler un training avec overfitting progressif
    simulated_metrics = [
        TrainingMetrics(step=100, epoch=0.5, train_loss=1.5, eval_loss=1.6),
        TrainingMetrics(step=200, epoch=1.0, train_loss=1.0, eval_loss=1.1),
        TrainingMetrics(step=300, epoch=1.5, train_loss=0.7, eval_loss=0.9),
        TrainingMetrics(step=400, epoch=2.0, train_loss=0.5, eval_loss=0.85),  # Eval stagne
        TrainingMetrics(step=500, epoch=2.5, train_loss=0.3, eval_loss=0.88),  # Eval augmente
        TrainingMetrics(step=600, epoch=3.0, train_loss=0.2, eval_loss=0.95),  # Overfitting clair
        TrainingMetrics(step=700, epoch=3.5, train_loss=0.1, eval_loss=1.05),  # Pire
    ]

    for metrics in simulated_metrics:
        result = detector.add_metrics(metrics)

        print(f"Step {metrics.step}:")
        print(f"  Train Loss: {metrics.train_loss:.3f}, Eval Loss: {metrics.eval_loss:.3f}")
        print(f"  Overfitting: {result['is_overfitting']}")
        if result['signals']:
            print(f"  Signals: {', '.join(result['signals'])}")
        print(f"  Patience: {result['patience_counter']}/{detector.patience}")
        print()

        if result['should_stop']:
            print("⚠️  EARLY STOPPING TRIGGERED!")
            break

    print("\n" + "="*60 + "\n")

    # Recommandations
    recommendations = detector.get_recommendation()

    print("=== Recommandations ===\n")
    print(f"Total signaux détectés: {recommendations['total_signals']}")
    print(f"Arrêter training: {recommendations['should_stop_training']}")
    print()

    for rec in recommendations['recommendations']:
        print(f"⚠️  {rec['issue']}")
        print("Solutions:")
        for solution in rec['solutions']:
            print(f"  • {solution}")
        print()

    print("\n" + "="*60 + "\n")

    # Afficher stratégies de régularisation
    print("=== Stratégies de Régularisation ===\n")

    strategies = RegularizationStrategies.get_strategies()

    for i, strategy in enumerate(strategies[:3], 1):  # Afficher les 3 premières
        print(f"{i}. {strategy['name']}")
        print(f"   {strategy['description']}")
        print(f"   Quand l'utiliser: {strategy['when_to_use']}")
        print()
```

*[Suite avec évaluation et projet dans la partie 4...]*
