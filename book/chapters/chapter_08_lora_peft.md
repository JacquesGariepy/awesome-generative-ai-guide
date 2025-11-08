# Chapitre 8: LoRA et Parameter-Efficient Fine-Tuning (PEFT)

## Introduction

Dans le chapitre précédent, nous avons vu comment fine-tuner des LLMs. Cependant, le **full fine-tuning** pose des problèmes majeurs:

- 💰 **Coût**: Nécessite stocker une copie complète du modèle pour chaque tâche
- 🔥 **VRAM**: Impossible sur GPUs grand public pour les modèles >7B
- ⏰ **Temps**: Training très long
- 📊 **Overfitting**: Risque élevé sur petits datasets

Le **Parameter-Efficient Fine-Tuning (PEFT)** résout ces problèmes en ne modifiant qu'une fraction des paramètres.

### Motivation: Le Problème du Full Fine-Tuning

```python
"""
Comparaison des besoins mémoire: Full Fine-tuning vs PEFT
"""

from dataclasses import dataclass
from typing import Dict
from enum import Enum


class ModelSize(Enum):
    """Tailles de modèles standards"""
    SMALL_1B = 1_000_000_000
    MEDIUM_7B = 7_000_000_000
    LARGE_13B = 13_000_000_000
    XLARGE_70B = 70_000_000_000


@dataclass
class MemoryRequirements:
    """Calcul des besoins mémoire pour training"""

    @staticmethod
    def calculate_full_finetuning_memory(
        num_parameters: int,
        precision: str = "fp32"
    ) -> Dict[str, float]:
        """
        Calcule la mémoire nécessaire pour full fine-tuning

        Composantes de mémoire:
        1. Model weights: num_params * bytes_per_param
        2. Gradients: num_params * bytes_per_param
        3. Optimizer states (Adam): 2 * num_params * bytes_per_param (momentum + variance)
        4. Activations: ~2-3x model size (depends on batch size)

        Total ≈ 6x model size pour FP32, 3x pour FP16/BF16

        Args:
            num_parameters: Nombre de paramètres du modèle
            precision: "fp32", "fp16", ou "bf16"

        Returns:
            Dict avec breakdown de la mémoire
        """

        bytes_per_param = {
            "fp32": 4,
            "fp16": 2,
            "bf16": 2
        }[precision]

        # Mémoire pour les poids
        model_memory_gb = (num_parameters * bytes_per_param) / (1024**3)

        # Gradients (même taille que les poids)
        gradient_memory_gb = model_memory_gb

        # Optimizer states (Adam: 2x les poids pour momentum + variance)
        optimizer_memory_gb = 2 * model_memory_gb

        # Activations (estimation conservative: 2x model size)
        activation_memory_gb = 2 * model_memory_gb

        total_memory_gb = (
            model_memory_gb +
            gradient_memory_gb +
            optimizer_memory_gb +
            activation_memory_gb
        )

        return {
            "model_weights": round(model_memory_gb, 2),
            "gradients": round(gradient_memory_gb, 2),
            "optimizer_states": round(optimizer_memory_gb, 2),
            "activations": round(activation_memory_gb, 2),
            "total": round(total_memory_gb, 2),
            "precision": precision
        }

    @staticmethod
    def calculate_lora_memory(
        num_parameters: int,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        target_modules_ratio: float = 0.1,
        precision: str = "fp32"
    ) -> Dict[str, float]:
        """
        Calcule la mémoire nécessaire pour LoRA fine-tuning

        LoRA ajoute des matrices low-rank A et B:
        - A: (rank, hidden_dim)
        - B: (hidden_dim, rank)
        - Total params ajoutés ≈ 2 * rank * hidden_dim

        Mais on ne train que ces nouveaux paramètres!

        Args:
            num_parameters: Nombre de paramètres du modèle base
            lora_rank: Rank des matrices LoRA
            lora_alpha: Scaling factor
            target_modules_ratio: % de modules ciblés (query, value, etc.)
            precision: "fp32", "fp16", ou "bf16"

        Returns:
            Dict avec breakdown de la mémoire
        """

        bytes_per_param = {
            "fp32": 4,
            "fp16": 2,
            "bf16": 2
        }[precision]

        # Mémoire pour le modèle base (frozen, pas de gradient)
        base_model_memory_gb = (num_parameters * bytes_per_param) / (1024**3)

        # Estimer le nombre de paramètres LoRA ajoutés
        # Pour chaque module ciblé: 2 * rank * hidden_dim
        # Approximation: hidden_dim ≈ sqrt(num_parameters)
        estimated_hidden_dim = int((num_parameters / 32) ** 0.5)  # Conservative estimate

        lora_params_per_module = 2 * lora_rank * estimated_hidden_dim
        num_target_modules = int(32 * target_modules_ratio)  # ~32 attention layers typically

        total_lora_params = lora_params_per_module * num_target_modules

        # Mémoire pour les poids LoRA
        lora_weights_gb = (total_lora_params * bytes_per_param) / (1024**3)

        # Gradients (seulement pour LoRA params)
        lora_gradients_gb = lora_weights_gb

        # Optimizer states (seulement pour LoRA params)
        lora_optimizer_gb = 2 * lora_weights_gb

        # Activations (légèrement plus que base model car forward pass complet)
        activation_memory_gb = 1.5 * base_model_memory_gb

        total_memory_gb = (
            base_model_memory_gb +
            lora_weights_gb +
            lora_gradients_gb +
            lora_optimizer_gb +
            activation_memory_gb
        )

        return {
            "base_model": round(base_model_memory_gb, 2),
            "lora_weights": round(lora_weights_gb, 2),
            "lora_gradients": round(lora_gradients_gb, 2),
            "lora_optimizer": round(lora_optimizer_gb, 2),
            "activations": round(activation_memory_gb, 2),
            "total": round(total_memory_gb, 2),
            "trainable_params": total_lora_params,
            "trainable_percent": round(100 * total_lora_params / num_parameters, 3),
            "precision": precision
        }


class PEFTComparison:
    """Comparaison des différentes approches PEFT"""

    @staticmethod
    def compare_approaches() -> Dict[str, Dict]:
        """Retourne comparaison détaillée de toutes les approches"""

        return {
            "Full Fine-Tuning": {
                "trainable_params": "100%",
                "memory_multiplier": "6x model size (FP32), 3x (FP16)",
                "training_speed": "Baseline (1x)",
                "performance": "⭐⭐⭐⭐⭐ (meilleur)",
                "use_cases": "Datasets très larges, ressources illimitées",
                "pros": [
                    "Performance maximale",
                    "Flexibilité totale",
                    "Adapte tout le modèle"
                ],
                "cons": [
                    "Très coûteux en mémoire",
                    "Lent",
                    "Risque d'overfitting",
                    "Nécessite GPU haut de gamme"
                ]
            },

            "LoRA (Low-Rank Adaptation)": {
                "trainable_params": "0.1-1%",
                "memory_multiplier": "1.5-2x model size",
                "training_speed": "2-3x plus rapide",
                "performance": "⭐⭐⭐⭐ (excellent, ~98% du full FT)",
                "use_cases": "Usage général, datasets moyens à larges",
                "pros": [
                    "Très efficace en mémoire",
                    "Performance proche du full FT",
                    "Training rapide",
                    "Modulaire (swap LoRA adapters)"
                ],
                "cons": [
                    "Légèrement moins performant que full FT",
                    "Nécessite tuning de rank/alpha"
                ]
            },

            "QLoRA (Quantized LoRA)": {
                "trainable_params": "0.1-1%",
                "memory_multiplier": "0.4-0.6x model size (!!)",
                "training_speed": "Légèrement plus lent que LoRA",
                "performance": "⭐⭐⭐⭐ (excellent)",
                "use_cases": "GPUs limités (8-16GB), grands modèles",
                "pros": [
                    "Fine-tune 70B sur 1x RTX 3090!",
                    "4-bit quantization du modèle base",
                    "Performance quasi-identique à LoRA",
                    "Révolutionnaire pour accessibilité"
                ],
                "cons": [
                    "Légèrement plus lent que LoRA FP16",
                    "Nécessite bitsandbytes library"
                ]
            },

            "Adapter Layers": {
                "trainable_params": "1-3%",
                "memory_multiplier": "2-2.5x model size",
                "training_speed": "Similar to LoRA",
                "performance": "⭐⭐⭐ (bon)",
                "use_cases": "Multi-task learning, modularity",
                "pros": [
                    "Très modulaire",
                    "Facile à comprendre",
                    "Bon pour multi-task"
                ],
                "cons": [
                    "Plus de paramètres que LoRA",
                    "Latence d'inférence légèrement plus haute"
                ]
            },

            "Prefix Tuning": {
                "trainable_params": "0.01-0.1%",
                "memory_multiplier": "1.2-1.5x model size",
                "training_speed": "Très rapide",
                "performance": "⭐⭐⭐ (bon pour certaines tâches)",
                "use_cases": "NLG tasks, prompt optimization",
                "pros": [
                    "Très peu de paramètres",
                    "Rapide",
                    "Pas de modification de l'architecture"
                ],
                "cons": [
                    "Performance variable selon tâche",
                    "Moins bon pour tasks complexes"
                ]
            },

            "P-Tuning v2": {
                "trainable_params": "0.1-0.5%",
                "memory_multiplier": "1.5-2x model size",
                "training_speed": "Rapide",
                "performance": "⭐⭐⭐⭐ (très bon)",
                "use_cases": "NLU tasks, question answering",
                "pros": [
                    "Bon compromis params/performance",
                    "Marche bien sur NLU",
                    "Continuous prompts"
                ],
                "cons": [
                    "Moins bon que LoRA en général",
                    "Spécialisé pour certaines tâches"
                ]
            }
        }


# Exemple d'utilisation
if __name__ == "__main__":
    print("="*60)
    print("COMPARAISON MÉMOIRE: Full Fine-tuning vs LoRA vs QLoRA")
    print("="*60)
    print()

    # Tester différentes tailles de modèles
    models = {
        "Llama 2 7B": ModelSize.MEDIUM_7B.value,
        "Llama 2 13B": ModelSize.LARGE_13B.value,
        "Llama 2 70B": ModelSize.XLARGE_70B.value,
    }

    for model_name, num_params in models.items():
        print(f"\n### {model_name} ({num_params/1e9:.0f}B parameters)")
        print()

        # Full Fine-tuning FP16
        full_ft = MemoryRequirements.calculate_full_finetuning_memory(
            num_params, precision="fp16"
        )
        print(f"Full Fine-tuning (FP16):")
        print(f"  Total VRAM: {full_ft['total']:.1f} GB")
        print(f"  Breakdown: Model={full_ft['model_weights']:.1f}GB, "
              f"Grads={full_ft['gradients']:.1f}GB, "
              f"Optimizer={full_ft['optimizer_states']:.1f}GB")

        # LoRA FP16
        lora = MemoryRequirements.calculate_lora_memory(
            num_params, lora_rank=8, precision="fp16"
        )
        print(f"\nLoRA (rank=8, FP16):")
        print(f"  Total VRAM: {lora['total']:.1f} GB")
        print(f"  Trainable params: {lora['trainable_params']:,} ({lora['trainable_percent']:.2f}%)")
        print(f"  Memory savings: {((full_ft['total'] - lora['total']) / full_ft['total'] * 100):.1f}%")

        # QLoRA (4-bit base + FP16 LoRA)
        # Base model en 4-bit = 0.5 bytes per param
        base_4bit_gb = (num_params * 0.5) / (1024**3)
        qlora_total = base_4bit_gb + lora['lora_weights'] + lora['lora_gradients'] + lora['lora_optimizer'] + lora['activations'] * 0.5
        print(f"\nQLoRA (4-bit base + rank=8):")
        print(f"  Total VRAM: {qlora_total:.1f} GB")
        print(f"  Base model (4-bit): {base_4bit_gb:.1f} GB")
        print(f"  Memory savings vs Full FT: {((full_ft['total'] - qlora_total) / full_ft['total'] * 100):.1f}%")

        print()
        print("-" * 60)

    print("\n" + "="*60)
    print("COMPARAISON DES APPROCHES PEFT")
    print("="*60)
    print()

    comparison = PEFTComparison.compare_approaches()

    for approach, details in list(comparison.items())[:3]:  # Afficher les 3 principales
        print(f"### {approach}")
        print(f"Trainable params: {details['trainable_params']}")
        print(f"Mémoire: {details['memory_multiplier']}")
        print(f"Performance: {details['performance']}")
        print(f"Use case: {details['use_cases']}")
        print()
```

## 1. LoRA: Low-Rank Adaptation

### 1.1 Théorie Mathématique

```python
"""
Théorie mathématique de LoRA

Paper: "LoRA: Low-Rank Adaptation of Large Language Models"
Authors: Hu et al. (Microsoft), 2021
"""

import torch
import torch.nn as nn
from typing import Optional
import math


class LoRATheory:
    """
    Explication mathématique de LoRA

    Idée clé: Les changements dans les poids pendant le fine-tuning
    ont un "rank intrinsèque" faible.

    Au lieu de mettre à jour la matrice complète W ∈ ℝ^(d×k):
        W' = W + ΔW

    LoRA approxime ΔW par un produit de deux matrices low-rank:
        ΔW = B·A

    Où:
    - A ∈ ℝ^(r×k), r << d (matrice down-projection)
    - B ∈ ℝ^(d×r), r << k (matrice up-projection)
    - r: rank, typiquement 8-64 vs d,k ~ 4096+

    Nombre de paramètres:
    - Full update: d × k (e.g., 4096 × 4096 = 16.7M)
    - LoRA: (d + k) × r (e.g., (4096 + 4096) × 8 = 65K)
    - Réduction: ~256x moins de paramètres!
    """

    @staticmethod
    def explain_rank_concept():
        """Explique le concept de rank faible"""

        explanation = """
        === Concept de Low-Rank ===

        1. RANK d'une matrice:
           - Le rank est le nombre de dimensions "indépendantes"
           - Matrice 1000×1000 peut avoir rank 10 si seulement 10 "directions" indépendantes

        2. INTUITION pour LoRA:
           - Pendant fine-tuning, on adapte le modèle pour une tâche spécifique
           - Cette adaptation ne nécessite pas de modifier TOUTES les directions
           - Seulement un petit nombre de "directions clés" suffisent
           - C'est le "low rank" de l'adaptation

        3. EXEMPLE concret:
           - Imaginez adapter un modèle de traduction EN→FR vers EN→ES
           - Beaucoup de connaissances sont réutilisables (grammaire, vocabulaire)
           - Seulement quelques aspects spécifiques à l'espagnol changent
           - Ces changements peuvent être capturés avec low-rank matrices

        4. MATHÉMATIQUEMENT:
           Full update:     W' = W + ΔW                 (d × k parameters)
           LoRA:            W' = W + B·A                 ((d+k) × r parameters)

           Forward pass:    h' = W'x = Wx + BAx = Wx + B(Ax)

           - Calcul en 2 étapes: d'abord Ax (projection), puis B(Ax) (élévation)
           - Permet de garder W frozen et train seulement A et B
        """

        return explanation

    @staticmethod
    def calculate_parameter_reduction(
        d: int,
        k: int,
        r: int
    ) -> dict:
        """
        Calcule la réduction de paramètres avec LoRA

        Args:
            d: Dimension de sortie
            k: Dimension d'entrée
            r: Rank LoRA

        Returns:
            Dict avec statistiques
        """

        full_params = d * k
        lora_params = (d + k) * r

        reduction_factor = full_params / lora_params
        reduction_percent = (1 - lora_params / full_params) * 100

        return {
            "full_params": full_params,
            "lora_params": lora_params,
            "reduction_factor": round(reduction_factor, 1),
            "reduction_percent": round(reduction_percent, 2),
            "memory_saved_mb": round((full_params - lora_params) * 4 / (1024**2), 2)  # FP32
        }


class LoRALayer(nn.Module):
    """
    Implémentation simple de LoRA Layer

    Peut être appliqué à n'importe quel nn.Linear layer
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.0
    ):
        """
        Args:
            in_features: Input dimension (k)
            out_features: Output dimension (d)
            rank: Rank r des matrices LoRA
            alpha: Scaling factor (typiquement 2*rank)
            dropout: Dropout sur l'input LoRA
        """
        super().__init__()

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank  # Scaling factor pour stabilité

        # Matrice A: (rank, in_features) - initialized avec Kaiming
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # Matrice B: (out_features, rank) - initialized à zéro!
        # Important: B=0 au début donc ΔW = BA = 0, ne change rien initially
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Dropout optionnel
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: compute BAx

        Args:
            x: Input tensor (..., in_features)

        Returns:
            LoRA output (..., out_features)
        """

        # x: (..., in_features)
        # lora_A: (rank, in_features)
        # lora_B: (out_features, rank)

        # Step 1: Project down to rank: Ax
        # x @ lora_A.T → (..., rank)
        down_proj = x @ self.lora_A.t()

        # Apply dropout
        down_proj = self.dropout(down_proj)

        # Step 2: Project up to out_features: B(Ax)
        # down_proj @ lora_B.T → (..., out_features)
        up_proj = down_proj @ self.lora_B.t()

        # Apply scaling
        return up_proj * self.scaling


class LinearWithLoRA(nn.Module):
    """
    Linear layer avec LoRA intégré

    Forward: y = Wx + BAx
           = (W + BA)x
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.0
    ):
        """
        Args:
            base_layer: Le nn.Linear original (frozen)
            rank: Rank LoRA
            alpha: Scaling factor
            dropout: Dropout
        """
        super().__init__()

        # Base layer (frozen)
        self.base_layer = base_layer
        for param in self.base_layer.parameters():
            param.requires_grad = False

        # LoRA adapter
        self.lora = LoRALayer(
            in_features=base_layer.in_features,
            out_features=base_layer.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward: y = Wx + BAx

        Args:
            x: Input

        Returns:
            Output
        """

        # Base forward (frozen)
        base_output = self.base_layer(x)

        # LoRA forward
        lora_output = self.lora(x)

        # Combine
        return base_output + lora_output

    def merge_weights(self) -> nn.Linear:
        """
        Merge LoRA weights into base layer: W' = W + BA

        Returns:
            New nn.Linear with merged weights
        """

        # Compute ΔW = BA
        delta_W = (self.lora.lora_B @ self.lora.lora_A) * self.lora.scaling

        # Create new layer
        merged = nn.Linear(
            self.base_layer.in_features,
            self.base_layer.out_features,
            bias=self.base_layer.bias is not None
        )

        # Merge weights: W' = W + ΔW
        merged.weight.data = self.base_layer.weight.data + delta_W

        if self.base_layer.bias is not None:
            merged.bias.data = self.base_layer.bias.data

        return merged


# Exemple d'utilisation
if __name__ == "__main__":
    print("="*60)
    print("LoRA THEORY & IMPLEMENTATION")
    print("="*60)
    print()

    # Théorie
    print(LoRATheory.explain_rank_concept())
    print("\n" + "="*60 + "\n")

    # Calcul réduction paramètres
    print("### Parameter Reduction")
    print()

    configs = [
        {"name": "Small layer", "d": 1024, "k": 1024, "r": 8},
        {"name": "Medium layer", "d": 4096, "k": 4096, "r": 16},
        {"name": "Large layer", "d": 8192, "k": 8192, "r": 32},
    ]

    for config in configs:
        stats = LoRATheory.calculate_parameter_reduction(
            d=config['d'],
            k=config['k'],
            r=config['r']
        )

        print(f"{config['name']} ({config['d']}×{config['k']}, rank={config['r']}):")
        print(f"  Full params: {stats['full_params']:,}")
        print(f"  LoRA params: {stats['lora_params']:,}")
        print(f"  Reduction: {stats['reduction_factor']}x ({stats['reduction_percent']:.1f}%)")
        print(f"  Memory saved: {stats['memory_saved_mb']:.1f} MB")
        print()

    print("="*60 + "\n")

    # Test implémentation
    print("### LoRA Implementation Test")
    print()

    # Create base layer
    base_layer = nn.Linear(512, 512)
    print(f"Base layer params: {sum(p.numel() for p in base_layer.parameters()):,}")

    # Wrap avec LoRA
    lora_layer = LinearWithLoRA(base_layer, rank=8, alpha=16)

    # Count trainable params
    trainable = sum(p.numel() for p in lora_layer.parameters() if p.requires_grad)
    total = sum(p.numel() for p in lora_layer.parameters())

    print(f"LoRA layer total params: {total:,}")
    print(f"LoRA layer trainable params: {trainable:,} ({100*trainable/total:.2f}%)")

    # Test forward
    x = torch.randn(4, 512)
    output = lora_layer(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # Test merge
    merged = lora_layer.merge_weights()
    print(f"\nMerged layer params: {sum(p.numel() for p in merged.parameters()):,}")
```

*[Suite avec techniques PEFT avancées dans la partie 2...]*
