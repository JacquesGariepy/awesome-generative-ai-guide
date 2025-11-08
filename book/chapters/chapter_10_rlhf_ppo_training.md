# Chapitre 14: RLHF et PPO Training - Aligner les LLMs avec les Préférences Humaines

## Table des Matières
1. [Introduction](#introduction)
2. [Reinforcement Learning from Human Feedback (RLHF)](#reinforcement-learning-from-human-feedback)
3. [Reward Modeling](#reward-modeling)
4. [PPO (Proximal Policy Optimization)](#ppo-proximal-policy-optimization)
5. [DPO (Direct Preference Optimization)](#dpo-direct-preference-optimization)
6. [Constitutional AI](#constitutional-ai)
7. [Pipeline RLHF Complet](#pipeline-rlhf-complet)
8. [Projets Pratiques](#projets-pratiques)
9. [Conclusion](#conclusion)

---

## Introduction

Le **Reinforcement Learning from Human Feedback (RLHF)** est devenu la technique de référence pour aligner les Large Language Models avec les préférences humaines. Cette approche est au cœur du succès de modèles comme ChatGPT, Claude, et autres assistants conversationnels modernes.

### Pourquoi RLHF?

Les modèles pré-entraînés avec un objectif de prédiction du prochain token (next token prediction) sont excellents pour **compléter du texte**, mais pas nécessairement pour:

- ✅ Être **utiles** (helpful)
- ✅ Être **honnêtes** (honest)
- ✅ Être **inoffensifs** (harmless)

Le RLHF permet de transformer un modèle de **complétion de texte** en un **assistant conversationnel** aligné avec les valeurs humaines.

### Objectifs d'Apprentissage

À la fin de ce chapitre, vous serez capable de:

1. Comprendre les fondements théoriques du RLHF
2. Implémenter un reward model from scratch
3. Entraîner un modèle avec PPO
4. Utiliser DPO comme alternative plus simple
5. Évaluer la qualité de l'alignment
6. Déployer un pipeline RLHF complet en production
7. Comprendre les trade-offs et limitations

### Le Pipeline RLHF en 3 Étapes

```
┌─────────────────────────────────────────────────────────────────┐
│                    Pipeline RLHF Complet                        │
└─────────────────────────────────────────────────────────────────┘

Étape 1: Supervised Fine-Tuning (SFT)
┌──────────────┐
│ Base Model   │  ──► Fine-tune sur données de haute qualité
│ (GPT, Llama) │      (instructions + réponses exemplaires)
└──────────────┘
       │
       ▼
┌──────────────┐
│  SFT Model   │  ──► Génère de bonnes réponses mais pas optimales
└──────────────┘

Étape 2: Reward Model Training
┌──────────────┐
│  SFT Model   │  ──► Génère plusieurs réponses par prompt
└──────────────┘
       │
       ▼
┌──────────────┐
│   Humains    │  ──► Classent les réponses (paire par paire)
└──────────────┘
       │
       ▼
┌──────────────┐
│Reward Model  │  ──► Apprend à prédire les préférences humaines
└──────────────┘

Étape 3: RL Fine-Tuning avec PPO
┌──────────────┐     ┌──────────────┐
│  SFT Model   │ ◄─► │Reward Model  │
│  (Policy)    │     │  (Critic)    │
└──────────────┘     └──────────────┘
       │                    │
       └────────┬───────────┘
                ▼
         ┌──────────────┐
         │ PPO Training │  ──► Optimise la politique pour maximiser reward
         └──────────────┘
                │
                ▼
         ┌──────────────┐
         │Aligned Model │  ──► Modèle final aligné avec préférences
         └──────────────┘
```

---

## Reinforcement Learning from Human Feedback

### 1.1 Fondements Théoriques

Le RLHF formule l'alignment comme un problème d'apprentissage par renforcement:

**Objectif**: Maximiser la récompense attendue selon les préférences humaines

```
π* = argmax E_π[R(x, y)]
      π

où:
- π = politique (le modèle de langage)
- x = prompt/contexte
- y = réponse générée
- R = fonction de récompense (apprise des humains)
```

### 1.2 Comparaison avec d'Autres Approches

```python
"""
Comparaison des approches d'alignment
"""

from dataclasses import dataclass
from typing import List, Dict
from enum import Enum


class AlignmentApproach(Enum):
    """Différentes approches d'alignment"""
    SUPERVISED_FINETUNING = "sft"
    RLHF = "rlhf"
    DPO = "dpo"
    CONSTITUTIONAL_AI = "constitutional_ai"
    INSTRUCTION_TUNING = "instruction_tuning"


@dataclass
class AlignmentMethod:
    """Caractéristiques d'une méthode d'alignment"""
    name: str
    approach: AlignmentApproach

    # Avantages
    advantages: List[str]

    # Inconvénients
    disadvantages: List[str]

    # Complexité (1-5, 5 = très complexe)
    complexity: int

    # Coût (1-5, 5 = très cher)
    cost: int

    # Efficacité (1-5, 5 = très efficace)
    effectiveness: int

    # Données requises
    data_requirements: str


class AlignmentComparison:
    """Comparaison des méthodes d'alignment"""

    def __init__(self):
        self.methods = self._initialize_methods()

    def _initialize_methods(self) -> Dict[str, AlignmentMethod]:
        """Initialise les méthodes d'alignment"""

        return {
            "sft": AlignmentMethod(
                name="Supervised Fine-Tuning",
                approach=AlignmentApproach.SUPERVISED_FINETUNING,
                advantages=[
                    "Simple à implémenter",
                    "Stable et prévisible",
                    "Nécessite moins de compute",
                    "Bonne baseline"
                ],
                disadvantages=[
                    "Nécessite des données d'experts (coûteux)",
                    "Difficile de capturer toutes les nuances",
                    "Peut sur-apprendre sur les exemples",
                    "Pas d'optimisation directe des préférences"
                ],
                complexity=2,
                cost=3,
                effectiveness=3,
                data_requirements="Paires (prompt, réponse de haute qualité)"
            ),

            "rlhf": AlignmentMethod(
                name="RLHF (Reinforcement Learning from Human Feedback)",
                approach=AlignmentApproach.RLHF,
                advantages=[
                    "Optimise directement pour préférences humaines",
                    "Peut découvrir des comportements non-montrés",
                    "État de l'art (ChatGPT, Claude)",
                    "Flexible et puissant"
                ],
                disadvantages=[
                    "Complexe à implémenter",
                    "Instable (reward hacking)",
                    "Très coûteux en compute",
                    "Nécessite expertise en RL",
                    "Hyperparamètres sensibles"
                ],
                complexity=5,
                cost=5,
                effectiveness=5,
                data_requirements="Comparaisons de paires (rankings humains)"
            ),

            "dpo": AlignmentMethod(
                name="DPO (Direct Preference Optimization)",
                approach=AlignmentApproach.DPO,
                advantages=[
                    "Plus simple que RLHF (pas de reward model)",
                    "Plus stable",
                    "Moins coûteux en compute",
                    "Résultats comparables à RLHF"
                ],
                disadvantages=[
                    "Moins flexible que RLHF",
                    "Recherche récente (moins mature)",
                    "Nécessite toujours des comparaisons humaines",
                    "Peut être moins robuste sur certains tasks"
                ],
                complexity=3,
                cost=3,
                effectiveness=4,
                data_requirements="Comparaisons de paires (rankings humains)"
            ),

            "constitutional_ai": AlignmentMethod(
                name="Constitutional AI",
                approach=AlignmentApproach.CONSTITUTIONAL_AI,
                advantages=[
                    "Réduit besoin de feedback humain",
                    "Principles explicites et modifiables",
                    "Self-critique et self-improvement",
                    "Scalable"
                ],
                disadvantages=[
                    "Nécessite toujours un modèle de base fort",
                    "Constitution peut être subjective",
                    "Performances dépendent de la qualité des principles"
                ],
                complexity=4,
                cost=3,
                effectiveness=4,
                data_requirements="Principles constitutionnels + comparaisons AI-generated"
            )
        }

    def compare(self, methods: List[str]) -> str:
        """Compare plusieurs méthodes"""

        report = ["=== Comparaison des Méthodes d'Alignment ===\n"]

        for method_key in methods:
            if method_key not in self.methods:
                continue

            method = self.methods[method_key]

            report.append(f"\n## {method.name}")
            report.append(f"Approche: {method.approach.value}")
            report.append(f"\nComplexité: {'★' * method.complexity}{'☆' * (5-method.complexity)}")
            report.append(f"Coût: {'★' * method.cost}{'☆' * (5-method.cost)}")
            report.append(f"Efficacité: {'★' * method.effectiveness}{'☆' * (5-method.effectiveness)}")

            report.append(f"\nAvantages:")
            for adv in method.advantages:
                report.append(f"  ✅ {adv}")

            report.append(f"\nInconvénients:")
            for disadv in method.disadvantages:
                report.append(f"  ❌ {disadv}")

            report.append(f"\nDonnées requises: {method.data_requirements}")
            report.append("\n" + "-" * 60)

        return "\n".join(report)

    def recommend(self, budget: str, expertise: str) -> str:
        """Recommande une approche basée sur le contexte"""

        if budget == "low" and expertise == "beginner":
            return "sft"
        elif budget == "medium" and expertise == "intermediate":
            return "dpo"
        elif budget == "high" and expertise == "advanced":
            return "rlhf"
        else:
            return "dpo"  # Default safe choice


# Exemple d'utilisation
if __name__ == "__main__":
    comparison = AlignmentComparison()

    # Comparer toutes les méthodes
    print(comparison.compare(["sft", "rlhf", "dpo", "constitutional_ai"]))

    # Recommandation
    print("\n\n=== Recommandations ===")
    print(f"Budget faible + Débutant: {comparison.recommend('low', 'beginner')}")
    print(f"Budget moyen + Intermédiaire: {comparison.recommend('medium', 'intermediate')}")
    print(f"Budget élevé + Avancé: {comparison.recommend('high', 'advanced')}")
```

### 1.3 Composants Clés du RLHF

```python
"""
Architecture RLHF - Vue d'ensemble des composants
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class RLHFConfig:
    """Configuration pour RLHF training"""

    # Modèles
    policy_model_name: str = "meta-llama/Llama-2-7b-hf"
    reward_model_name: str = "meta-llama/Llama-2-7b-hf"
    ref_model_name: str = "meta-llama/Llama-2-7b-hf"  # Reference model (frozen)

    # PPO Hyperparameters
    learning_rate: float = 1.41e-5
    batch_size: int = 16
    mini_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    ppo_epochs: int = 4

    # PPO specific
    init_kl_coef: float = 0.2  # KL divergence penalty
    target_kl: float = 6.0
    cliprange: float = 0.2
    cliprange_value: float = 0.2
    vf_coef: float = 0.1  # Value function coefficient
    gamma: float = 1.0  # Discount factor
    lam: float = 0.95  # GAE lambda

    # Generation
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.95

    # Training
    num_training_steps: int = 10000
    save_freq: int = 1000
    eval_freq: int = 500

    # Logging
    log_with: str = "wandb"
    project_name: str = "rlhf-training"


class RLHFComponents:
    """
    Composants d'un système RLHF

    Cette classe illustre l'architecture et les interactions
    entre les différents composants du RLHF.
    """

    def __init__(self, config: RLHFConfig):
        self.config = config

        # Les 3 modèles principaux
        self.policy_model = None  # Le modèle qu'on entraîne (π_θ)
        self.reward_model = None  # Prédit le reward (R_φ)
        self.reference_model = None  # Version frozen du policy (π_ref)

        # Optimiseur PPO
        self.optimizer = None

        # Statistiques
        self.training_stats = {
            "rewards": [],
            "kl_divergence": [],
            "policy_loss": [],
            "value_loss": []
        }

    def initialize_models(self):
        """
        Initialise les 3 modèles nécessaires pour RLHF

        1. Policy Model (π_θ): Le modèle qu'on optimise
        2. Reward Model (R_φ): Estime la qualité des réponses
        3. Reference Model (π_ref): Version frozen pour KL penalty
        """

        print("Initialisation des modèles RLHF...")

        # En production, utiliser transformers.AutoModelForCausalLM
        # self.policy_model = AutoModelForCausalLM.from_pretrained(
        #     self.config.policy_model_name
        # )

        # Pour cet exemple, placeholders
        self.policy_model = "PolicyModel"
        self.reward_model = "RewardModel"
        self.reference_model = "ReferenceModel (frozen)"

        print(f"✅ Policy Model: {self.config.policy_model_name}")
        print(f"✅ Reward Model: {self.config.reward_model_name}")
        print(f"✅ Reference Model: {self.config.ref_model_name} (frozen)")

    def ppo_step_overview(self) -> Dict[str, str]:
        """
        Décrit les étapes d'un pas PPO

        Returns:
            Dict avec les étapes du processus
        """

        return {
            "step_1": "Génération de réponses avec π_θ (policy model)",
            "step_2": "Évaluation avec R_φ (reward model)",
            "step_3": "Calcul de KL divergence avec π_ref",
            "step_4": "Calcul de la reward totale: r = R_φ - β*KL",
            "step_5": "Calcul des advantages avec GAE",
            "step_6": "Update π_θ avec PPO loss",
            "step_7": "Repeat pour mini_batch_size batches"
        }

    def explain_kl_penalty(self) -> str:
        """
        Explique le rôle du KL penalty

        Le KL penalty empêche le modèle de trop s'éloigner de la référence,
        ce qui prévient le "reward hacking" et maintient la qualité du langage.
        """

        explanation = """
        === KL Divergence Penalty ===

        Formule: KL(π_θ || π_ref)

        Rôle:
        1. Prévenir reward hacking (exploitation des failles du reward model)
        2. Maintenir la fluidité du langage
        3. Éviter le mode collapse
        4. Garantir la stabilité de l'entraînement

        Reward Total:
        r(x,y) = R_φ(x,y) - β * KL(π_θ(·|x) || π_ref(·|x))

        où:
        - R_φ(x,y) = reward du reward model
        - β = coefficient KL (typiquement 0.1-0.5)
        - KL = divergence KL entre policy et reference

        Trade-off:
        - β trop petit → reward hacking
        - β trop grand → le modèle ne change pas assez
        """

        return explanation

    def training_loop_pseudocode(self) -> str:
        """Pseudo-code du training loop RLHF"""

        pseudocode = """
        === RLHF Training Loop (PPO) ===

        for epoch in range(num_epochs):
            # 1. Rollout: Générer des réponses
            prompts = sample_prompts(batch_size)
            responses = policy_model.generate(prompts)

            # 2. Évaluation
            rewards = reward_model(prompts, responses)
            ref_logprobs = reference_model.forward(prompts, responses)
            policy_logprobs = policy_model.forward(prompts, responses)

            # 3. Calcul KL divergence
            kl_div = policy_logprobs - ref_logprobs

            # 4. Reward total
            total_reward = rewards - kl_coef * kl_div

            # 5. Calcul advantages (GAE)
            advantages = compute_gae(total_reward)

            # 6. PPO update (multiple epochs sur le même batch)
            for ppo_epoch in range(ppo_epochs):
                for mini_batch in split(batch_size, mini_batch_size):
                    # Calcul PPO loss
                    ratio = exp(new_logprobs - old_logprobs)
                    clipped_ratio = clip(ratio, 1-cliprange, 1+cliprange)
                    policy_loss = -min(ratio * advantages,
                                      clipped_ratio * advantages)

                    # Value loss
                    value_loss = (values - returns)^2

                    # Total loss
                    loss = policy_loss + vf_coef * value_loss

                    # Backprop
                    loss.backward()
                    optimizer.step()

            # 7. Logging et évaluation
            if epoch % eval_freq == 0:
                evaluate_model()
        """

        return pseudocode

    def explain_reward_hacking(self) -> Dict[str, any]:
        """
        Explique le phénomène de reward hacking
        """

        return {
            "definition": "Le modèle trouve des failles dans le reward model pour obtenir des scores élevés sans réellement améliorer la qualité",

            "examples": [
                {
                    "scenario": "Longueur des réponses",
                    "hack": "Si le reward model favorise les réponses longues, le policy model génère des réponses très longues mais peu informatives",
                    "solution": "KL penalty + diverse reward metrics"
                },
                {
                    "scenario": "Mots clés",
                    "hack": "Répéter des mots/phrases qui augmentent artificiellement le reward",
                    "solution": "Reward model robuste + regularization"
                },
                {
                    "scenario": "Format",
                    "hack": "Exploiter un format spécifique que le reward model favorise",
                    "solution": "Diverse training data + ensemble de reward models"
                }
            ],

            "mitigations": [
                "KL divergence penalty (empêche de trop s'éloigner)",
                "Reward model ensemble (plusieurs reward models)",
                "Human-in-the-loop evaluation",
                "Adversarial testing",
                "Regular reward model updates"
            ]
        }

    def get_training_overview(self) -> str:
        """Génère un overview complet du training RLHF"""

        overview = ["=== RLHF Training Overview ===\n"]

        overview.append("## Composants")
        overview.append(f"Policy Model: {self.policy_model}")
        overview.append(f"Reward Model: {self.reward_model}")
        overview.append(f"Reference Model: {self.reference_model}\n")

        overview.append("## Étapes PPO")
        for step, description in self.ppo_step_overview().items():
            overview.append(f"{step}: {description}")

        overview.append(f"\n## Configuration")
        overview.append(f"Learning Rate: {self.config.learning_rate}")
        overview.append(f"Batch Size: {self.config.batch_size}")
        overview.append(f"KL Coefficient: {self.config.init_kl_coef}")
        overview.append(f"PPO Epochs: {self.config.ppo_epochs}")

        return "\n".join(overview)


# Exemple d'utilisation
if __name__ == "__main__":
    # Configuration
    config = RLHFConfig(
        policy_model_name="meta-llama/Llama-2-7b-hf",
        learning_rate=1.41e-5,
        batch_size=16
    )

    # Initialiser les composants
    rlhf = RLHFComponents(config)
    rlhf.initialize_models()

    # Overview
    print("\n" + rlhf.get_training_overview())

    # KL Penalty explanation
    print("\n" + rlhf.explain_kl_penalty())

    # Training loop
    print("\n" + rlhf.training_loop_pseudocode())

    # Reward hacking
    print("\n=== Reward Hacking ===")
    hacking_info = rlhf.explain_reward_hacking()
    print(f"Définition: {hacking_info['definition']}\n")
    print("Exemples:")
    for ex in hacking_info['examples']:
        print(f"  - {ex['scenario']}: {ex['hack']}")
        print(f"    Solution: {ex['solution']}")
```

---

## Reward Modeling

Le **reward model** est le cœur du RLHF. Il apprend à prédire les préférences humaines à partir de comparaisons de paires de réponses.

### 2.1 Entraînement du Reward Model

```python
"""
Reward Model Implementation
Apprend à prédire les préférences humaines
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class PreferenceData:
    """Données de préférence humaine"""
    prompt: str
    chosen_response: str
    rejected_response: str
    margin: float = 1.0  # Marge de préférence (optionnel)


class PreferenceDataset(Dataset):
    """Dataset de comparaisons de paires"""

    def __init__(
        self,
        preferences: List[PreferenceData],
        tokenizer,
        max_length: int = 512
    ):
        self.preferences = preferences
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.preferences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Retourne une paire de (prompt, réponse) à comparer

        Format:
        - chosen: prompt + chosen_response
        - rejected: prompt + rejected_response
        """

        pref = self.preferences[idx]

        # Tokenize chosen
        chosen_text = f"{pref.prompt}\n{pref.chosen_response}"
        chosen_encoding = self.tokenizer(
            chosen_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Tokenize rejected
        rejected_text = f"{pref.prompt}\n{pref.rejected_response}"
        rejected_encoding = self.tokenizer(
            rejected_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'chosen_input_ids': chosen_encoding['input_ids'].squeeze(0),
            'chosen_attention_mask': chosen_encoding['attention_mask'].squeeze(0),
            'rejected_input_ids': rejected_encoding['input_ids'].squeeze(0),
            'rejected_attention_mask': rejected_encoding['attention_mask'].squeeze(0),
            'margin': torch.tensor(pref.margin, dtype=torch.float)
        }


class RewardModel(nn.Module):
    """
    Reward Model basé sur un LLM

    Architecture:
    1. LLM base (ex: GPT, Llama)
    2. Value head (projette vers un scalaire)

    Le modèle prédit un score de reward pour (prompt, response)
    """

    def __init__(
        self,
        base_model,  # Pre-trained LLM
        hidden_size: int = 4096
    ):
        super().__init__()

        self.base_model = base_model

        # Value head: transforme hidden states en reward score
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1)  # Sortie scalaire
        )

        # Freeze base model (optionnel, souvent on fine-tune tout)
        # for param in self.base_model.parameters():
        #     param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]

        Returns:
            rewards: [batch_size] - score de reward pour chaque séquence
        """

        # Get hidden states from base model
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        # Get last hidden state
        hidden_states = outputs.hidden_states[-1]  # [batch, seq_len, hidden_size]

        # Use last token's hidden state (ou mean pooling)
        # Ici on utilise le dernier token non-padded
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = hidden_states.shape[0]

        last_hidden_states = hidden_states[
            torch.arange(batch_size, device=hidden_states.device),
            sequence_lengths
        ]  # [batch_size, hidden_size]

        # Project to reward score
        rewards = self.value_head(last_hidden_states).squeeze(-1)  # [batch_size]

        return rewards


class RewardModelTrainer:
    """Trainer pour le Reward Model"""

    def __init__(
        self,
        model: RewardModel,
        train_dataset: PreferenceDataset,
        val_dataset: Optional[PreferenceDataset] = None,
        learning_rate: float = 1e-5,
        batch_size: int = 4,
        num_epochs: int = 3,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model.to(device)
        self.device = device

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False
        ) if val_dataset else None

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate
        )

        self.num_epochs = num_epochs

        # Statistiques
        self.training_stats = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }

    def compute_loss(
        self,
        chosen_rewards: torch.Tensor,
        rejected_rewards: torch.Tensor,
        margin: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calcule la pairwise ranking loss

        Loss: -log(sigmoid(r_chosen - r_rejected))

        On veut: r_chosen > r_rejected

        Args:
            chosen_rewards: [batch_size]
            rejected_rewards: [batch_size]
            margin: [batch_size]

        Returns:
            loss: scalaire
            accuracy: scalaire (proportion où r_chosen > r_rejected)
        """

        # Pairwise ranking loss
        # Loss = -log(σ(r_chosen - r_rejected - margin))
        loss = -F.logsigmoid(chosen_rewards - rejected_rewards - margin).mean()

        # Accuracy: combien de fois r_chosen > r_rejected
        accuracy = (chosen_rewards > rejected_rewards).float().mean()

        return loss, accuracy

    def train_epoch(self) -> Dict[str, float]:
        """Entraîne pour une epoch"""

        self.model.train()

        epoch_loss = 0.0
        epoch_accuracy = 0.0
        num_batches = 0

        for batch in self.train_loader:
            # Move to device
            chosen_input_ids = batch['chosen_input_ids'].to(self.device)
            chosen_attention_mask = batch['chosen_attention_mask'].to(self.device)
            rejected_input_ids = batch['rejected_input_ids'].to(self.device)
            rejected_attention_mask = batch['rejected_attention_mask'].to(self.device)
            margin = batch['margin'].to(self.device)

            # Forward pass
            chosen_rewards = self.model(chosen_input_ids, chosen_attention_mask)
            rejected_rewards = self.model(rejected_input_ids, rejected_attention_mask)

            # Compute loss
            loss, accuracy = self.compute_loss(chosen_rewards, rejected_rewards, margin)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Accumulate stats
            epoch_loss += loss.item()
            epoch_accuracy += accuracy.item()
            num_batches += 1

        return {
            'loss': epoch_loss / num_batches,
            'accuracy': epoch_accuracy / num_batches
        }

    def evaluate(self) -> Dict[str, float]:
        """Évalue sur le validation set"""

        if self.val_loader is None:
            return {}

        self.model.eval()

        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                # Move to device
                chosen_input_ids = batch['chosen_input_ids'].to(self.device)
                chosen_attention_mask = batch['chosen_attention_mask'].to(self.device)
                rejected_input_ids = batch['rejected_input_ids'].to(self.device)
                rejected_attention_mask = batch['rejected_attention_mask'].to(self.device)
                margin = batch['margin'].to(self.device)

                # Forward pass
                chosen_rewards = self.model(chosen_input_ids, chosen_attention_mask)
                rejected_rewards = self.model(rejected_input_ids, rejected_attention_mask)

                # Compute loss
                loss, accuracy = self.compute_loss(chosen_rewards, rejected_rewards, margin)

                total_loss += loss.item()
                total_accuracy += accuracy.item()
                num_batches += 1

        return {
            'loss': total_loss / num_batches,
            'accuracy': total_accuracy / num_batches
        }

    def train(self) -> Dict[str, List[float]]:
        """Training loop complet"""

        print("=== Training Reward Model ===\n")

        for epoch in range(self.num_epochs):
            # Train
            train_metrics = self.train_epoch()
            self.training_stats['train_loss'].append(train_metrics['loss'])
            self.training_stats['train_accuracy'].append(train_metrics['accuracy'])

            # Evaluate
            val_metrics = self.evaluate()
            if val_metrics:
                self.training_stats['val_loss'].append(val_metrics['loss'])
                self.training_stats['val_accuracy'].append(val_metrics['accuracy'])

            # Log
            print(f"Epoch {epoch+1}/{self.num_epochs}")
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            print(f"  Train Acc:  {train_metrics['accuracy']:.4f}")

            if val_metrics:
                print(f"  Val Loss:   {val_metrics['loss']:.4f}")
                print(f"  Val Acc:    {val_metrics['accuracy']:.4f}")

            print()

        return self.training_stats


# Exemple d'utilisation (conceptuel)
if __name__ == "__main__":
    print("=== Reward Model Training Example ===\n")

    # Note: Ceci est un exemple conceptuel
    # En production, utiliser HuggingFace Transformers

    # Créer des données d'exemple
    preferences = [
        PreferenceData(
            prompt="What is the capital of France?",
            chosen_response="The capital of France is Paris.",
            rejected_response="France.",
            margin=1.0
        ),
        PreferenceData(
            prompt="Explain quantum computing",
            chosen_response="Quantum computing uses quantum-mechanical phenomena like superposition and entanglement to perform computation...",
            rejected_response="It's computers but quantum.",
            margin=2.0
        )
    ]

    print(f"Dataset size: {len(preferences)} preference pairs")
    print(f"\nExample preference:")
    print(f"  Prompt: {preferences[0].prompt}")
    print(f"  Chosen: {preferences[0].chosen_response}")
    print(f"  Rejected: {preferences[0].rejected_response}")
    print(f"  Margin: {preferences[0].margin}")

    # En production:
    # 1. Load pre-trained LLM
    # from transformers import AutoModel, AutoTokenizer
    # base_model = AutoModel.from_pretrained("meta-llama/Llama-2-7b-hf")
    # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

    # 2. Create reward model
    # reward_model = RewardModel(base_model, hidden_size=4096)

    # 3. Create dataset
    # dataset = PreferenceDataset(preferences, tokenizer)

    # 4. Train
    # trainer = RewardModelTrainer(reward_model, dataset)
    # stats = trainer.train()
```

*[Suite du chapitre dans le prochain message...]*
