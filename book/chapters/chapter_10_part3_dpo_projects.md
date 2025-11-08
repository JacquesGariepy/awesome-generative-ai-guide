# Chapitre 14 (Suite): DPO et Projets Pratiques

## DPO (Direct Preference Optimization)

DPO est une alternative plus simple à RLHF qui optimise directement les préférences sans nécessiter de reward model séparé ni de PPO. C'est une approche plus récente (2023) qui gagne rapidement en popularité.

### 4.1 Théorie DPO

```python
"""
DPO (Direct Preference Optimization) - Théorie et Implementation
"""

from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass


class DPOTheory:
    """Explications théoriques de DPO"""

    @staticmethod
    def explain_motivation() -> str:
        """Pourquoi DPO?"""

        explanation = """
        === Motivation pour DPO ===

        Problèmes avec RLHF/PPO:
        1. Complexité: Nécessite reward model + PPO training
        2. Instabilité: PPO peut être instable et difficile à tuner
        3. Compute: Très coûteux (3 modèles + RL loop)
        4. Reward hacking: Le policy peut exploiter le reward model

        DPO simplifie le processus:
        - ❌ Pas de reward model séparé
        - ❌ Pas de PPO (RL algorithm)
        - ✅ Optimisation directe sur les préférences
        - ✅ Plus stable et simple
        - ✅ Résultats comparables à RLHF

        Pipeline:
        RLHF: SFT → Reward Model → PPO Training
        DPO:  SFT → DPO Training (direct!)
        """

        return explanation

    @staticmethod
    def explain_objective() -> str:
        """Objectif DPO"""

        explanation = """
        === Objectif DPO ===

        Idée clé: Reparamétrisation du reward model

        Au lieu d'apprendre un reward model séparé R(x,y), DPO dérive le reward
        directement de la policy:

        r(x, y) = β * log(π_θ(y|x) / π_ref(y|x))

        où:
        - π_θ = policy qu'on entraîne
        - π_ref = reference policy (frozen, typiquement le SFT model)
        - β = temperature parameter (typiquement 0.1)

        Objectif DPO:
        L_DPO(θ) = -E[(x, y_w, y_l) ~ D] [
            log σ(β * log(π_θ(y_w|x)/π_ref(y_w|x)) - β * log(π_θ(y_l|x)/π_ref(y_l|x)))
        ]

        où:
        - y_w = réponse "gagnante" (préférée)
        - y_l = réponse "perdante" (non-préférée)
        - σ = sigmoid function
        - D = dataset de comparaisons de paires

        Interprétation:
        On veut maximiser la probabilité que:
        - π_θ(y_w|x) > π_θ(y_l|x) * (π_ref(y_w|x) / π_ref(y_l|x))

        Autrement dit: Augmenter la proba de y_w et diminuer celle de y_l,
        tout en restant proche de π_ref.

        Avantages:
        1. Une seule loss (pas de reward model à entraîner séparément)
        2. Stable (pas de RL loop)
        3. Efficient (2 modèles au lieu de 3)
        4. Résultats compétitifs avec RLHF
        """

        return explanation

    @staticmethod
    def compare_rlhf_dpo() -> str:
        """Compare RLHF et DPO"""

        comparison = """
        === RLHF vs DPO ===

        ┌─────────────────┬──────────────────────┬─────────────────────┐
        │                 │        RLHF          │         DPO         │
        ├─────────────────┼──────────────────────┼─────────────────────┤
        │ Modèles         │ 3 (policy, reward,   │ 2 (policy, ref)     │
        │                 │    reference)        │                     │
        ├─────────────────┼──────────────────────┼─────────────────────┤
        │ Étapes          │ 1. Train reward model│ 1. DPO training     │
        │                 │ 2. PPO training      │    (c'est tout!)    │
        ├─────────────────┼──────────────────────┼─────────────────────┤
        │ Complexité      │ ★★★★★ (très élevée)  │ ★★☆☆☆ (moyenne)     │
        ├─────────────────┼──────────────────────┼─────────────────────┤
        │ Stabilité       │ ★★☆☆☆ (instable)     │ ★★★★☆ (stable)      │
        ├─────────────────┼──────────────────────┼─────────────────────┤
        │ Compute         │ ★★★★★ (très élevé)   │ ★★★☆☆ (modéré)      │
        ├─────────────────┼──────────────────────┼─────────────────────┤
        │ Hyperparams     │ Beaucoup (~20)       │ Peu (~5)            │
        ├─────────────────┼──────────────────────┼─────────────────────┤
        │ Performances    │ ★★★★★                │ ★★★★☆               │
        ├─────────────────┼──────────────────────┼─────────────────────┤
        │ Reward hacking  │ Possible             │ Moins probable      │
        ├─────────────────┼──────────────────────┼─────────────────────┤
        │ Maturité        │ Prouvé (ChatGPT)     │ Récent (2023)       │
        └─────────────────┴──────────────────────┴─────────────────────┘

        Recommandation:
        - Commencer par DPO (plus simple, bon résultats)
        - Passer à RLHF si besoin de performances maximales ou de flexibilité
        """

        return comparison


@dataclass
class DPOConfig:
    """Configuration pour DPO training"""

    # Model
    model_name: str = "meta-llama/Llama-2-7b-hf"
    ref_model_name: str = "meta-llama/Llama-2-7b-hf"

    # DPO specific
    beta: float = 0.1
    """Temperature parameter. Contrôle l'importance du KL penalty"""

    # Training
    learning_rate: float = 5e-7
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_epochs: int = 3
    max_length: int = 512
    max_prompt_length: int = 256

    # Optimization
    warmup_steps: int = 100
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0

    # Logging
    logging_steps: int = 10
    eval_steps: int = 500
    save_steps: int = 1000


class DPOLoss(nn.Module):
    """DPO Loss Function"""

    def __init__(self, beta: float = 0.1):
        super().__init__()
        self.beta = beta

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute DPO loss

        Args:
            policy_chosen_logps: log P(y_w|x) sous π_θ
            policy_rejected_logps: log P(y_l|x) sous π_θ
            reference_chosen_logps: log P(y_w|x) sous π_ref
            reference_rejected_logps: log P(y_l|x) sous π_ref

        Returns:
            loss: scalaire
            stats: dictionnaire de statistiques
        """

        # Compute ratios: log(π/π_ref)
        policy_chosen_ratio = policy_chosen_logps - reference_chosen_logps
        policy_rejected_ratio = policy_rejected_logps - reference_rejected_logps

        # DPO loss:
        # -log σ(β * [log(π_θ(y_w)/π_ref(y_w)) - log(π_θ(y_l)/π_ref(y_l))])
        logits = self.beta * (policy_chosen_ratio - policy_rejected_ratio)
        loss = -F.logsigmoid(logits).mean()

        # Statistiques
        with torch.no_grad():
            # Reward accuracy: combien de fois r_chosen > r_rejected
            rewards_chosen = self.beta * policy_chosen_ratio
            rewards_rejected = self.beta * policy_rejected_ratio
            reward_accuracy = (rewards_chosen > rewards_rejected).float().mean()

            # Reward margin
            reward_margin = (rewards_chosen - rewards_rejected).mean()

        stats = {
            'loss': loss.item(),
            'reward_accuracy': reward_accuracy.item(),
            'reward_margin': reward_margin.item(),
            'chosen_reward': rewards_chosen.mean().item(),
            'rejected_reward': rewards_rejected.mean().item()
        }

        return loss, stats


class DPOTrainer:
    """
    Trainer pour DPO

    Simplifie l'entraînement DPO avec toute la logique nécessaire
    """

    def __init__(
        self,
        model,  # Policy model à entraîner
        ref_model,  # Reference model (frozen)
        tokenizer,
        train_dataset,
        eval_dataset,
        config: DPOConfig,
        device: str = "cuda"
    ):
        self.model = model.to(device)
        self.ref_model = ref_model.to(device)
        self.tokenizer = tokenizer
        self.config = config
        self.device = device

        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False

        # Loss
        self.dpo_loss = DPOLoss(beta=config.beta)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # Datasets
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True
        )

        self.eval_loader = DataLoader(
            eval_dataset,
            batch_size=config.batch_size,
            shuffle=False
        ) if eval_dataset else None

        # Stats
        self.global_step = 0
        self.training_stats = []

    def concatenated_forward(
        self,
        model,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass pour chosen et rejected en un seul batch

        Returns:
            chosen_logps: log probabilities pour chosen responses
            rejected_logps: log probabilities pour rejected responses
        """

        # Concatenate chosen et rejected
        input_ids = torch.cat([
            batch['chosen_input_ids'],
            batch['rejected_input_ids']
        ], dim=0)

        attention_mask = torch.cat([
            batch['chosen_attention_mask'],
            batch['rejected_attention_mask']
        ], dim=0)

        labels = torch.cat([
            batch['chosen_labels'],
            batch['rejected_labels']
        ], dim=0)

        # Forward
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        logits = outputs.logits

        # Compute log probabilities
        # Shift pour aligner logits et labels
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        # Log softmax
        log_probs = F.log_softmax(shift_logits, dim=-1)

        # Gather log probs pour les tokens effectifs
        per_token_logps = torch.gather(
            log_probs,
            dim=2,
            index=shift_labels.unsqueeze(2)
        ).squeeze(2)

        # Masquer les padding tokens
        loss_mask = (shift_labels != -100).float()
        per_token_logps = per_token_logps * loss_mask

        # Sum over sequence length
        sequence_logps = per_token_logps.sum(dim=-1)

        # Split chosen et rejected
        batch_size = batch['chosen_input_ids'].shape[0]
        chosen_logps = sequence_logps[:batch_size]
        rejected_logps = sequence_logps[batch_size:]

        return chosen_logps, rejected_logps

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Un step d'entraînement"""

        self.model.train()

        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}

        # Forward avec policy model
        policy_chosen_logps, policy_rejected_logps = self.concatenated_forward(
            self.model,
            batch
        )

        # Forward avec reference model (no grad)
        with torch.no_grad():
            ref_chosen_logps, ref_rejected_logps = self.concatenated_forward(
                self.ref_model,
                batch
            )

        # Compute DPO loss
        loss, stats = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps
        )

        # Backward
        loss = loss / self.config.gradient_accumulation_steps
        loss.backward()

        # Update weights (si gradient accumulation complété)
        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config.max_grad_norm
            )

            self.optimizer.step()
            self.optimizer.zero_grad()

        self.global_step += 1

        return stats

    def evaluate(self) -> Dict[str, float]:
        """Évalue sur le eval dataset"""

        if self.eval_loader is None:
            return {}

        self.model.eval()

        all_stats = []

        with torch.no_grad():
            for batch in self.eval_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Forward
                policy_chosen_logps, policy_rejected_logps = self.concatenated_forward(
                    self.model,
                    batch
                )

                ref_chosen_logps, ref_rejected_logps = self.concatenated_forward(
                    self.ref_model,
                    batch
                )

                # Loss
                loss, stats = self.dpo_loss(
                    policy_chosen_logps,
                    policy_rejected_logps,
                    ref_chosen_logps,
                    ref_rejected_logps
                )

                all_stats.append(stats)

        # Average stats
        avg_stats = {
            f"eval_{key}": np.mean([s[key] for s in all_stats])
            for key in all_stats[0].keys()
        }

        return avg_stats

    def train(self) -> List[Dict[str, float]]:
        """Training loop complet"""

        print("=== Training DPO Model ===\n")
        print(f"Epochs: {self.config.num_epochs}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"Beta: {self.config.beta}\n")

        for epoch in range(self.config.num_epochs):
            print(f"Epoch {epoch + 1}/{self.config.num_epochs}")

            epoch_stats = []

            for step, batch in enumerate(self.train_loader):
                # Train step
                stats = self.train_step(batch)
                epoch_stats.append(stats)

                # Log
                if step % self.config.logging_steps == 0:
                    avg_stats = {
                        key: np.mean([s[key] for s in epoch_stats[-self.config.logging_steps:]])
                        for key in stats.keys()
                    }

                    print(f"  Step {step}: Loss={avg_stats['loss']:.4f}, "
                          f"Acc={avg_stats['reward_accuracy']:.4f}")

                # Evaluate
                if step % self.config.eval_steps == 0 and self.eval_loader:
                    eval_stats = self.evaluate()
                    print(f"  Eval: {eval_stats}")

            # Epoch summary
            avg_epoch_stats = {
                key: np.mean([s[key] for s in epoch_stats])
                for key in epoch_stats[0].keys()
            }

            print(f"Epoch {epoch+1} summary:")
            for key, value in avg_epoch_stats.items():
                print(f"  {key}: {value:.4f}")

            self.training_stats.append(avg_epoch_stats)

            print()

        return self.training_stats


# Exemple d'utilisation
if __name__ == "__main__":
    import numpy as np

    print("=== DPO Theory ===\n")

    theory = DPOTheory()
    print(theory.explain_motivation())
    print("\n" + "="*60 + "\n")
    print(theory.explain_objective())
    print("\n" + "="*60 + "\n")
    print(theory.compare_rlhf_dpo())

    print("\n" + "="*60 + "\n")
    print("=== DPO Configuration ===\n")

    config = DPOConfig(
        model_name="meta-llama/Llama-2-7b-hf",
        beta=0.1,
        learning_rate=5e-7,
        batch_size=4,
        num_epochs=3
    )

    print(f"Model: {config.model_name}")
    print(f"Beta: {config.beta}")
    print(f"Learning Rate: {config.learning_rate}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Epochs: {config.num_epochs}")

    # En production:
    # 1. Load models
    # model = AutoModelForCausalLM.from_pretrained(config.model_name)
    # ref_model = AutoModelForCausalLM.from_pretrained(config.ref_model_name)
    # tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # 2. Create datasets
    # train_dataset = DPODataset(train_preferences, tokenizer)
    # eval_dataset = DPODataset(eval_preferences, tokenizer)

    # 3. Train
    # trainer = DPOTrainer(
    #     model=model,
    #     ref_model=ref_model,
    #     tokenizer=tokenizer,
    #     train_dataset=train_dataset,
    #     eval_dataset=eval_dataset,
    #     config=config
    # )
    # stats = trainer.train()
```

---

## Projets Pratiques

### Projet 1: Pipeline RLHF Complet avec TRL

```python
"""
Projet Pratique: Pipeline RLHF complet avec TRL (Transformer Reinforcement Learning)

TRL est la librairie officielle de HuggingFace pour RLHF
"""

from typing import List, Dict, Optional
import torch
from dataclasses import dataclass


@dataclass
class RLHFProjectConfig:
    """Configuration du projet RLHF"""

    # Models
    base_model: str = "meta-llama/Llama-2-7b-chat-hf"
    sft_model_path: str = "./models/sft-llama-2-7b"
    reward_model_path: str = "./models/reward-llama-2-7b"
    final_model_path: str = "./models/rlhf-llama-2-7b"

    # Datasets
    sft_dataset: str = "OpenAssistant/oasst1"
    reward_dataset: str = "Anthropic/hh-rlhf"
    ppo_dataset: str = "Dahoas/rm-static"

    # Training
    sft_epochs: int = 3
    reward_epochs: int = 1
    ppo_steps: int = 10000

    # PPO params
    learning_rate: float = 1.41e-5
    batch_size: int = 16
    mini_batch_size: int = 4
    ppo_epochs: int = 4
    init_kl_coef: float = 0.2
    cliprange: float = 0.2


class RLHFPipeline:
    """
    Pipeline RLHF complet

    Étapes:
    1. Supervised Fine-Tuning (SFT)
    2. Reward Model Training
    3. PPO Training
    4. Evaluation
    """

    def __init__(self, config: RLHFProjectConfig):
        self.config = config

    def step1_supervised_finetuning(self):
        """
        Étape 1: SFT

        Fine-tune un base model sur des exemples de haute qualité
        """

        print("="*60)
        print("ÉTAPE 1: Supervised Fine-Tuning")
        print("="*60 + "\n")

        code_example = '''
# 1. Load base model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    load_in_8bit=True,  # Quantization pour économiser mémoire
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token = tokenizer.eos_token

# 2. Prepare dataset
from datasets import load_dataset

dataset = load_dataset("OpenAssistant/oasst1")

def format_prompt(example):
    """Format: <prompt>\\n<response>"""
    return {
        "text": f"{example['prompt']}\\n{example['response']}"
    }

train_dataset = dataset['train'].map(format_prompt)

# 3. Training arguments
training_args = TrainingArguments(
    output_dir="./sft-llama-2-7b",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    logging_steps=10,
    save_steps=500,
    fp16=True,
)

# 4. Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer
)

trainer.train()
trainer.save_model("./models/sft-llama-2-7b")
        '''

        print("Code pour SFT:")
        print(code_example)

        print("\n✅ Résultat: SFT Model entraîné")
        print(f"Sauvegardé dans: {self.config.sft_model_path}\n")

    def step2_reward_model_training(self):
        """
        Étape 2: Reward Model Training

        Entraîne un reward model sur des comparaisons de paires
        """

        print("="*60)
        print("ÉTAPE 2: Reward Model Training")
        print("="*60 + "\n")

        code_example = '''
# 1. Load SFT model comme base
from transformers import AutoModelForSequenceClassification

reward_model = AutoModelForSequenceClassification.from_pretrained(
    "./models/sft-llama-2-7b",
    num_labels=1  # Single scalar reward
)

# 2. Load preference dataset
from datasets import load_dataset

dataset = load_dataset("Anthropic/hh-rlhf")

# Format: chaque exemple a 'chosen' et 'rejected'
# {
#   "prompt": "...",
#   "chosen": "...",
#   "rejected": "..."
# }

# 3. Training avec TRL RewardTrainer
from trl import RewardTrainer, RewardConfig

config = RewardConfig(
    output_dir="./models/reward-llama-2-7b",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    learning_rate=1e-5,
    max_length=512
)

trainer = RewardTrainer(
    model=reward_model,
    args=config,
    train_dataset=dataset['train'],
    tokenizer=tokenizer
)

trainer.train()
trainer.save_model("./models/reward-llama-2-7b")
        '''

        print("Code pour Reward Model:")
        print(code_example)

        print("\n✅ Résultat: Reward Model entraîné")
        print(f"Sauvegardé dans: {self.config.reward_model_path}\n")

    def step3_ppo_training(self):
        """
        Étape 3: PPO Training

        Optimise le SFT model avec PPO en utilisant le reward model
        '''

        print("="*60)
        print("ÉTAPE 3: PPO Training")
        print("="*60 + "\n")

        code_example = '''
# 1. Load models
from transformers import AutoModelForCausalLM
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

# Policy model (avec value head)
policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    "./models/sft-llama-2-7b"
)

# Reference model (frozen)
ref_model = AutoModelForCausalLM.from_pretrained(
    "./models/sft-llama-2-7b"
)

# Reward model
reward_model = AutoModelForSequenceClassification.from_pretrained(
    "./models/reward-llama-2-7b"
)

# 2. PPO Configuration
ppo_config = PPOConfig(
    model_name="llama-2-7b-rlhf",
    learning_rate=1.41e-5,
    batch_size=16,
    mini_batch_size=4,
    ppo_epochs=4,
    init_kl_coef=0.2,
    target_kl=6.0,
    cliprange=0.2,
)

# 3. Create PPO Trainer
ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=policy_model,
    ref_model=ref_model,
    tokenizer=tokenizer,
)

# 4. Training loop
from datasets import load_dataset

dataset = load_dataset("Dahoas/rm-static")

for epoch in range(10):
    for batch in dataset['train'].iter(batch_size=16):
        # 1. Generate responses
        prompts = batch['prompt']
        query_tensors = [tokenizer.encode(p, return_tensors="pt") for p in prompts]

        response_tensors = ppo_trainer.generate(
            query_tensors,
            max_new_tokens=128,
            **generation_kwargs
        )

        # 2. Compute rewards
        texts = [tokenizer.decode(r) for r in response_tensors]
        rewards = [reward_model(t).logits[0].item() for t in texts]

        # 3. PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

        # 4. Log
        print(f"Epoch {epoch}, Reward: {np.mean(rewards):.2f}")

# 5. Save final model
ppo_trainer.save_model("./models/rlhf-llama-2-7b")
        '''

        print("Code pour PPO Training:")
        print(code_example)

        print("\n✅ Résultat: RLHF Model complet")
        print(f"Sauvegardé dans: {self.config.final_model_path}\n")

    def step4_evaluation(self):
        """
        Étape 4: Évaluation

        Compare Base / SFT / RLHF models
        """

        print("="*60)
        print("ÉTAPE 4: Évaluation")
        print("="*60 + "\n")

        code_example = '''
# 1. Load all models
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
sft_model = AutoModelForCausalLM.from_pretrained("./models/sft-llama-2-7b")
rlhf_model = AutoModelForCausalLM.from_pretrained("./models/rlhf-llama-2-7b")

# 2. Test prompts
test_prompts = [
    "Explain quantum computing in simple terms.",
    "What is the capital of France?",
    "Write a Python function to reverse a string."
]

# 3. Generate responses
def generate_response(model, prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=128)
    return tokenizer.decode(outputs[0])

# 4. Compare
for prompt in test_prompts:
    print(f"\\nPrompt: {prompt}")
    print(f"Base:  {generate_response(base_model, prompt)}")
    print(f"SFT:   {generate_response(sft_model, prompt)}")
    print(f"RLHF:  {generate_response(rlhf_model, prompt)}")

# 5. Metrics
from evaluate import load

# Reward scores
reward_model = AutoModelForSequenceClassification.from_pretrained(
    "./models/reward-llama-2-7b"
)

for model_name, model in [("Base", base_model), ("SFT", sft_model), ("RLHF", rlhf_model)]:
    rewards = []
    for prompt in test_prompts:
        response = generate_response(model, prompt)
        reward = reward_model(f"{prompt}\\n{response}").logits[0].item()
        rewards.append(reward)

    print(f"{model_name} Average Reward: {np.mean(rewards):.2f}")
        '''

        print("Code pour Évaluation:")
        print(code_example)

        print("\n✅ Résultats attendus:")
        print("  Base Model:  Reward ~ 0.0-2.0")
        print("  SFT Model:   Reward ~ 3.0-5.0")
        print("  RLHF Model:  Reward ~ 6.0-8.0\n")

    def run_complete_pipeline(self):
        """Exécute le pipeline complet"""

        print("\n" + "="*60)
        print("PIPELINE RLHF COMPLET")
        print("="*60 + "\n")

        self.step1_supervised_finetuning()
        self.step2_reward_model_training()
        self.step3_ppo_training()
        self.step4_evaluation()

        print("="*60)
        print("✅ PIPELINE TERMINÉ")
        print("="*60)
        print(f"\nModèle final: {self.config.final_model_path}")
        print("\nProchaines étapes:")
        print("  1. Tester le modèle sur des prompts variés")
        print("  2. Évaluer sur des benchmarks (MMLU, TruthfulQA, etc.)")
        print("  3. Red teaming pour tester la sécurité")
        print("  4. Déployer en production avec guardrails\n")


# Exemple d'utilisation
if __name__ == "__main__":
    config = RLHFProjectConfig(
        base_model="meta-llama/Llama-2-7b-hf",
        sft_epochs=3,
        reward_epochs=1,
        ppo_steps=10000
    )

    pipeline = RLHFPipeline(config)
    pipeline.run_complete_pipeline()
```

*[Conclusion et ressources dans le prochain message...]*
