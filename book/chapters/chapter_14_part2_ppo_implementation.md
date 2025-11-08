# Chapitre 14 (Suite): PPO Implementation Complète

## PPO (Proximal Policy Optimization)

PPO est l'algorithme de reinforcement learning utilisé dans RLHF. Il permet d'optimiser la politique (le modèle) de manière stable et efficace.

### 3.1 Théorie PPO

```python
"""
PPO Theory et Formulation Mathématique
"""

from dataclasses import dataclass
from typing import Dict
import numpy as np


class PPOTheory:
    """Explications théoriques de PPO"""

    @staticmethod
    def explain_objective() -> str:
        """
        Explique l'objectif de PPO

        PPO maximise un objectif "clipped" qui empêche les mises à jour trop grandes
        """

        explanation = """
        === Objectif PPO (Proximal Policy Optimization) ===

        Objectif Standard (Policy Gradient):
        L^PG(θ) = E[r_t * ∇log π_θ(a_t|s_t)]

        Problème: Les mises à jour peuvent être trop grandes et déstabiliser l'entraînement

        Objectif PPO (avec clipping):
        L^CLIP(θ) = E[min(r_t(θ) * A_t, clip(r_t(θ), 1-ε, 1+ε) * A_t)]

        où:
        - r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)  (ratio de probabilités)
        - A_t = advantage à  l'instant t
        - ε = cliprange (typiquement 0.2)
        - clip(x, min, max) = max(min, min(x, max))

        Interprétation:
        - Si A_t > 0 (bonne action): on limite l'augmentation de probabilité à (1+ε)
        - Si A_t < 0 (mauvaise action): on limite la diminution à (1-ε)

        Cela empêche les changements trop drastiques et stabilise l'entraînement.

        Loss Totale:
        L(θ) = L^CLIP(θ) - c_1 * L^VF(θ) + c_2 * H(π_θ)

        où:
        - L^VF = value function loss (MSE entre V(s) et returns)
        - H = entropy bonus (encourage exploration)
        - c_1, c_2 = coefficients (typiquement c_1=0.5, c_2=0.01)
        """

        return explanation

    @staticmethod
    def explain_advantages() -> str:
        """Explique le calcul des advantages avec GAE"""

        explanation = """
        === Generalized Advantage Estimation (GAE) ===

        L'advantage A_t mesure "à quel point une action est meilleure que la moyenne"

        Formule:
        A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...

        où:
        - δ_t = r_t + γV(s_{t+1}) - V(s_t)  (TD error)
        - γ = discount factor (typiquement 0.99)
        - λ = GAE parameter (typiquement 0.95)

        Trade-off γλ:
        - λ=0: A_t = δ_t (faible variance, haut bias)
        - λ=1: A_t = sum des rewards futurs - V(s_t) (haute variance, faible bias)

        GAE balance variance et bias pour un apprentissage stable.
        """

        return explanation

    @staticmethod
    def visualize_clipping() -> str:
        """Visualise l'effet du clipping"""

        viz = """
        === Visualisation du Clipping PPO ===

        Sans clipping:
        ────────────────────────────────────
        │                      /
        │                    /
        │                  /
        │                /
        │              /
        │            /
        │          /
        │        /
        │      /
        │    /
        │  /
        │/
        ────────────────────────────────────
         ratio (π_new / π_old)

        Avec clipping (ε=0.2):
        ────────────────────────────────────
        │           ┌─────────────
        │          /
        │         /
        │        /
        │       /
        │      /
        │     /
        │    /
        │───────────┘
        │
        │
        │
        ────────────────────────────────────
         0.8    1.0    1.2
              ratio

        Le clipping empêche les mises à jour trop agressives
        et maintient le ratio proche de 1.0
        """

        return viz


@dataclass
class PPOHyperparameters:
    """Hyperparamètres PPO avec explications"""

    # Learning
    learning_rate: float = 1.41e-5
    """Learning rate pour l'optimizer"""

    # PPO specific
    cliprange: float = 0.2
    """Clipping range pour PPO (ε). Typiquement 0.1-0.3"""

    cliprange_value: float = 0.2
    """Clipping range pour value function"""

    ppo_epochs: int = 4
    """Nombre d'epochs PPO sur chaque batch de rollout"""

    # Value function
    vf_coef: float = 0.5
    """Coefficient pour value function loss"""

    # Entropy
    ent_coef: float = 0.01
    """Coefficient pour entropy bonus (encourage exploration)"""

    # GAE
    gamma: float = 1.0
    """Discount factor. 1.0 pour language tasks (pas de vraie temporalité)"""

    lam: float = 0.95
    """GAE lambda. Balance variance/bias des advantages"""

    # KL divergence
    init_kl_coef: float = 0.2
    """Coefficient initial pour KL penalty"""

    target_kl: float = 6.0
    """Target KL divergence. Si dépassé, augmente kl_coef"""

    # Batch sizes
    batch_size: int = 256
    """Nombre de samples par batch de rollout"""

    mini_batch_size: int = 64
    """Taille des mini-batches pour PPO updates"""

    def explain(self) -> str:
        """Explique les hyperparamètres"""

        explanation = ["=== Hyperparamètres PPO ===\n"]

        explanation.append(f"Learning Rate: {self.learning_rate}")
        explanation.append("  → Vitesse d'apprentissage. Petit pour stabilité\n")

        explanation.append(f"Cliprange (ε): {self.cliprange}")
        explanation.append("  → Limite les changements de policy. 0.2 = ±20% max\n")

        explanation.append(f"PPO Epochs: {self.ppo_epochs}")
        explanation.append("  → Nombre de passes sur chaque batch\n")

        explanation.append(f"Value Function Coef: {self.vf_coef}")
        explanation.append("  → Poids de la value loss dans la loss totale\n")

        explanation.append(f"Entropy Coef: {self.ent_coef}")
        explanation.append("  → Bonus d'entropie pour encourager l'exploration\n")

        explanation.append(f"Gamma (γ): {self.gamma}")
        explanation.append("  → Discount factor. 1.0 car pas de vraie temporalité en LLM\n")

        explanation.append(f"Lambda (λ): {self.lam}")
        explanation.append("  → GAE parameter. 0.95 = bon équilibre variance/bias\n")

        explanation.append(f"KL Coefficient: {self.init_kl_coef}")
        explanation.append("  → Pénalité pour s'éloigner de la reference policy\n")

        explanation.append(f"Target KL: {self.target_kl}")
        explanation.append("  → Si KL > target, on augmente kl_coef\n")

        explanation.append(f"Batch Size: {self.batch_size}")
        explanation.append(f"Mini Batch Size: {self.mini_batch_size}")
        explanation.append(f"  → {self.batch_size // self.mini_batch_size} mini-batches par update")

        return "\n".join(explanation)


# Exemple d'utilisation
if __name__ == "__main__":
    theory = PPOTheory()

    print(theory.explain_objective())
    print("\n" + "="*60 + "\n")
    print(theory.explain_advantages())
    print("\n" + "="*60 + "\n")
    print(theory.visualize_clipping())

    print("\n" + "="*60 + "\n")

    hyperparams = PPOHyperparameters()
    print(hyperparams.explain())
```

### 3.2 Implementation Complète de PPO

```python
"""
Implementation complète de PPO pour RLHF
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class PPOBatch:
    """Batch de données pour PPO training"""

    # Inputs
    query_tensors: torch.Tensor  # Prompts
    response_tensors: torch.Tensor  # Réponses générées

    # Logprobs
    old_logprobs: torch.Tensor  # Log probs de la policy lors de la génération
    old_values: torch.Tensor  # Values prédites par la value function
    old_rewards: torch.Tensor  # Rewards du reward model

    # Computed
    advantages: torch.Tensor  # GAE advantages
    returns: torch.Tensor  # Targets pour value function


class GAEComputer:
    """Calcule Generalized Advantage Estimation"""

    def __init__(self, gamma: float = 1.0, lam: float = 0.95):
        self.gamma = gamma
        self.lam = lam

    def compute(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calcule GAE advantages et returns

        Args:
            rewards: [batch_size, seq_len] - rewards à chaque timestep
            values: [batch_size, seq_len] - value estimates
            dones: [batch_size, seq_len] - 1 si épisode terminé

        Returns:
            advantages: [batch_size, seq_len]
            returns: [batch_size, seq_len]
        """

        batch_size, seq_len = rewards.shape

        if dones is None:
            dones = torch.zeros_like(rewards)

        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)

        # Calcul backward (de la fin vers le début)
        gae = 0
        next_value = 0

        for t in reversed(range(seq_len)):
            # TD error: δ_t = r_t + γV(s_{t+1}) - V(s_t)
            if t == seq_len - 1:
                next_value = 0
            else:
                next_value = values[:, t + 1]

            delta = rewards[:, t] + self.gamma * next_value * (1 - dones[:, t]) - values[:, t]

            # GAE: A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
            gae = delta + self.gamma * self.lam * (1 - dones[:, t]) * gae
            advantages[:, t] = gae

            # Returns: R_t = A_t + V(s_t)
            returns[:, t] = advantages[:, t] + values[:, t]

        return advantages, returns


class PPOLoss:
    """Calcule les différentes composantes de la PPO loss"""

    def __init__(
        self,
        cliprange: float = 0.2,
        cliprange_value: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01
    ):
        self.cliprange = cliprange
        self.cliprange_value = cliprange_value
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef

    def compute_policy_loss(
        self,
        old_logprobs: torch.Tensor,
        new_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calcule la PPO clipped policy loss

        Args:
            old_logprobs: [batch_size, seq_len] - log probs de l'ancien policy
            new_logprobs: [batch_size, seq_len] - log probs du nouveau policy
            advantages: [batch_size, seq_len] - advantages GAE
            mask: [batch_size, seq_len] - masque pour tokens valides

        Returns:
            loss: scalaire
            stats: dictionnaire de statistiques
        """

        if mask is None:
            mask = torch.ones_like(old_logprobs)

        # Ratio: r_t = π_new / π_old = exp(log π_new - log π_old)
        logratio = new_logprobs - old_logprobs
        ratio = torch.exp(logratio)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO clipped loss
        # L = min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)
        policy_loss_1 = -advantages * ratio
        policy_loss_2 = -advantages * torch.clamp(
            ratio,
            1.0 - self.cliprange,
            1.0 + self.cliprange
        )
        policy_loss = torch.max(policy_loss_1, policy_loss_2)

        # Masked mean
        policy_loss = (policy_loss * mask).sum() / mask.sum()

        # Statistiques
        with torch.no_grad():
            clipfrac = ((ratio - 1.0).abs() > self.cliprange).float()
            clipfrac = (clipfrac * mask).sum() / mask.sum()

            approx_kl = ((ratio - 1.0) - logratio)
            approx_kl = (approx_kl * mask).sum() / mask.sum()

        stats = {
            'policy_loss': policy_loss.item(),
            'clipfrac': clipfrac.item(),
            'approx_kl': approx_kl.item(),
            'ratio_mean': (ratio * mask).sum().item() / mask.sum().item()
        }

        return policy_loss, stats

    def compute_value_loss(
        self,
        old_values: torch.Tensor,
        new_values: torch.Tensor,
        returns: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calcule la value function loss (avec clipping optionnel)

        Args:
            old_values: [batch_size, seq_len]
            new_values: [batch_size, seq_len]
            returns: [batch_size, seq_len]
            mask: [batch_size, seq_len]

        Returns:
            loss: scalaire
            stats: dictionnaire
        """

        if mask is None:
            mask = torch.ones_like(old_values)

        # Value loss avec clipping (optionnel)
        if self.cliprange_value is not None:
            values_clipped = old_values + torch.clamp(
                new_values - old_values,
                -self.cliprange_value,
                self.cliprange_value
            )

            vf_loss_1 = (new_values - returns) ** 2
            vf_loss_2 = (values_clipped - returns) ** 2
            vf_loss = torch.max(vf_loss_1, vf_loss_2)
        else:
            vf_loss = (new_values - returns) ** 2

        # Masked mean
        vf_loss = (vf_loss * mask).sum() / mask.sum()

        stats = {
            'value_loss': vf_loss.item(),
            'value_mean': (new_values * mask).sum().item() / mask.sum().item()
        }

        return vf_loss, stats

    def compute_entropy(
        self,
        logits: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calcule l'entropie pour encourager l'exploration

        Args:
            logits: [batch_size, seq_len, vocab_size]
            mask: [batch_size, seq_len]

        Returns:
            entropy: scalaire (négatif pour la loss)
        """

        if mask is None:
            mask = torch.ones(logits.shape[:2], device=logits.device)

        # Entropy: H = -sum(p * log(p))
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)

        # Masked mean
        entropy = (entropy * mask).sum() / mask.sum()

        return -entropy  # Négatif car on veut maximiser l'entropie

    def compute_total_loss(
        self,
        old_logprobs: torch.Tensor,
        new_logprobs: torch.Tensor,
        old_values: torch.Tensor,
        new_values: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        logits: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calcule la loss totale PPO

        Loss = L_policy + c1 * L_value + c2 * L_entropy

        Returns:
            total_loss: scalaire
            stats: toutes les statistiques
        """

        # Policy loss
        policy_loss, policy_stats = self.compute_policy_loss(
            old_logprobs, new_logprobs, advantages, mask
        )

        # Value loss
        value_loss, value_stats = self.compute_value_loss(
            old_values, new_values, returns, mask
        )

        # Entropy loss
        entropy_loss = self.compute_entropy(logits, mask)

        # Total loss
        total_loss = (
            policy_loss +
            self.vf_coef * value_loss +
            self.ent_coef * entropy_loss
        )

        # Combine stats
        stats = {
            **policy_stats,
            **value_stats,
            'entropy_loss': entropy_loss.item(),
            'total_loss': total_loss.item()
        }

        return total_loss, stats


class PPOTrainer:
    """
    PPO Trainer pour RLHF

    Gère l'entraînement complet avec PPO
    """

    def __init__(
        self,
        policy_model,  # Le modèle à entraîner
        ref_model,  # Reference model (frozen)
        reward_model,  # Reward model
        tokenizer,
        optimizer,
        config: PPOHyperparameters,
        device: str = "cuda"
    ):
        self.policy_model = policy_model.to(device)
        self.ref_model = ref_model.to(device)
        self.reward_model = reward_model.to(device)
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.config = config
        self.device = device

        # Freeze reference et reward models
        for param in self.ref_model.parameters():
            param.requires_grad = False
        for param in self.reward_model.parameters():
            param.requires_grad = False

        # Composants
        self.gae_computer = GAEComputer(
            gamma=config.gamma,
            lam=config.lam
        )

        self.ppo_loss = PPOLoss(
            cliprange=config.cliprange,
            cliprange_value=config.cliprange_value,
            vf_coef=config.vf_coef,
            ent_coef=config.ent_coef
        )

        # KL coefficient (adaptative)
        self.kl_coef = config.init_kl_coef

        # Stats
        self.training_stats = []

    def generate_responses(
        self,
        prompts: List[str],
        max_new_tokens: int = 128
    ) -> Dict[str, torch.Tensor]:
        """
        Génère des réponses pour un batch de prompts

        Returns:
            Dict avec query_tensors, response_tensors, logprobs, etc.
        """

        # Tokenize prompts
        query_tensors = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).input_ids.to(self.device)

        # Generate responses
        with torch.no_grad():
            output_tensors = self.policy_model.generate(
                query_tensors,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id
            )

        # Extract responses (sans le prompt)
        response_tensors = output_tensors[:, query_tensors.shape[1]:]

        # Compute logprobs et values pour les réponses
        logprobs, values = self._compute_logprobs_and_values(
            query_tensors,
            response_tensors
        )

        return {
            'query_tensors': query_tensors,
            'response_tensors': response_tensors,
            'logprobs': logprobs,
            'values': values
        }

    def _compute_logprobs_and_values(
        self,
        query_tensors: torch.Tensor,
        response_tensors: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calcule log probabilities et values pour les réponses

        Returns:
            logprobs: [batch_size, response_len]
            values: [batch_size, response_len]
        """

        # Concatenate query + response
        full_tensors = torch.cat([query_tensors, response_tensors], dim=1)

        with torch.no_grad():
            # Forward pass
            outputs = self.policy_model(
                full_tensors,
                output_hidden_states=True,
                return_dict=True
            )

            logits = outputs.logits
            hidden_states = outputs.hidden_states[-1]

            # Get logprobs pour les tokens de réponse
            # Shift logits et tokens
            shift_logits = logits[:, query_tensors.shape[1]-1:-1, :]
            shift_tokens = response_tensors

            # Log probabilities
            log_probs = F.log_softmax(shift_logits, dim=-1)
            logprobs = torch.gather(
                log_probs,
                2,
                shift_tokens.unsqueeze(-1)
            ).squeeze(-1)

            # Values (en pratique, utiliser une value head séparée)
            # Ici simplifié
            values = torch.zeros_like(logprobs)

        return logprobs, values

    def compute_rewards(
        self,
        query_tensors: torch.Tensor,
        response_tensors: torch.Tensor,
        ref_logprobs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calcule les rewards totaux

        reward = reward_model_score - β * KL(π || π_ref)

        Args:
            query_tensors: [batch_size, query_len]
            response_tensors: [batch_size, response_len]
            ref_logprobs: [batch_size, response_len] - logprobs de π_ref

        Returns:
            rewards: [batch_size, response_len]
        """

        # 1. Reward model score
        full_tensors = torch.cat([query_tensors, response_tensors], dim=1)

        with torch.no_grad():
            rm_rewards = self.reward_model(full_tensors)  # [batch_size]

        # Distribute reward sur le dernier token (ou tous)
        # Ici on met tout le reward sur le dernier token
        batch_size, seq_len = response_tensors.shape
        rewards = torch.zeros(batch_size, seq_len, device=self.device)
        rewards[:, -1] = rm_rewards

        # 2. KL penalty (si ref_logprobs fourni)
        if ref_logprobs is not None:
            # Compute current logprobs
            curr_logprobs, _ = self._compute_logprobs_and_values(
                query_tensors,
                response_tensors
            )

            # KL divergence
            kl_div = curr_logprobs - ref_logprobs

            # Subtract KL penalty
            rewards = rewards - self.kl_coef * kl_div

        return rewards

    def ppo_step(self, batch: PPOBatch) -> Dict[str, float]:
        """
        Un step d'entraînement PPO

        Args:
            batch: PPOBatch avec tous les tensors nécessaires

        Returns:
            stats: dictionnaire de statistiques
        """

        # Forward pass avec le policy actuel
        full_tensors = torch.cat([
            batch.query_tensors,
            batch.response_tensors
        ], dim=1)

        outputs = self.policy_model(
            full_tensors,
            output_hidden_states=True,
            return_dict=True
        )

        # Compute new logprobs et values
        new_logprobs, new_values = self._compute_logprobs_and_values(
            batch.query_tensors,
            batch.response_tensors
        )

        # Create mask (1 pour tokens réels, 0 pour padding)
        mask = (batch.response_tensors != self.tokenizer.pad_token_id).float()

        # Compute PPO loss
        loss, stats = self.ppo_loss.compute_total_loss(
            old_logprobs=batch.old_logprobs,
            new_logprobs=new_logprobs,
            old_values=batch.old_values,
            new_values=new_values,
            advantages=batch.advantages,
            returns=batch.returns,
            logits=outputs.logits[:, -batch.response_tensors.shape[1]:, :],
            mask=mask
        )

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.policy_model.parameters(),
            max_norm=1.0
        )

        self.optimizer.step()

        return stats

    def train_step(self, prompts: List[str]) -> Dict[str, float]:
        """
        Un step complet d'entraînement RLHF

        1. Generate responses
        2. Compute rewards
        3. Compute advantages
        4. PPO updates (multiple epochs)

        Returns:
            Statistiques d'entraînement
        """

        # 1. Generate responses
        generation = self.generate_responses(prompts)

        # 2. Compute reference logprobs (pour KL)
        with torch.no_grad():
            ref_logprobs, _ = self._compute_logprobs_and_values(
                generation['query_tensors'],
                generation['response_tensors']
            )

        # 3. Compute rewards
        rewards = self.compute_rewards(
            generation['query_tensors'],
            generation['response_tensors'],
            ref_logprobs
        )

        # 4. Compute advantages avec GAE
        advantages, returns = self.gae_computer.compute(
            rewards,
            generation['values']
        )

        # 5. Create PPO batch
        ppo_batch = PPOBatch(
            query_tensors=generation['query_tensors'],
            response_tensors=generation['response_tensors'],
            old_logprobs=generation['logprobs'],
            old_values=generation['values'],
            old_rewards=rewards,
            advantages=advantages,
            returns=returns
        )

        # 6. PPO updates (multiple epochs)
        all_stats = []
        for ppo_epoch in range(self.config.ppo_epochs):
            stats = self.ppo_step(ppo_batch)
            all_stats.append(stats)

            # Early stopping si KL trop élevé
            if stats['approx_kl'] > self.config.target_kl:
                print(f"Early stopping at PPO epoch {ppo_epoch} (KL={stats['approx_kl']:.3f})")
                break

        # Average stats
        avg_stats = {
            key: np.mean([s[key] for s in all_stats])
            for key in all_stats[0].keys()
        }

        # Adaptive KL coefficient
        if avg_stats['approx_kl'] > self.config.target_kl * 1.5:
            self.kl_coef *= 1.5
        elif avg_stats['approx_kl'] < self.config.target_kl / 1.5:
            self.kl_coef /= 1.5

        avg_stats['kl_coef'] = self.kl_coef

        return avg_stats


# Exemple d'utilisation (conceptuel)
if __name__ == "__main__":
    print("=== PPO Training Example ===\n")

    # Configuration
    config = PPOHyperparameters(
        learning_rate=1.41e-5,
        batch_size=16,
        mini_batch_size=4,
        ppo_epochs=4,
        cliprange=0.2,
        init_kl_coef=0.2
    )

    print(config.explain())

    # En production:
    # 1. Load models
    # policy_model = AutoModelForCausalLM.from_pretrained(...)
    # ref_model = AutoModelForCausalLM.from_pretrained(...)
    # reward_model = RewardModel(...)

    # 2. Create trainer
    # trainer = PPOTrainer(
    #     policy_model=policy_model,
    #     ref_model=ref_model,
    #     reward_model=reward_model,
    #     tokenizer=tokenizer,
    #     optimizer=optimizer,
    #     config=config
    # )

    # 3. Training loop
    # for step in range(num_training_steps):
    #     prompts = sample_prompts(batch_size)
    #     stats = trainer.train_step(prompts)
    #     log_stats(stats)
```

*[Suite avec DPO et projets pratiques dans le prochain message...]*
