# Chapitre 9 (Suite): Chat Templates et Multi-Turn

## 3. Chat Templates et Formatage

```python
"""
Chat Templates: Formatage standard pour conversations

Chaque modèle a son propre format de chat:
- Llama 2: [INST] ... [/INST]
- ChatML: <|im_start|> ... <|im_end|>
- Alpaca: ### Instruction: ... ### Response:
- Vicuna: USER: ... ASSISTANT:
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum


class ChatTemplate(Enum):
    """Templates de chat populaires"""
    LLAMA2 = "llama2"
    CHATML = "chatml"  # ChatML (OpenAI, Mistral)
    ALPACA = "alpaca"
    VICUNA = "vicuna"
    ZEPHYR = "zephyr"


@dataclass
class ChatMessage:
    """Un message dans une conversation"""
    role: str  # "system", "user", "assistant"
    content: str


class ChatFormatter:
    """
    Formatter universel pour conversations chat

    Gère les différents formats de modèles
    """

    @staticmethod
    def format_llama2(messages: List[ChatMessage]) -> str:
        """
        Format Llama 2 Chat

        Format:
        <s>[INST] <<SYS>>
        {system_message}
        <</SYS>>

        {user_message_1} [/INST] {assistant_message_1}</s>
        <s>[INST] {user_message_2} [/INST] {assistant_message_2}</s>
        """

        formatted = []
        system_message = None

        # Extract system message
        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
                break

        # Format conversation
        user_msgs = []
        assistant_msgs = []

        for msg in messages:
            if msg.role == "user":
                user_msgs.append(msg.content)
            elif msg.role == "assistant":
                assistant_msgs.append(msg.content)

        # First turn (avec system si existe)
        if user_msgs:
            if system_message:
                formatted.append(f"<s>[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n{user_msgs[0]} [/INST]")
            else:
                formatted.append(f"<s>[INST] {user_msgs[0]} [/INST]")

            if assistant_msgs:
                formatted.append(f" {assistant_msgs[0]}</s>")

        # Subsequent turns
        for i in range(1, len(user_msgs)):
            formatted.append(f"<s>[INST] {user_msgs[i]} [/INST]")
            if i < len(assistant_msgs):
                formatted.append(f" {assistant_msgs[i]}</s>")

        return "".join(formatted)

    @staticmethod
    def format_chatml(messages: List[ChatMessage]) -> str:
        """
        Format ChatML (OpenAI, Mistral)

        Format:
        <|im_start|>system
        {system_message}<|im_end|>
        <|im_start|>user
        {user_message}<|im_end|>
        <|im_start|>assistant
        {assistant_message}<|im_end|>
        """

        formatted = []

        for msg in messages:
            formatted.append(f"<|im_start|>{msg.role}\n{msg.content}<|im_end|>")

        return "\n".join(formatted)

    @staticmethod
    def format_alpaca(messages: List[ChatMessage]) -> str:
        """
        Format Alpaca

        Format:
        ### Instruction:
        {user_message}

        ### Response:
        {assistant_message}
        """

        formatted = []

        # Alpaca généralement single-turn, mais on peut concaténer
        for i, msg in enumerate(messages):
            if msg.role == "system":
                formatted.append(f"### System:\n{msg.content}\n")
            elif msg.role == "user":
                formatted.append(f"### Instruction:\n{msg.content}\n")
            elif msg.role == "assistant":
                formatted.append(f"### Response:\n{msg.content}\n")

        return "\n".join(formatted)

    @staticmethod
    def format_vicuna(messages: List[ChatMessage]) -> str:
        """
        Format Vicuna

        Format:
        A chat between a curious user and an artificial intelligence assistant.

        USER: {user_message}
        ASSISTANT: {assistant_message}
        """

        formatted = []

        # Add system prompt
        system = "A chat between a curious user and an artificial intelligence assistant."
        for msg in messages:
            if msg.role == "system":
                system = msg.content
                break

        formatted.append(system + "\n")

        # Format messages
        for msg in messages:
            if msg.role == "user":
                formatted.append(f"USER: {msg.content}")
            elif msg.role == "assistant":
                formatted.append(f"ASSISTANT: {msg.content}")

        return "\n".join(formatted)

    @staticmethod
    def format_zephyr(messages: List[ChatMessage]) -> str:
        """
        Format Zephyr (HuggingFace)

        Format:
        <|system|>
        {system_message}</s>
        <|user|>
        {user_message}</s>
        <|assistant|>
        {assistant_message}</s>
        """

        formatted = []

        for msg in messages:
            formatted.append(f"<|{msg.role}|>\n{msg.content}</s>")

        return "\n".join(formatted)

    @staticmethod
    def format_conversation(
        messages: List[ChatMessage],
        template: ChatTemplate = ChatTemplate.CHATML
    ) -> str:
        """
        Format conversation selon le template choisi

        Args:
            messages: Liste de messages
            template: Template à utiliser

        Returns:
            Conversation formatée
        """

        if template == ChatTemplate.LLAMA2:
            return ChatFormatter.format_llama2(messages)
        elif template == ChatTemplate.CHATML:
            return ChatFormatter.format_chatml(messages)
        elif template == ChatTemplate.ALPACA:
            return ChatFormatter.format_alpaca(messages)
        elif template == ChatTemplate.VICUNA:
            return ChatFormatter.format_vicuna(messages)
        elif template == ChatTemplate.ZEPHYR:
            return ChatFormatter.format_zephyr(messages)
        else:
            raise ValueError(f"Unknown template: {template}")


# Exemple
if __name__ == "__main__":
    print("="*60)
    print("CHAT TEMPLATES COMPARISON")
    print("="*60)
    print()

    # Conversation exemple
    messages = [
        ChatMessage(role="system", content="You are a helpful AI assistant."),
        ChatMessage(role="user", content="What is Python?"),
        ChatMessage(role="assistant", content="Python is a high-level programming language known for its simplicity."),
        ChatMessage(role="user", content="What are its main uses?"),
        ChatMessage(role="assistant", content="Python is used for web development, data science, AI, automation, and more.")
    ]

    # Test tous les formats
    templates = [
        ChatTemplate.LLAMA2,
        ChatTemplate.CHATML,
        ChatTemplate.ALPACA,
        ChatTemplate.VICUNA,
        ChatTemplate.ZEPHYR
    ]

    for template in templates:
        print(f"### {template.value.upper()} Format")
        print()
        formatted = ChatFormatter.format_conversation(messages, template)
        print(formatted)
        print()
        print("-"*60)
        print()
```

## 4. Multi-Turn Conversation Handling

```python
"""
Gestion des conversations multi-tour

Défis:
1. Contexte cumulatif (mémoire)
2. Cohérence entre tours
3. Truncation intelligente
4. Training sur conversations partielles
"""

from typing import List, Tuple
import torch


class MultiTurnConversationProcessor:
    """
    Processeur pour conversations multi-turn

    Gère:
    - Context window
    - Truncation
    - Training masking
    """

    def __init__(
        self,
        max_length: int = 2048,
        response_template: str = "### Response:",
        instruction_template: str = "### Instruction:"
    ):
        """
        Args:
            max_length: Longueur max de contexte
            response_template: Template pour réponses
            instruction_template: Template pour instructions
        """
        self.max_length = max_length
        self.response_template = response_template
        self.instruction_template = instruction_template

    def create_training_examples(
        self,
        conversation: List[ChatMessage],
        tokenizer
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Crée exemples de training depuis une conversation

        Stratégie:
        - Chaque tour = 1 exemple de training
        - Contexte = tous les tours précédents
        - Loss seulement sur la réponse (pas sur l'instruction)

        Args:
            conversation: Liste de messages
            tokenizer: Tokenizer

        Returns:
            Liste de (input_ids, labels) pour chaque tour
        """

        examples = []

        # Accumuler contexte au fur et à mesure
        context = []

        for i, msg in enumerate(conversation):
            if msg.role == "user":
                # Ajouter user message au contexte
                context.append(f"{self.instruction_template}\n{msg.content}")

            elif msg.role == "assistant":
                # Créer exemple de training

                # Contexte complet jusqu'ici
                full_context = "\n\n".join(context)

                # Ajouter template de réponse
                prompt = f"{full_context}\n\n{self.response_template}\n"

                # Response
                response = msg.content

                # Tokenize
                prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
                response_ids = tokenizer.encode(response, add_special_tokens=False)

                # Combiner
                input_ids = prompt_ids + response_ids

                # Labels: -100 pour prompt (no loss), response_ids pour response (compute loss)
                labels = [-100] * len(prompt_ids) + response_ids

                # Truncate si nécessaire
                if len(input_ids) > self.max_length:
                    # Garder fin (plus récent)
                    input_ids = input_ids[-self.max_length:]
                    labels = labels[-self.max_length:]

                examples.append((
                    torch.tensor(input_ids),
                    torch.tensor(labels)
                ))

                # Ajouter response au contexte pour prochain tour
                context.append(f"{self.response_template}\n{response}")

        return examples

    def intelligent_truncation(
        self,
        conversation: List[ChatMessage],
        max_tokens: int
    ) -> List[ChatMessage]:
        """
        Truncate intelligemment une conversation

        Stratégies:
        1. Garder system message
        2. Garder les N derniers tours (plus récents)
        3. Summarize ancien contexte (optionnel)

        Args:
            conversation: Conversation complète
            max_tokens: Max tokens autorisés

        Returns:
            Conversation tronquée
        """

        # Extract system message
        system_msg = None
        messages = []

        for msg in conversation:
            if msg.role == "system":
                system_msg = msg
            else:
                messages.append(msg)

        # Estimer tokens (rough: 1 token ≈ 4 chars)
        def estimate_tokens(msg: ChatMessage) -> int:
            return len(msg.content) // 4

        # Garder system
        truncated = []
        if system_msg:
            truncated.append(system_msg)
            remaining_tokens = max_tokens - estimate_tokens(system_msg)
        else:
            remaining_tokens = max_tokens

        # Ajouter messages du plus récent au plus ancien
        for msg in reversed(messages):
            msg_tokens = estimate_tokens(msg)
            if msg_tokens <= remaining_tokens:
                truncated.insert(1 if system_msg else 0, msg)  # Insert after system
                remaining_tokens -= msg_tokens
            else:
                break  # Plus d'espace

        return truncated


class ConversationQualityChecker:
    """
    Vérifie la qualité des conversations pour training

    Checks:
    1. Cohérence (pas de contradictions)
    2. Longueur appropriée
    3. Alternance user/assistant
    4. Pas de répétitions
    """

    @staticmethod
    def check_alternation(conversation: List[ChatMessage]) -> bool:
        """
        Vérifie alternance user/assistant

        Returns:
            True si alternance correcte
        """

        # Filter out system messages
        non_system = [msg for msg in conversation if msg.role != "system"]

        if len(non_system) < 2:
            return True

        # Check alternation
        for i in range(len(non_system) - 1):
            current_role = non_system[i].role
            next_role = non_system[i + 1].role

            # Doit alterner
            if current_role == next_role:
                return False

        return True

    @staticmethod
    def check_length(conversation: List[ChatMessage]) -> Dict[str, any]:
        """
        Vérifie longueurs des messages

        Returns:
            Stats de longueur
        """

        lengths = [len(msg.content) for msg in conversation]

        return {
            "min_length": min(lengths) if lengths else 0,
            "max_length": max(lengths) if lengths else 0,
            "avg_length": sum(lengths) / len(lengths) if lengths else 0,
            "num_messages": len(lengths),
            "total_length": sum(lengths)
        }

    @staticmethod
    def check_repetition(conversation: List[ChatMessage]) -> bool:
        """
        Détecte répétitions excessives

        Returns:
            True si pas de répétition excessive
        """

        contents = [msg.content.lower() for msg in conversation]

        # Check exact duplicates
        if len(contents) != len(set(contents)):
            return False

        # Check substring repetition (simple)
        for i, content in enumerate(contents):
            for other in contents[i+1:]:
                # Si un message contient >80% d'un autre
                if len(content) > 20:  # Skip very short
                    overlap = sum(1 for c in content if c in other)
                    if overlap / len(content) > 0.8:
                        return False

        return True

    @staticmethod
    def validate_conversation(conversation: List[ChatMessage]) -> Dict[str, any]:
        """
        Valide une conversation complète

        Returns:
            Dict avec résultats de validation
        """

        checks = {
            "alternation": ConversationQualityChecker.check_alternation(conversation),
            "length_stats": ConversationQualityChecker.check_length(conversation),
            "no_repetition": ConversationQualityChecker.check_repetition(conversation)
        }

        # Overall valid si tous les checks passent
        checks["valid"] = (
            checks["alternation"] and
            checks["no_repetition"] and
            checks["length_stats"]["min_length"] > 5  # Pas trop court
        )

        return checks


# Exemple
if __name__ == "__main__":
    print("\n=== Multi-Turn Conversation Processing ===\n")

    # Conversation exemple
    conversation = [
        ChatMessage(role="system", content="You are a helpful assistant."),
        ChatMessage(role="user", content="What is machine learning?"),
        ChatMessage(role="assistant", content="Machine learning is a subset of AI..."),
        ChatMessage(role="user", content="Can you give an example?"),
        ChatMessage(role="assistant", content="Sure! Email spam detection is a common example...")
    ]

    # Quality checks
    checker = ConversationQualityChecker()
    validation = checker.validate_conversation(conversation)

    print("Validation results:")
    print(f"  Alternation: {'✅' if validation['alternation'] else '❌'}")
    print(f"  No repetition: {'✅' if validation['no_repetition'] else '❌'}")
    print(f"  Overall valid: {'✅' if validation['valid'] else '❌'}")
    print()
    print("Length stats:")
    stats = validation['length_stats']
    print(f"  Num messages: {stats['num_messages']}")
    print(f"  Avg length: {stats['avg_length']:.0f} chars")
    print(f"  Total length: {stats['total_length']} chars")

    print("\n" + "="*60 + "\n")

    # Truncation test
    processor = MultiTurnConversationProcessor(max_length=512)

    # Add more messages to test truncation
    long_conversation = conversation + [
        ChatMessage(role="user", content="What about deep learning?"),
        ChatMessage(role="assistant", content="Deep learning uses neural networks..."),
        ChatMessage(role="user", content="How is it different?"),
        ChatMessage(role="assistant", content="Deep learning can learn hierarchical features...")
    ]

    truncated = processor.intelligent_truncation(long_conversation, max_tokens=200)

    print(f"Original conversation: {len(long_conversation)} messages")
    print(f"Truncated conversation: {len(truncated)} messages")
    print()
    print("Truncated messages:")
    for msg in truncated:
        preview = msg.content[:50] + "..." if len(msg.content) > 50 else msg.content
        print(f"  [{msg.role}] {preview}")
```

## 5. Best Practices pour Instruction Tuning

```python
"""
Best practices basées sur la recherche et expérience pratique
"""

INSTRUCTION_TUNING_BEST_PRACTICES = {
    "Data Quality": {
        "diversity": {
            "description": "Diversité des tâches et domaines",
            "recommendations": [
                "Au moins 10+ types de tâches différentes",
                "Couvrir multiples domaines (science, art, tech, etc.)",
                "Varier difficulté (simple à complexe)",
                "Inclure edge cases"
            ],
            "example_tasks": [
                "Question answering",
                "Summarization",
                "Translation",
                "Code generation",
                "Creative writing",
                "Math problem solving",
                "Reasoning",
                "Classification"
            ]
        },

        "quality_over_quantity": {
            "description": "Mieux 10k exemples de qualité que 100k médiocres",
            "recommendations": [
                "Vérifier corrections manuellement (échantillon)",
                "Utiliser GPT-4 plutôt que GPT-3.5 pour génération",
                "Filtrer exemples trop courts ou incohérents",
                "Valider logique et factualité"
            ],
            "quality_metrics": {
                "instruction_clarity": ">90%",
                "output_correctness": ">95%",
                "output_completeness": ">90%",
                "no_hallucination": ">98%"
            }
        },

        "response_length": {
            "description": "Longueur appropriée des réponses",
            "recommendations": {
                "too_short": "Éviter <20 chars (sauf yes/no)",
                "too_long": "Limiter à ~500 tokens (lisibilité)",
                "optimal": "100-300 tokens pour la plupart des tâches"
            }
        }
    },

    "Training Strategy": {
        "dataset_size": {
            "description": "Taille recommandée du dataset",
            "recommendations": {
                "minimum": "1k-5k exemples (proof of concept)",
                "good": "10k-50k exemples (production-ready)",
                "excellent": "50k-100k+ exemples (SOTA)"
            }
        },

        "epochs": {
            "description": "Nombre d'époques de training",
            "recommendations": {
                "small_dataset_<5k": "3-5 epochs",
                "medium_dataset_5k-50k": "2-3 epochs",
                "large_dataset_>50k": "1-2 epochs"
            },
            "warning": "Trop d'époques → overfitting sur format, perte de connaissances générales"
        },

        "learning_rate": {
            "description": "Learning rate pour instruction tuning",
            "recommendations": {
                "full_finetuning": "1e-5 à 5e-5",
                "LoRA": "1e-4 à 3e-4",
                "QLoRA": "2e-4 à 5e-4"
            }
        },

        "loss_masking": {
            "description": "Compute loss seulement sur response (pas instruction)",
            "recommendations": [
                "Masquer instructions avec labels=-100",
                "Compute loss seulement sur assistant responses",
                "Améliore apprentissage et réduit overfitting"
            ]
        }
    },

    "Evaluation": {
        "metrics": {
            "description": "Comment évaluer un modèle instruction-tuned",
            "automatic_metrics": [
                "Perplexity (lower = better)",
                "ROUGE/BLEU pour génération",
                "Exact Match pour Q&A",
                "Pass@k pour code"
            ],
            "human_evaluation": [
                "Helpfulness (suit l'instruction?)",
                "Harmlessness (safe?)",
                "Honesty (factuel?)",
                "Overall quality"
            ]
        },

        "test_set": {
            "description": "Création du test set",
            "recommendations": [
                "Séparer train/test AVANT training",
                "Test set = 10-20% du dataset",
                "Inclure toutes les catégories de tâches",
                "Ajouter out-of-distribution examples"
            ]
        }
    },

    "Common Pitfalls": {
        "overfitting_on_format": {
            "description": "Modèle apprend le format mais pas le contenu",
            "symptoms": [
                "Génère toujours '### Response:' même hors contexte",
                "Perte de capacités de base du modèle",
                "Répond seulement dans le format d'entraînement"
            ],
            "solutions": [
                "Réduire nombre d'époques",
                "Augmenter diversité du dataset",
                "Utiliser PEFT (LoRA) plutôt que full FT",
                "Mixer avec pre-training data (10-20%)"
            ]
        },

        "catastrophic_forgetting": {
            "description": "Modèle oublie connaissances pré-training",
            "symptoms": [
                "Performance dégradée sur tâches générales",
                "Moins de connaissances factuelles",
                "Incapable de faire tasks non vus en instruction tuning"
            ],
            "solutions": [
                "Learning rate plus bas",
                "Moins d'époques",
                "Utiliser LoRA/QLoRA",
                "Replay pre-training data"
            ]
        },

        "instruction_following_vs_knowledge": {
            "description": "Trade-off entre suivre instructions et connaissances",
            "balance": "Instruction tuning améliore following, peut réduire knowledge retrieval",
            "solution": "Combiner avec RAG pour knowledge, instruction tuning pour following"
        }
    }
}


def print_best_practices():
    """Print best practices"""

    print("="*60)
    print("INSTRUCTION TUNING BEST PRACTICES")
    print("="*60)
    print()

    for category, items in INSTRUCTION_TUNING_BEST_PRACTICES.items():
        print(f"\n### {category}")
        print()

        for key, value in items.items():
            print(f"**{key.replace('_', ' ').title()}**")

            if isinstance(value, dict):
                if 'description' in value:
                    print(f"  {value['description']}")
                if 'recommendations' in value:
                    if isinstance(value['recommendations'], list):
                        for rec in value['recommendations'][:3]:
                            print(f"    • {rec}")
                    elif isinstance(value['recommendations'], dict):
                        for k, v in list(value['recommendations'].items())[:3]:
                            print(f"    • {k}: {v}")
            print()


if __name__ == "__main__":
    print_best_practices()
```

*[Suite avec projet pratique dans la partie 3...]*
