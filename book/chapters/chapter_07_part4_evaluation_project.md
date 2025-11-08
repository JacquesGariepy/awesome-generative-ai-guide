# Chapitre 7 (Suite 3): Évaluation et Projet Pratique

## 3.5 Métriques d'Évaluation

```python
"""
Métriques d'évaluation pour fine-tuning de LLMs
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
import math
import numpy as np
from collections import Counter
import re


@dataclass
class EvaluationMetrics:
    """Container pour toutes les métriques d'évaluation"""
    loss: float
    perplexity: float
    bleu: Optional[float] = None
    rouge_1: Optional[float] = None
    rouge_2: Optional[float] = None
    rouge_l: Optional[float] = None
    exact_match: Optional[float] = None
    f1_score: Optional[float] = None


class MetricsCalculator:
    """Calculateur de métriques pour évaluation de LLMs"""

    @staticmethod
    def compute_perplexity(loss: float) -> float:
        """
        Calcule perplexity à partir de la loss

        Perplexity = exp(loss)

        Interprétation:
        - Perplexity de 1: Modèle parfait (prédit toujours le bon token)
        - Perplexity de 10: Modèle hésite entre ~10 tokens
        - Perplexity de 100: Modèle très incertain
        - Perplexity de vocab_size: Modèle random (baseline)
        """
        return math.exp(loss)

    @staticmethod
    def compute_bleu(
        predictions: List[str],
        references: List[str],
        max_n: int = 4
    ) -> float:
        """
        Calcule BLEU score (BiLingual Evaluation Understudy)

        Utilisé pour:
        - Traduction
        - Génération de texte
        - Summarization

        Score de 0 à 1 (ou 0 à 100 si multiplié par 100)
        - 0: Aucune correspondance
        - 1: Correspondance parfaite

        Args:
            predictions: Liste de textes générés
            references: Liste de textes de référence
            max_n: Maximum n-gram order (default 4 pour BLEU-4)

        Returns:
            BLEU score
        """
        from collections import defaultdict

        def get_ngrams(tokens: List[str], n: int) -> Counter:
            """Extrait n-grams d'une liste de tokens"""
            return Counter([tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)])

        def tokenize(text: str) -> List[str]:
            """Tokenisation simple"""
            return text.lower().split()

        total_precision = []

        for pred, ref in zip(predictions, references):
            pred_tokens = tokenize(pred)
            ref_tokens = tokenize(ref)

            precisions = []

            for n in range(1, max_n + 1):
                pred_ngrams = get_ngrams(pred_tokens, n)
                ref_ngrams = get_ngrams(ref_tokens, n)

                if not pred_ngrams:
                    precisions.append(0.0)
                    continue

                # Count matches
                matches = sum(min(pred_ngrams[ng], ref_ngrams[ng]) for ng in pred_ngrams)
                total = sum(pred_ngrams.values())

                precision = matches / total if total > 0 else 0.0
                precisions.append(precision)

            # Geometric mean of precisions
            if all(p > 0 for p in precisions):
                bleu = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
            else:
                bleu = 0.0

            # Brevity penalty
            pred_len = len(pred_tokens)
            ref_len = len(ref_tokens)
            if pred_len < ref_len:
                bp = math.exp(1 - ref_len / pred_len)
            else:
                bp = 1.0

            total_precision.append(bleu * bp)

        return sum(total_precision) / len(total_precision) if total_precision else 0.0

    @staticmethod
    def compute_rouge(
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """
        Calcule ROUGE scores (Recall-Oriented Understudy for Gisting Evaluation)

        Utilisé pour:
        - Summarization
        - Question answering
        - Génération de texte

        Types:
        - ROUGE-1: Unigram overlap
        - ROUGE-2: Bigram overlap
        - ROUGE-L: Longest Common Subsequence

        Returns:
            Dict avec rouge-1, rouge-2, rouge-l (F1 scores)
        """

        def get_ngrams(tokens: List[str], n: int) -> Counter:
            return Counter([tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)])

        def tokenize(text: str) -> List[str]:
            return text.lower().split()

        def lcs_length(seq1: List[str], seq2: List[str]) -> int:
            """Longest Common Subsequence length"""
            m, n = len(seq1), len(seq2)
            dp = [[0] * (n + 1) for _ in range(m + 1)]

            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if seq1[i-1] == seq2[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])

            return dp[m][n]

        rouge_1_scores = []
        rouge_2_scores = []
        rouge_l_scores = []

        for pred, ref in zip(predictions, references):
            pred_tokens = tokenize(pred)
            ref_tokens = tokenize(ref)

            # ROUGE-1
            pred_1grams = get_ngrams(pred_tokens, 1)
            ref_1grams = get_ngrams(ref_tokens, 1)

            overlap_1 = sum(min(pred_1grams[ng], ref_1grams[ng]) for ng in pred_1grams)

            if len(ref_1grams) > 0 and len(pred_1grams) > 0:
                recall_1 = overlap_1 / sum(ref_1grams.values())
                precision_1 = overlap_1 / sum(pred_1grams.values())

                if recall_1 + precision_1 > 0:
                    f1_1 = 2 * recall_1 * precision_1 / (recall_1 + precision_1)
                else:
                    f1_1 = 0.0
            else:
                f1_1 = 0.0

            rouge_1_scores.append(f1_1)

            # ROUGE-2
            pred_2grams = get_ngrams(pred_tokens, 2)
            ref_2grams = get_ngrams(ref_tokens, 2)

            overlap_2 = sum(min(pred_2grams[ng], ref_2grams[ng]) for ng in pred_2grams)

            if len(ref_2grams) > 0 and len(pred_2grams) > 0:
                recall_2 = overlap_2 / sum(ref_2grams.values())
                precision_2 = overlap_2 / sum(pred_2grams.values())

                if recall_2 + precision_2 > 0:
                    f1_2 = 2 * recall_2 * precision_2 / (recall_2 + precision_2)
                else:
                    f1_2 = 0.0
            else:
                f1_2 = 0.0

            rouge_2_scores.append(f1_2)

            # ROUGE-L
            lcs_len = lcs_length(pred_tokens, ref_tokens)

            if len(ref_tokens) > 0 and len(pred_tokens) > 0:
                recall_l = lcs_len / len(ref_tokens)
                precision_l = lcs_len / len(pred_tokens)

                if recall_l + precision_l > 0:
                    f1_l = 2 * recall_l * precision_l / (recall_l + precision_l)
                else:
                    f1_l = 0.0
            else:
                f1_l = 0.0

            rouge_l_scores.append(f1_l)

        return {
            "rouge-1": sum(rouge_1_scores) / len(rouge_1_scores) if rouge_1_scores else 0.0,
            "rouge-2": sum(rouge_2_scores) / len(rouge_2_scores) if rouge_2_scores else 0.0,
            "rouge-l": sum(rouge_l_scores) / len(rouge_l_scores) if rouge_l_scores else 0.0,
        }

    @staticmethod
    def compute_exact_match(
        predictions: List[str],
        references: List[str]
    ) -> float:
        """
        Calcule Exact Match score

        Utilisé pour:
        - Question answering
        - Classification

        Returns:
            Pourcentage de matches exacts (0 à 1)
        """

        def normalize(text: str) -> str:
            """Normalise le texte pour comparaison"""
            text = text.lower()
            text = re.sub(r'\s+', ' ', text)  # Normaliser espaces
            text = re.sub(r'[^\w\s]', '', text)  # Retirer ponctuation
            return text.strip()

        matches = sum(
            normalize(pred) == normalize(ref)
            for pred, ref in zip(predictions, references)
        )

        return matches / len(predictions) if predictions else 0.0

    @staticmethod
    def compute_f1(
        predictions: List[str],
        references: List[str]
    ) -> float:
        """
        Calcule F1 score token-level

        Utilisé pour:
        - Question answering
        - Named Entity Recognition

        Returns:
            F1 score moyen (0 à 1)
        """

        def tokenize(text: str) -> set:
            text = text.lower()
            text = re.sub(r'[^\w\s]', '', text)
            return set(text.split())

        f1_scores = []

        for pred, ref in zip(predictions, references):
            pred_tokens = tokenize(pred)
            ref_tokens = tokenize(ref)

            if not pred_tokens and not ref_tokens:
                f1_scores.append(1.0)
                continue

            if not pred_tokens or not ref_tokens:
                f1_scores.append(0.0)
                continue

            overlap = len(pred_tokens & ref_tokens)

            precision = overlap / len(pred_tokens)
            recall = overlap / len(ref_tokens)

            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0

            f1_scores.append(f1)

        return sum(f1_scores) / len(f1_scores) if f1_scores else 0.0


class MetricsComparison:
    """Guide pour choisir les bonnes métriques"""

    @staticmethod
    def get_metrics_guide() -> Dict[str, Dict]:
        """Retourne un guide des métriques par tâche"""

        return {
            "Text Generation": {
                "primary_metrics": ["Perplexity", "BLEU"],
                "secondary_metrics": ["ROUGE-L"],
                "explanation": "Perplexity mesure la qualité du modèle de langage, BLEU mesure la similarité avec référence",
                "thresholds": {
                    "perplexity": {"excellent": "<10", "good": "10-30", "poor": ">100"},
                    "bleu": {"excellent": ">0.4", "good": "0.2-0.4", "poor": "<0.2"}
                }
            },

            "Summarization": {
                "primary_metrics": ["ROUGE-1", "ROUGE-2", "ROUGE-L"],
                "secondary_metrics": ["BLEU"],
                "explanation": "ROUGE mesure le overlap avec le résumé de référence",
                "thresholds": {
                    "rouge-1": {"excellent": ">0.4", "good": "0.3-0.4", "poor": "<0.3"},
                    "rouge-2": {"excellent": ">0.2", "good": "0.15-0.2", "poor": "<0.15"}
                }
            },

            "Question Answering": {
                "primary_metrics": ["Exact Match", "F1 Score"],
                "secondary_metrics": ["ROUGE-L"],
                "explanation": "EM mesure les réponses parfaites, F1 mesure l'overlap token-level",
                "thresholds": {
                    "exact_match": {"excellent": ">0.8", "good": "0.6-0.8", "poor": "<0.6"},
                    "f1": {"excellent": ">0.85", "good": "0.7-0.85", "poor": "<0.7"}
                }
            },

            "Translation": {
                "primary_metrics": ["BLEU"],
                "secondary_metrics": ["Perplexity"],
                "explanation": "BLEU est le standard pour la traduction automatique",
                "thresholds": {
                    "bleu": {"excellent": ">0.5", "good": "0.3-0.5", "poor": "<0.3"}
                }
            },

            "Code Generation": {
                "primary_metrics": ["Exact Match", "Pass@k"],
                "secondary_metrics": ["BLEU"],
                "explanation": "EM mesure code identique, Pass@k mesure si le code fonctionne",
                "thresholds": {
                    "exact_match": {"excellent": ">0.5", "good": "0.3-0.5", "poor": "<0.3"},
                    "pass@1": {"excellent": ">0.7", "good": "0.5-0.7", "poor": "<0.5"}
                }
            },

            "Classification": {
                "primary_metrics": ["Accuracy", "F1 Score"],
                "secondary_metrics": ["Precision", "Recall"],
                "explanation": "Accuracy pour datasets balancés, F1 pour déséquilibrés",
                "thresholds": {
                    "accuracy": {"excellent": ">0.95", "good": "0.85-0.95", "poor": "<0.85"},
                    "f1": {"excellent": ">0.9", "good": "0.8-0.9", "poor": "<0.8"}
                }
            }
        }


# Exemple d'utilisation
if __name__ == "__main__":
    calc = MetricsCalculator()

    print("=== Démonstration des Métriques ===\n")

    # Exemple 1: Perplexity
    print("1. Perplexity")
    losses = [0.5, 1.0, 2.0, 3.0, 5.0]
    for loss in losses:
        ppl = calc.compute_perplexity(loss)
        print(f"  Loss {loss:.1f} → Perplexity {ppl:.2f}")
    print()

    # Exemple 2: BLEU
    print("2. BLEU Score")
    predictions = [
        "The cat sat on the mat",
        "I love machine learning",
        "This is a test"
    ]
    references = [
        "The cat is sitting on the mat",
        "I enjoy machine learning",
        "This is a test"
    ]
    bleu = calc.compute_bleu(predictions, references)
    print(f"  BLEU Score: {bleu:.3f}")
    print()

    # Exemple 3: ROUGE
    print("3. ROUGE Scores")
    rouge = calc.compute_rouge(predictions, references)
    for metric, score in rouge.items():
        print(f"  {metric}: {score:.3f}")
    print()

    # Exemple 4: Exact Match et F1
    print("4. Exact Match et F1")
    em = calc.compute_exact_match(predictions, references)
    f1 = calc.compute_f1(predictions, references)
    print(f"  Exact Match: {em:.3f}")
    print(f"  F1 Score: {f1:.3f}")
    print()

    print("\n" + "="*60 + "\n")

    # Guide des métriques
    print("=== Guide des Métriques par Tâche ===\n")

    guide = MetricsComparison.get_metrics_guide()

    for task, info in list(guide.items())[:3]:  # Afficher 3 premières tâches
        print(f"### {task}")
        print(f"Métriques primaires: {', '.join(info['primary_metrics'])}")
        print(f"Explication: {info['explanation']}")
        print("Seuils:")
        for metric, thresholds in info['thresholds'].items():
            print(f"  {metric}:")
            print(f"    Excellent: {thresholds['excellent']}")
            print(f"    Bon: {thresholds['good']}")
            print(f"    Faible: {thresholds['poor']}")
        print()
```

## 4. Projet Pratique Complet: Fine-tune Llama 2 pour Q&A

```python
"""
Projet complet: Fine-tuning de Llama 2 7B pour Question Answering

Ce projet couvre:
1. Préparation des données
2. Configuration du modèle et tokenizer
3. Setup du training avec HuggingFace Trainer
4. Monitoring et évaluation
5. Sauvegarde et inférence

Prérequis:
- GPU avec >=16GB VRAM (ou utiliser QLoRA pour 8GB)
- HuggingFace account avec accès à Llama 2
"""

import os
from typing import Dict, List, Optional
from dataclasses import dataclass
import json

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from datasets import load_dataset
import evaluate


# =====================
# 1. Configuration
# =====================

@dataclass
class ProjectConfig:
    """Configuration du projet"""

    # Modèle
    model_name: str = "meta-llama/Llama-2-7b-hf"
    max_seq_length: int = 512

    # Dataset
    dataset_name: str = "squad"  # Stanford Question Answering Dataset
    num_train_samples: int = 10000  # Pour accélérer (utiliser -1 pour tout)
    num_eval_samples: int = 1000

    # Training
    output_dir: str = "./llama2-qa-finetuned"
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01

    # Evaluation
    eval_steps: int = 500
    save_steps: int = 500
    logging_steps: int = 50

    # System
    use_fp16: bool = False
    use_bf16: bool = True  # Recommandé pour Ampere+ GPUs
    gradient_checkpointing: bool = True


# =====================
# 2. Préparation des Données
# =====================

class QADatasetPreparator:
    """Prépare le dataset SQuAD pour fine-tuning"""

    def __init__(self, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def format_qa_example(self, example: Dict) -> str:
        """
        Formate un exemple Q&A en format instruction

        Format:
        ### Question: {question}
        ### Context: {context}
        ### Answer: {answer}
        """

        question = example['question']
        context = example['context']
        answer = example['answers']['text'][0] if example['answers']['text'] else "N/A"

        formatted = f"""### Question: {question}

### Context: {context}

### Answer: {answer}"""

        return formatted

    def tokenize_function(self, examples):
        """Tokenize les exemples"""

        # Formater tous les exemples
        texts = [self.format_qa_example(ex) for ex in examples]

        # Tokenizer
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors=None  # Return lists, not tensors
        )

        # Pour causal LM, labels = input_ids
        tokenized["labels"] = tokenized["input_ids"].copy()

        return tokenized

    def prepare_dataset(self, num_train: int = -1, num_eval: int = -1):
        """
        Charge et prépare le dataset SQuAD

        Returns:
            Tuple de (train_dataset, eval_dataset)
        """

        # Charger SQuAD
        dataset = load_dataset("squad")

        # Limiter la taille si demandé
        if num_train > 0:
            dataset['train'] = dataset['train'].select(range(num_train))
        if num_eval > 0:
            dataset['validation'] = dataset['validation'].select(range(num_eval))

        # Convertir en liste de dicts pour faciliter le formatage
        train_examples = []
        for i in range(len(dataset['train'])):
            example = {
                'question': dataset['train'][i]['question'],
                'context': dataset['train'][i]['context'],
                'answers': dataset['train'][i]['answers']
            }
            train_examples.append(example)

        eval_examples = []
        for i in range(len(dataset['validation'])):
            example = {
                'question': dataset['validation'][i]['question'],
                'context': dataset['validation'][i]['context'],
                'answers': dataset['validation'][i]['answers']
            }
            eval_examples.append(example)

        # Tokenize
        print(f"Tokenizing {len(train_examples)} training examples...")
        train_tokenized = self.tokenize_function(train_examples)

        print(f"Tokenizing {len(eval_examples)} eval examples...")
        eval_tokenized = self.tokenize_function(eval_examples)

        return train_tokenized, eval_tokenized


# =====================
# 3. Custom Dataset Class
# =====================

class TokenizedQADataset(Dataset):
    """Dataset PyTorch pour données tokenizées"""

    def __init__(self, tokenized_data):
        self.input_ids = tokenized_data['input_ids']
        self.attention_mask = tokenized_data['attention_mask']
        self.labels = tokenized_data['labels']

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.input_ids[idx], dtype=torch.long),
            'attention_mask': torch.tensor(self.attention_mask[idx], dtype=torch.long),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


# =====================
# 4. Training Pipeline
# =====================

class Llama2QATrainer:
    """Pipeline complet pour fine-tuning Llama 2 sur Q&A"""

    def __init__(self, config: ProjectConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.trainer = None

    def setup(self):
        """Configure tokenizer et modèle"""

        print("=== Setup ===")

        # Tokenizer
        print(f"Loading tokenizer: {self.config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            use_fast=True,
            trust_remote_code=True
        )

        # Add pad token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Modèle
        print(f"Loading model: {self.config.model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.bfloat16 if self.config.use_bf16 else torch.float32,
            device_map="auto",  # Automatically distribute across GPUs
            trust_remote_code=True
        )

        # Enable gradient checkpointing
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        print(f"Model loaded. Parameters: {self.model.num_parameters():,}")
        print()

    def prepare_data(self):
        """Prépare les datasets"""

        print("=== Data Preparation ===")

        preparator = QADatasetPreparator(
            self.tokenizer,
            max_length=self.config.max_seq_length
        )

        train_tokenized, eval_tokenized = preparator.prepare_dataset(
            num_train=self.config.num_train_samples,
            num_eval=self.config.num_eval_samples
        )

        train_dataset = TokenizedQADataset(train_tokenized)
        eval_dataset = TokenizedQADataset(eval_tokenized)

        print(f"Train dataset: {len(train_dataset)} examples")
        print(f"Eval dataset: {len(eval_dataset)} examples")
        print()

        return train_dataset, eval_dataset

    def create_trainer(self, train_dataset, eval_dataset):
        """Crée le Trainer HuggingFace"""

        print("=== Creating Trainer ===")

        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            logging_dir=f"{self.config.output_dir}/logs",
            logging_steps=self.config.logging_steps,
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            save_total_limit=3,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=self.config.use_fp16,
            bf16=self.config.use_bf16,
            gradient_checkpointing=self.config.gradient_checkpointing,
            optim="adamw_torch",
            report_to=["tensorboard"],
            push_to_hub=False,
        )

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # Causal LM, not masked LM
        )

        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=3,
                    early_stopping_threshold=0.01
                )
            ]
        )

        effective_batch = (
            self.config.batch_size *
            self.config.gradient_accumulation_steps *
            torch.cuda.device_count()
        )

        print(f"Effective batch size: {effective_batch}")
        print(f"Total training steps: {len(train_dataset) // effective_batch * self.config.num_epochs}")
        print()

    def train(self):
        """Lance le training"""

        print("=== Training ===")
        print("Starting training... (this may take several hours)")
        print()

        # Train
        train_result = self.trainer.train()

        # Save final model
        print("\nSaving final model...")
        self.trainer.save_model(self.config.output_dir)
        self.tokenizer.save_pretrained(self.config.output_dir)

        # Print metrics
        print("\n=== Training Complete ===")
        print(f"Final train loss: {train_result.training_loss:.4f}")

        metrics = self.trainer.evaluate()
        print(f"Final eval loss: {metrics['eval_loss']:.4f}")
        print(f"Final perplexity: {torch.exp(torch.tensor(metrics['eval_loss'])):.2f}")

        return train_result, metrics

    def run_full_pipeline(self):
        """Exécute le pipeline complet"""

        # Setup
        self.setup()

        # Prepare data
        train_dataset, eval_dataset = self.prepare_data()

        # Create trainer
        self.create_trainer(train_dataset, eval_dataset)

        # Train
        train_result, metrics = self.train()

        return train_result, metrics


# =====================
# 5. Inférence
# =====================

class Llama2QAInference:
    """Inférence avec le modèle fine-tuné"""

    def __init__(self, model_path: str):
        print(f"Loading fine-tuned model from {model_path}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

        print("Model loaded successfully!")

    def answer_question(
        self,
        question: str,
        context: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7
    ) -> str:
        """
        Répond à une question donnée un contexte

        Args:
            question: La question
            context: Le contexte
            max_new_tokens: Nombre max de tokens à générer
            temperature: Température de génération

        Returns:
            La réponse générée
        """

        # Format prompt
        prompt = f"""### Question: {question}

### Context: {context}

### Answer:"""

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
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract answer (everything after "### Answer:")
        if "### Answer:" in full_response:
            answer = full_response.split("### Answer:")[-1].strip()
        else:
            answer = full_response

        return answer


# =====================
# 6. Main
# =====================

def main():
    """Point d'entrée principal"""

    print("="*60)
    print("Llama 2 Fine-tuning pour Question Answering")
    print("="*60)
    print()

    # Configuration
    config = ProjectConfig(
        num_train_samples=1000,  # Utiliser petit dataset pour demo
        num_eval_samples=200,
        num_epochs=1,  # 1 epoch pour demo
    )

    # Training pipeline
    trainer_pipeline = Llama2QATrainer(config)

    # Run (commenté pour ne pas lancer vraiment)
    # train_result, metrics = trainer_pipeline.run_full_pipeline()

    print("Training pipeline créé!")
    print("\nPour lancer le training:")
    print("  train_result, metrics = trainer_pipeline.run_full_pipeline()")
    print("\nPour l'inférence:")
    print("  inference = Llama2QAInference('./llama2-qa-finetuned')")
    print("  answer = inference.answer_question(question, context)")


if __name__ == "__main__":
    main()
```

## 5. Conclusion et Best Practices

### Récapitulatif du Chapitre

Dans ce chapitre, nous avons couvert l'ensemble du processus de fine-tuning:

1. **Choix de l'Approche**
   - Comparaison: Prompt Engineering → Few-Shot → RAG → Fine-tuning → Pré-entraînement
   - Arbres de décision pour choisir
   - Types de fine-tuning (Full, LoRA, QLoRA, etc.)

2. **Préparation des Données**
   - Formats: Completion, Instruction, Chat
   - Quality checking automatisé
   - Guidelines pour des données de qualité

3. **Hyperparamètres**
   - Configuration complète
   - Presets pour différents cas d'usage
   - Estimation du temps de training

4. **Prévention Overfitting**
   - Détection automatique de signaux
   - 8 stratégies de régularisation
   - Recommandations personnalisées

5. **Évaluation**
   - Perplexity, BLEU, ROUGE, Exact Match, F1
   - Guide des métriques par tâche
   - Seuils de performance

6. **Projet Pratique**
   - Pipeline complet Llama 2 fine-tuning
   - Code production-ready
   - Inférence

### Best Practices Finales

```python
"""
Checklist complète pour un fine-tuning réussi
"""

FINETUNING_BEST_PRACTICES = {
    "Avant de commencer": [
        "✅ Essayer prompt engineering d'abord",
        "✅ Essayer few-shot learning",
        "✅ Considérer RAG si besoin de connaissances externes",
        "✅ Vérifier que vous avez ≥1000 exemples de qualité",
        "✅ Définir des métriques d'évaluation claires"
    ],

    "Préparation des données": [
        "✅ Nettoyer et valider tous les exemples",
        "✅ Utiliser DataQualityChecker",
        "✅ Split train/eval/test (80/10/10)",
        "✅ Vérifier la distribution (pas de biais)",
        "✅ Format cohérent (instruction, chat, ou completion)"
    ],

    "Configuration": [
        "✅ Commencer avec un preset adapté à votre cas",
        "✅ Utiliser BF16 si GPU Ampere+",
        "✅ Activer gradient checkpointing si VRAM limité",
        "✅ Batch size: remplir VRAM à ~90%",
        "✅ Learning rate: 2e-5 (full) ou 1e-4 (LoRA)"
    ],

    "Training": [
        "✅ Monitorer train ET eval loss",
        "✅ Utiliser early stopping",
        "✅ Sauvegarder checkpoints régulièrement",
        "✅ Logger toutes les métriques (TensorBoard)",
        "✅ Faire au moins 1 epoch complet"
    ],

    "Évaluation": [
        "✅ Calculer métriques appropriées à votre tâche",
        "✅ Tester sur exemples réels (pas seulement métriques)",
        "✅ Comparer avec baseline (modèle base)",
        "✅ Vérifier overfitting (train vs eval)",
        "✅ Valider qualitativement les outputs"
    ],

    "Debugging": [
        "❌ Loss ne descend pas → Réduire LR, vérifier données",
        "❌ Loss explose → Réduire LR, activer gradient clipping",
        "❌ Overfitting → Weight decay, dropout, plus de données",
        "❌ VRAM overflow → Réduire batch, activer checkpointing",
        "❌ Training trop lent → Désactiver checkpointing, BF16"
    ],

    "Production": [
        "✅ Sauvegarder config complète (reproductibilité)",
        "✅ Versionner modèle et données",
        "✅ Documenter hyperparams et métriques",
        "✅ Tester inférence performance",
        "✅ Monitorer production metrics"
    ]
}


def print_best_practices():
    """Affiche les best practices"""

    print("="*60)
    print("BEST PRACTICES POUR FINE-TUNING")
    print("="*60)
    print()

    for section, practices in FINETUNING_BEST_PRACTICES.items():
        print(f"### {section}")
        for practice in practices:
            print(f"  {practice}")
        print()


if __name__ == "__main__":
    print_best_practices()
```

### Ressources Additionnelles

**Documentation**:
- HuggingFace Transformers: https://huggingface.co/docs/transformers
- HuggingFace PEFT: https://huggingface.co/docs/peft
- PyTorch: https://pytorch.org/docs

**Datasets**:
- HuggingFace Hub: https://huggingface.co/datasets
- SQuAD: https://rajpurkar.github.io/SQuAD-explorer/
- GLUE: https://gluebenchmark.com/

**Papers Importants**:
- "Parameter-Efficient Transfer Learning for NLP" (Adapters)
- "LoRA: Low-Rank Adaptation of Large Language Models"
- "QLoRA: Efficient Finetuning of Quantized LLMs"

---

**Chapitre 7 terminé!** Vous maîtrisez maintenant le fine-tuning de LLMs de A à Z.

**Prochain chapitre**: Chapitre 8 - LoRA et Parameter-Efficient Fine-Tuning (PEFT) en profondeur.
