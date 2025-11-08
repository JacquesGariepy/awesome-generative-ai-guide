# Chapitre 21 - Partie 3: Évaluation, Retrieval et Projet Long Context

## Évaluation du Long Context

Comment mesurer si un modèle utilise efficacement son long contexte?

```python
"""
ÉVALUATION DU LONG CONTEXT

Benchmarks principaux:

1. NEEDLE IN A HAYSTACK:
   • Cacher une information dans un long contexte
   • Poser une question dessus
   • Tester: Position (début, milieu, fin) × Longueur contexte

   Exemple:
     Context: 100k tokens de texte aléatoire
     Needle: "Le code secret est 7391"
     Question: "Quel est le code secret?"

   Métrique: Accuracy selon position et longueur

2. LOST IN THE MIDDLE (Liu et al. 2023):
   • Observation: Modèles utilisent mieux début/fin que milieu
   • "U-shaped curve": Performance chute au milieu
   • Important pour retrieval (où placer documents?)

3. LONGBENCH (Bai et al. 2023):
   • 21 tâches, 6 langues
   • QA, summarization, code, etc.
   • Contextes: 5k-15k tokens moyenne

4. L-EVAL (An et al. 2023):
   • Tâches nécessitant raisonnement long
   • "Closed-book" (pas de retrieval externe)

Résultats (2024):
  GPT-4 (128k): ~95% needle in haystack
  Claude 3 (200k): ~98% (meilleur)
  Gemini 1.5 (1M): ~93% (bon malgré 1M context!)
  Open-source: 60-85% selon modèle
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import random


# ============================================================================
# NEEDLE IN A HAYSTACK BENCHMARK
# ============================================================================

@dataclass
class NeedleTestResult:
    """Résultat d'un test needle in haystack"""
    context_length: int
    needle_position: int      # Position relative (0-1)
    found: bool               # L'info a été trouvée?
    confidence: float         # Score de confiance
    latency_ms: float        # Temps de réponse


class NeedleInHaystackBenchmark:
    """
    Benchmark "Needle in a Haystack"

    Test si le modèle peut retrouver une info spécifique
    cachée dans un très long contexte

    Méthodologie:
      1. Générer contexte distractor (texte non pertinent)
      2. Insérer "needle" (info cible) à position donnée
      3. Poser question sur le needle
      4. Vérifier si réponse correcte

    Variables testées:
      • Longueur contexte: 1k, 2k, 4k, ..., 128k
      • Position needle: 0%, 25%, 50%, 75%, 100%
    """

    def __init__(self, model: Any):
        self.model = model

    def run_test(
        self,
        context_length: int,
        needle_position: float  # 0.0 à 1.0
    ) -> NeedleTestResult:
        """
        Execute un test

        Args:
            context_length: Longueur totale du contexte (tokens)
            needle_position: Position relative du needle (0=début, 1=fin)

        Returns:
            Résultat du test
        """
        # 1. Générer needle (info à cacher)
        secret_code = random.randint(1000, 9999)
        needle = f"Le code secret pour la mission est {secret_code}. Mémorisez-le."

        # 2. Générer contexte distractor
        distractor = self._generate_distractor(context_length - len(needle.split()))

        # 3. Insérer needle à la position spécifiée
        insert_pos = int(len(distractor) * needle_position)
        context = (
            " ".join(distractor[:insert_pos]) +
            " " + needle + " " +
            " ".join(distractor[insert_pos:])
        )

        # 4. Poser question
        question = "Quel est le code secret pour la mission?"

        # 5. Obtenir réponse du modèle
        # En production: appel au modèle
        # response = self.model.generate(context + "\n\n" + question)

        # Simulation
        response, confidence, latency = self._simulate_model_response(
            secret_code, needle_position, context_length
        )

        # 6. Vérifier si correct
        found = str(secret_code) in response

        return NeedleTestResult(
            context_length=context_length,
            needle_position=needle_position,
            found=found,
            confidence=confidence,
            latency_ms=latency
        )

    def _generate_distractor(self, num_tokens: int) -> List[str]:
        """
        Génère texte distractor (non pertinent)

        En vrai benchmark: utiliser textes du web, livres, etc.
        """
        # Texte répétitif pour simulation
        base_text = """
        Les systèmes de machine learning continuent d'évoluer rapidement.
        Les transformers ont révolutionné le traitement du langage naturel.
        L'attention est un mécanisme clé dans les modèles modernes.
        """.split()

        tokens = []
        while len(tokens) < num_tokens:
            tokens.extend(base_text)

        return tokens[:num_tokens]

    def _simulate_model_response(
        self,
        secret_code: int,
        needle_position: float,
        context_length: int
    ) -> Tuple[str, float, float]:
        """
        Simule réponse du modèle

        Hypothèses réalistes basées sur recherche:
          • Début/fin: ~95% accuracy
          • Milieu (0.4-0.6): ~70% accuracy (lost in middle)
          • Plus long = plus difficile
        """
        # Score de base selon position (U-curve)
        if needle_position < 0.2:  # Début
            base_score = 0.95
        elif needle_position > 0.8:  # Fin
            base_score = 0.92
        elif 0.4 <= needle_position <= 0.6:  # Milieu
            base_score = 0.70
        else:  # Intermédiaire
            base_score = 0.85

        # Pénalité pour longueur
        length_penalty = 0.1 * (context_length / 100000)
        final_score = max(0.5, base_score - length_penalty)

        # Déterminer si trouvé
        found = random.random() < final_score

        if found:
            response = f"Le code secret est {secret_code}."
        else:
            response = "Je ne trouve pas l'information sur le code secret."

        # Latence: augmente avec longueur
        latency = 100 + (context_length / 1000) * 50  # ms

        return response, final_score, latency

    def run_full_benchmark(
        self,
        context_lengths: List[int] = [1000, 2000, 4000, 8000, 16000, 32000],
        positions: List[float] = [0.0, 0.25, 0.5, 0.75, 1.0],
        trials_per_config: int = 3
    ) -> Dict[str, Any]:
        """
        Execute benchmark complet

        Args:
            context_lengths: Longueurs à tester
            positions: Positions relatives à tester
            trials_per_config: Nombre d'essais par config

        Returns:
            Résultats agrégés
        """
        print("\n" + "="*80)
        print("NEEDLE IN A HAYSTACK - BENCHMARK COMPLET")
        print("="*80)
        print(f"Context lengths: {context_lengths}")
        print(f"Positions: {positions}")
        print(f"Trials per config: {trials_per_config}")

        results = []
        total_tests = len(context_lengths) * len(positions) * trials_per_config

        test_num = 0

        for ctx_len in context_lengths:
            for pos in positions:
                for trial in range(trials_per_config):
                    test_num += 1
                    print(f"\rTest {test_num}/{total_tests}...", end="")

                    result = self.run_test(ctx_len, pos)
                    results.append(result)

        print("\n\n✅ Benchmark complété!")

        # Analyser résultats
        analysis = self._analyze_results(results)

        return {
            "results": results,
            "analysis": analysis
        }

    def _analyze_results(self, results: List[NeedleTestResult]) -> Dict:
        """Analyse les résultats"""
        # Accuracy globale
        total = len(results)
        found = sum(1 for r in results if r.found)
        accuracy = found / total

        # Accuracy par position
        positions = [0.0, 0.25, 0.5, 0.75, 1.0]
        accuracy_by_pos = {}

        for pos in positions:
            pos_results = [r for r in results if abs(r.needle_position - pos) < 0.01]
            if pos_results:
                pos_found = sum(1 for r in pos_results if r.found)
                accuracy_by_pos[pos] = pos_found / len(pos_results)

        # Accuracy par longueur
        lengths = sorted(set(r.context_length for r in results))
        accuracy_by_length = {}

        for length in lengths:
            len_results = [r for r in results if r.context_length == length]
            len_found = sum(1 for r in len_results if r.found)
            accuracy_by_length[length] = len_found / len(len_results)

        # Latence moyenne
        avg_latency = np.mean([r.latency_ms for r in results])

        return {
            "overall_accuracy": accuracy,
            "accuracy_by_position": accuracy_by_pos,
            "accuracy_by_length": accuracy_by_length,
            "avg_latency_ms": avg_latency
        }

    def visualize_results(self, analysis: Dict):
        """Affiche résultats du benchmark"""
        print("\n" + "="*80)
        print("RÉSULTATS DU BENCHMARK")
        print("="*80)

        print(f"\n📊 Accuracy globale: {analysis['overall_accuracy']:.1%}")

        print(f"\n📍 Accuracy par position (U-curve expected):")
        for pos, acc in sorted(analysis['accuracy_by_position'].items()):
            bar = "█" * int(acc * 50)
            print(f"  {pos:.2f}: {bar} {acc:.1%}")

        print(f"\n📏 Accuracy par longueur:")
        for length, acc in sorted(analysis['accuracy_by_length'].items()):
            bar = "█" * int(acc * 50)
            print(f"  {length:6,} tokens: {bar} {acc:.1%}")

        print(f"\n⏱️  Latence moyenne: {analysis['avg_latency_ms']:.0f}ms")


# ============================================================================
# RETRIEVAL FROM LONG CONTEXT
# ============================================================================

"""
RETRIEVAL FROM LONG CONTEXT

Problème: Même avec 1M tokens, comment trouver info pertinente?

Approches:

1. FULL CONTEXT (brute force):
   • Mettre tout le contexte dans le prompt
   • Laisser le modèle chercher
   • ❌ Coûteux, lent, "lost in middle"

2. SMART CHUNKING + RERANKING:
   • Diviser contexte en chunks
   • Embed chaque chunk
   • Retrieve top-k chunks pertinents
   • Rerank avec LLM
   • ✅ Plus efficace

3. HIERARCHICAL RETRIEVAL:
   • Niveau 1: Retrieve sections larges
   • Niveau 2: Retrieve paragraphes dans sections
   • Niveau 3: Retrieve phrases exactes
   • ✅ Précis et efficace

4. ATTENTION SINKS (Xiao et al. 2023):
   • Garder premiers tokens (attention sinks)
   • + fenêtre glissante sur récent
   • ✅ Streaming infini possible
"""

class LongContextRetriever:
    """
    Retrieval efficace depuis long contexte

    Stratégie: Chunking + Embedding + Top-K + Rerank
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        top_k: int = 5
    ):
        """
        Args:
            chunk_size: Taille des chunks (tokens)
            chunk_overlap: Overlap entre chunks
            top_k: Nombre de chunks à retrieve
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k

    def chunk_context(self, context: str) -> List[Dict[str, Any]]:
        """
        Divise contexte en chunks avec overlap

        Args:
            context: Texte complet

        Returns:
            Liste de chunks avec métadonnées
        """
        tokens = context.split()  # Simplification
        chunks = []

        stride = self.chunk_size - self.chunk_overlap
        start = 0

        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))

            chunk_text = " ".join(tokens[start:end])

            chunks.append({
                "text": chunk_text,
                "start_pos": start,
                "end_pos": end,
                "chunk_id": len(chunks)
            })

            start += stride

            if end >= len(tokens):
                break

        return chunks

    def retrieve(
        self,
        query: str,
        context: str
    ) -> List[Dict[str, Any]]:
        """
        Retrieve chunks pertinents

        Args:
            query: Question/requête
            context: Contexte complet

        Returns:
            Top-K chunks les plus pertinents
        """
        print(f"\n🔍 Retrieval from long context")
        print(f"  Query: {query}")
        print(f"  Context: {len(context.split())} tokens")

        # 1. Chunking
        chunks = self.chunk_context(context)
        print(f"  Chunks créés: {len(chunks)}")

        # 2. Embed query et chunks
        # En production: utiliser modèle d'embedding
        # query_emb = embed_model.encode(query)
        # chunk_embs = embed_model.encode([c["text"] for c in chunks])

        # Simulation: scores aléatoires
        for chunk in chunks:
            chunk["score"] = random.random()

        # 3. Retrieve top-K
        top_chunks = sorted(chunks, key=lambda x: x["score"], reverse=True)[:self.top_k]

        print(f"\n📊 Top {self.top_k} chunks retrieved:")
        for i, chunk in enumerate(top_chunks, 1):
            print(f"  {i}. Chunk #{chunk['chunk_id']} (score: {chunk['score']:.3f})")
            print(f"     Position: {chunk['start_pos']}-{chunk['end_pos']}")
            print(f"     Text: {chunk['text'][:100]}...")

        return top_chunks


# ============================================================================
# PROJET COMPLET: SYSTÈME DE GESTION LONG CONTEXT
# ============================================================================

class LongContextManager:
    """
    Système complet de gestion de long contexte

    Fonctionnalités:
      • Compression automatique
      • Retrieval efficace
      • Caching intelligent
      • Monitoring de l'utilisation
    """

    def __init__(
        self,
        max_context: int = 128000,
        compression_threshold: int = 100000
    ):
        self.max_context = max_context
        self.compression_threshold = compression_threshold

        # Historique conversationnel
        self.full_history: List[Dict] = []
        self.compressed_history: List[Dict] = []

        # Retriever
        self.retriever = LongContextRetriever()

        # Stats
        self.stats = {
            "total_tokens": 0,
            "compressions": 0,
            "retrievals": 0
        }

    def add_message(
        self,
        role: str,
        content: str
    ):
        """
        Ajoute message à l'historique

        Args:
            role: "user" ou "assistant"
            content: Contenu du message
        """
        message = {
            "role": role,
            "content": content,
            "tokens": len(content.split())  # Simplification
        }

        self.full_history.append(message)
        self.stats["total_tokens"] += message["tokens"]

        # Vérifier si compression nécessaire
        if self.stats["total_tokens"] > self.compression_threshold:
            self._compress_history()

    def _compress_history(self):
        """Compresse l'historique ancien"""
        print(f"\n🗜️  Compression de l'historique")
        print(f"  Tokens avant: {self.stats['total_tokens']:,}")

        # Garder messages récents
        recent_tokens = 0
        recent_messages = []

        for msg in reversed(self.full_history):
            if recent_tokens + msg["tokens"] <= self.max_context // 2:
                recent_messages.insert(0, msg)
                recent_tokens += msg["tokens"]
            else:
                break

        # Résumer anciens messages
        old_messages = self.full_history[:-len(recent_messages)]

        if old_messages:
            # En production: LLM summarize
            summary_text = f"[Résumé des {len(old_messages)} messages précédents: ...]"

            self.compressed_history.append({
                "role": "system",
                "content": summary_text,
                "tokens": len(summary_text.split()),
                "is_summary": True
            })

        # Mettre à jour historique
        self.full_history = self.compressed_history + recent_messages
        self.stats["total_tokens"] = sum(m["tokens"] for m in self.full_history)
        self.stats["compressions"] += 1

        print(f"  Tokens après: {self.stats['total_tokens']:,}")
        print(f"  Ratio: {self.stats['total_tokens'] / (self.stats['total_tokens'] + len(old_messages) * 100):.1%}")

    def query_context(self, query: str) -> str:
        """
        Query l'historique avec retrieval

        Args:
            query: Question

        Returns:
            Réponse basée sur contexte pertinent
        """
        # Construire contexte complet
        full_context = "\n\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in self.full_history
        ])

        # Retrieve chunks pertinents
        relevant_chunks = self.retriever.retrieve(query, full_context)

        # Construire prompt avec chunks pertinents
        context_for_prompt = "\n\n".join([
            chunk["text"] for chunk in relevant_chunks
        ])

        prompt = f"""Contexte pertinent:
{context_for_prompt}

Question: {query}

Réponse:"""

        # En production: appel LLM
        response = "[Réponse basée sur retrieval...]"

        self.stats["retrievals"] += 1

        return response

    def get_stats(self) -> Dict:
        """Retourne statistiques d'utilisation"""
        return {
            "total_messages": len(self.full_history),
            "total_tokens": self.stats["total_tokens"],
            "compressions": self.stats["compressions"],
            "retrievals": self.stats["retrievals"],
            "context_utilization": self.stats["total_tokens"] / self.max_context
        }


# ============================================================================
# DÉMONSTRATIONS
# ============================================================================

def demo_needle_benchmark():
    """Démo needle in haystack"""
    print("="*80)
    print("DÉMONSTRATION: NEEDLE IN A HAYSTACK")
    print("="*80)

    benchmark = NeedleInHaystackBenchmark(model=None)

    # Benchmark complet
    results = benchmark.run_full_benchmark(
        context_lengths=[2000, 4000, 8000, 16000],
        positions=[0.0, 0.25, 0.5, 0.75, 1.0],
        trials_per_config=3
    )

    # Visualiser
    benchmark.visualize_results(results["analysis"])


def demo_long_context_manager():
    """Démo du gestionnaire de long context"""
    print("\n\n" + "="*80)
    print("DÉMONSTRATION: LONG CONTEXT MANAGER")
    print("="*80)

    manager = LongContextManager(
        max_context=128000,
        compression_threshold=100000
    )

    # Simuler conversation longue
    print("\n📝 Simulation conversation longue...")

    for i in range(50):
        # User message
        user_msg = f"Message utilisateur #{i}: " + " ".join(["token"] * 500)
        manager.add_message("user", user_msg)

        # Assistant response
        asst_msg = f"Réponse assistant #{i}: " + " ".join(["token"] * 500)
        manager.add_message("assistant", asst_msg)

    # Stats
    print("\n📊 Statistiques:")
    stats = manager.get_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2%}" if value < 1 else f"  {key}: {value:,.0f}")
        else:
            print(f"  {key}: {value:,}")

    # Query
    print("\n🔍 Query du contexte:")
    response = manager.query_context("Que s'est-il passé au message #25?")


if __name__ == "__main__":
    # Demos
    demo_needle_benchmark()
    demo_long_context_manager()

    print("\n\n" + "="*80)
    print("🎉 CHAPITRE 21 COMPLÉTÉ: LONG CONTEXT")
    print("="*80)
    print("""
RÉCAPITULATIF COMPLET:

PARTIE 1: Position Encodings
  • Sinusoidal: Original, extrapolation mauvaise
  • RoPE: Relative positions, extensible
  • ALiBi: Bias linéaire, extrapolation parfaite
  • Extensions: Position Interpolation, NTK-aware

PARTIE 2: Optimisations
  • Flash Attention: 2-4x rapide, 10-20x moins mémoire
  • Sliding Window: O(n×W) au lieu de O(n²)
  • GQA/MQA: Réduction KV cache 4-32x
  • Compression: Résumé automatique du contexte

PARTIE 3: Évaluation et Projet
  • Needle in Haystack: Benchmark standard
  • Lost in Middle: U-curve performance
  • Retrieval: Chunking + Embedding + Rerank
  • LongContextManager: Système complet

CAPACITÉS ACTUELLES (2024):
  ✅ GPT-4: 128k tokens
  ✅ Claude 3: 200k tokens
  ✅ Gemini 1.5: 1M tokens
  ✅ Open-source: 32k-100k (LLaMA, Mistral, Qwen)

APPLICATIONS PRATIQUES:
  • Analyse de livres/documents complets
  • Codebase entière dans contexte
  • Conversations très longues
  • Analyse vidéo/audio longue
  • Documents légaux/médicaux

BEST PRACTICES:
  ✅ Utiliser Flash Attention si possible
  ✅ GQA pour réduire KV cache
  ✅ Retrieval pour > 100k tokens
  ✅ Monitor "lost in middle"
  ✅ Compresser historique ancien
  ✅ Cacher résultats fréquents

LIMITATIONS:
  ❌ Coût linéaire avec longueur
  ❌ Latence augmente
  ❌ Qualité variable selon position
  ❌ Nécessite GPU puissant

FUTUR:
  • Modèles 10M+ tokens
  • Attention O(n) vraie
  • Retrieval-augmented natif
  • Compression apprise

PROCHAINS CHAPITRES:
  • Ch.22: Chain-of-Thought et Reasoning
  • Ch.23-25: Projets pratiques complets
    """)
