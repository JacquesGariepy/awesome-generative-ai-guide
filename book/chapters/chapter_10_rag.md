# Chapitre 10: Retrieval-Augmented Generation (RAG)

## Introduction

Le **Retrieval-Augmented Generation (RAG)** est devenu LA technique incontournable pour créer des applications LLM en production.

### Pourquoi RAG?

```python
"""
Problème: LLMs ont des limitations

1. Knowledge Cutoff
   - GPT-4: Training jusqu'à avril 2023
   - Ne connaît pas les événements récents
   - Informations obsolètes

2. Hallucinations
   - Génère des faits incorrects avec confiance
   - Invente des sources, citations, statistiques
   - Dangereux pour applications critiques

3. Domaines Spécifiques
   - Pas d'accès à vos documents internes
   - Ne connaît pas votre base de connaissances
   - Impossible de personnaliser pour votre entreprise

Solution: RAG = Retrieval + Generation
"""

# ❌ Sans RAG
prompt = "Quel est le chiffre d'affaires de notre entreprise en 2024?"
response = llm(prompt)
# → "Je ne peux pas accéder à des informations spécifiques sur votre entreprise."
# OU PIRE: invente un chiffre!

# ✅ Avec RAG
# 1. Retrieve documents pertinents de votre base de données
docs = retrieve("chiffre d'affaires 2024")
# → ["Rapport financier Q3 2024: CA de 150M€...", "Croissance de 25% YoY..."]

# 2. Augment le prompt avec les documents
prompt = f"""Contexte: {docs}

Question: Quel est le chiffre d'affaires de notre entreprise en 2024?"""

response = llm(prompt)
# → "Selon le rapport financier Q3 2024, le chiffre d'affaires est de 150M€,
#     en croissance de 25% par rapport à 2023."
```

### Architecture RAG

```python
"""
Pipeline RAG standard

┌─────────────┐
│   Query     │
│ "Question"  │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│  Embedding      │  Convert query → vector
│  Model          │  (e.g., BERT, E5, BGE)
└────────┬────────┘
         │
         ▼
┌──────────────────┐
│  Vector DB       │  Search similar documents
│  (FAISS, Chroma) │  Cosine similarity
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Reranking       │  Optional: reorder results
│  (Cross-encoder) │  Better relevance
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Prompt          │  Inject retrieved docs
│  Construction    │  + original query
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  LLM             │  Generate answer
│  (GPT-4, Llama)  │  Based on context
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Answer          │  Grounded in retrieved docs
│  + Citations     │  Reduced hallucinations
└──────────────────┘
"""

from typing import List, Dict, Tuple
from dataclasses import dataclass
from enum import Enum


class RAGComponent(Enum):
    """Composants du pipeline RAG"""
    EMBEDDING = "embedding"  # Query → Vector
    RETRIEVAL = "retrieval"  # Vector → Documents
    RERANKING = "reranking"  # Reorder by relevance
    GENERATION = "generation"  # Generate answer


@dataclass
class Document:
    """Un document dans la base de connaissances"""
    id: str
    content: str
    metadata: Dict = None
    embedding: List[float] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class RetrievalResult:
    """Résultat de retrieval"""
    document: Document
    score: float  # Similarity score
    rank: int  # Position in results


class RAGPipeline:
    """
    Pipeline RAG complet

    Simple mais production-ready
    """

    def __init__(
        self,
        embedding_model,
        vector_store,
        llm,
        top_k: int = 5
    ):
        """
        Args:
            embedding_model: Modèle d'embedding (encode text → vector)
            vector_store: Base vectorielle (FAISS, Chroma, etc.)
            llm: Large Language Model pour génération
            top_k: Nombre de documents à récupérer
        """
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.llm = llm
        self.top_k = top_k

    def embed_query(self, query: str) -> List[float]:
        """
        Convert query to vector

        Args:
            query: Text query

        Returns:
            Embedding vector
        """
        return self.embedding_model.encode(query)

    def retrieve(
        self,
        query: str,
        top_k: int = None
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant documents

        Args:
            query: Search query
            top_k: Number of results (override default)

        Returns:
            List of retrieved documents with scores
        """
        if top_k is None:
            top_k = self.top_k

        # Embed query
        query_vector = self.embed_query(query)

        # Search vector store
        results = self.vector_store.search(
            query_vector,
            k=top_k
        )

        # Format results
        retrieval_results = []
        for rank, (doc, score) in enumerate(results):
            retrieval_results.append(
                RetrievalResult(
                    document=doc,
                    score=score,
                    rank=rank
                )
            )

        return retrieval_results

    def construct_prompt(
        self,
        query: str,
        retrieved_docs: List[RetrievalResult]
    ) -> str:
        """
        Construct prompt with retrieved context

        Args:
            query: Original query
            retrieved_docs: Retrieved documents

        Returns:
            Prompt for LLM
        """
        # Build context from documents
        context_parts = []
        for result in retrieved_docs:
            doc = result.document
            context_parts.append(f"[Document {result.rank + 1}]\n{doc.content}")

        context = "\n\n".join(context_parts)

        # Construct prompt
        prompt = f"""Use the following context to answer the question. If the answer is not in the context, say "I don't have enough information to answer this question."

Context:
{context}

Question: {query}

Answer:"""

        return prompt

    def generate(
        self,
        query: str,
        retrieved_docs: List[RetrievalResult]
    ) -> str:
        """
        Generate answer using LLM

        Args:
            query: Original query
            retrieved_docs: Retrieved documents

        Returns:
            Generated answer
        """
        # Construct prompt
        prompt = self.construct_prompt(query, retrieved_docs)

        # Generate
        answer = self.llm.generate(prompt)

        return answer

    def query(
        self,
        query: str,
        return_sources: bool = True
    ) -> Dict:
        """
        Complete RAG pipeline

        Args:
            query: User query
            return_sources: Include source documents in response

        Returns:
            Dict with answer and optionally sources
        """
        # Step 1: Retrieve
        retrieved_docs = self.retrieve(query)

        # Step 2: Generate
        answer = self.generate(query, retrieved_docs)

        # Format response
        response = {
            "query": query,
            "answer": answer
        }

        if return_sources:
            response["sources"] = [
                {
                    "content": result.document.content,
                    "score": result.score,
                    "metadata": result.document.metadata
                }
                for result in retrieved_docs
            ]

        return response


# Exemple d'utilisation
if __name__ == "__main__":
    print("="*60)
    print("RAG PIPELINE - ARCHITECTURE")
    print("="*60)
    print()

    print("Components:")
    for component in RAGComponent:
        print(f"  • {component.value}")
    print()

    print("Flow:")
    print("  1. Query → Embedding")
    print("  2. Embedding → Vector Search")
    print("  3. Retrieved Docs → Prompt Construction")
    print("  4. Prompt → LLM → Answer")
    print()

    print("="*60)
    print()

    # Pseudo-code example
    print("Example usage (pseudo-code):")
    print("""
# Initialize
rag = RAGPipeline(
    embedding_model=SentenceTransformer('all-MiniLM-L6-v2'),
    vector_store=FAISSVectorStore(dimension=384),
    llm=OpenAI('gpt-4'),
    top_k=5
)

# Query
response = rag.query("What is the capital of France?")

print(response['answer'])
# → "The capital of France is Paris."

print(response['sources'])
# → [Document 1: "Paris is the capital...", ...]
    """)
```

## 1. Embeddings et Similarité

```python
"""
Embeddings: Représenter le texte comme vecteur numérique

Principe: Des textes similaires ont des vecteurs proches
Distance: Cosine similarity, Euclidean distance, dot product
"""

import numpy as np
from typing import List
import torch
import torch.nn.functional as F


class EmbeddingExplainer:
    """Explique les embeddings et similarité"""

    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Cosine similarity entre 2 vecteurs

        Formula: cos(θ) = (A · B) / (||A|| × ||B||)

        Range: [-1, 1]
        - 1: Identiques (même direction)
        - 0: Orthogonaux (pas de relation)
        - -1: Opposés

        Args:
            vec1, vec2: Vectors

        Returns:
            Similarity score
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        return dot_product / (norm1 * norm2)

    @staticmethod
    def euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Distance euclidienne entre 2 vecteurs

        Formula: sqrt(Σ(a_i - b_i)²)

        Range: [0, ∞]
        - 0: Identiques
        - Plus grand: Plus différents

        Args:
            vec1, vec2: Vectors

        Returns:
            Distance
        """
        return np.linalg.norm(vec1 - vec2)

    @staticmethod
    def dot_product_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Dot product (produit scalaire)

        Formula: A · B = Σ(a_i × b_i)

        Range: [-∞, ∞]
        - Plus grand: Plus similaires

        Note: Favorise vecteurs de grande magnitude

        Args:
            vec1, vec2: Vectors

        Returns:
            Dot product
        """
        return np.dot(vec1, vec2)

    @staticmethod
    def explain_embedding_models():
        """Explique différents modèles d'embedding"""

        return {
            "Sentence-BERT (SBERT)": {
                "description": "BERT fine-tuné pour sentence embeddings",
                "dimension": "384 ou 768",
                "performance": "⭐⭐⭐⭐ Excellent",
                "speed": "Rapide",
                "use_cases": "Usage général, baseline",
                "models": [
                    "all-MiniLM-L6-v2 (384d, rapide)",
                    "all-mpnet-base-v2 (768d, meilleur)"
                ]
            },

            "E5": {
                "description": "Text Embeddings by Weakly-Supervised Contrastive Pre-training",
                "dimension": "768 ou 1024",
                "performance": "⭐⭐⭐⭐⭐ SOTA",
                "speed": "Moyen",
                "use_cases": "Production, meilleure qualité",
                "models": [
                    "intfloat/e5-small-v2 (384d)",
                    "intfloat/e5-base-v2 (768d)",
                    "intfloat/e5-large-v2 (1024d)"
                ]
            },

            "BGE (BAAI General Embedding)": {
                "description": "Chinese Academy of Sciences, SOTA",
                "dimension": "768 ou 1024",
                "performance": "⭐⭐⭐⭐⭐ SOTA",
                "speed": "Moyen",
                "use_cases": "Production, multilingual",
                "models": [
                    "BAAI/bge-small-en-v1.5 (384d)",
                    "BAAI/bge-base-en-v1.5 (768d)",
                    "BAAI/bge-large-en-v1.5 (1024d)"
                ]
            },

            "OpenAI Ada-002": {
                "description": "OpenAI embedding API",
                "dimension": "1536",
                "performance": "⭐⭐⭐⭐⭐ Excellent",
                "speed": "API (latence réseau)",
                "use_cases": "Simplicité, pas d'infrastructure",
                "cost": "$0.0001 / 1K tokens"
            },

            "Cohere Embed": {
                "description": "Cohere embedding API",
                "dimension": "768 ou 1024",
                "performance": "⭐⭐⭐⭐⭐ Excellent",
                "speed": "API",
                "use_cases": "Multilingual, production",
                "cost": "$0.0001 / 1K tokens"
            }
        }


class SimpleEmbeddingModel:
    """
    Simple embedding model wrapper

    Wraps SentenceTransformers pour usage facile
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Args:
            model_name: HuggingFace model name
        """
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
        except ImportError:
            print("Install sentence-transformers: pip install sentence-transformers")
            raise

    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encode texts to embeddings

        Args:
            texts: List of texts
            batch_size: Batch size
            normalize: Normalize embeddings (for cosine similarity)

        Returns:
            Embeddings matrix (n_texts, dimension)
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=False
        )

        return embeddings

    def similarity(self, text1: str, text2: str) -> float:
        """
        Compute similarity between 2 texts

        Args:
            text1, text2: Texts to compare

        Returns:
            Cosine similarity
        """
        emb1, emb2 = self.encode([text1, text2])
        return EmbeddingExplainer.cosine_similarity(emb1, emb2)


# Exemple
if __name__ == "__main__":
    print("\n=== Embeddings et Similarité ===\n")

    # Demo cosine similarity
    vec1 = np.array([1.0, 2.0, 3.0])
    vec2 = np.array([2.0, 4.0, 6.0])  # 2x vec1
    vec3 = np.array([1.0, 0.0, 0.0])  # Orthogonal

    explainer = EmbeddingExplainer()

    print("Vecteur 1:", vec1)
    print("Vecteur 2:", vec2, "(2x vec1)")
    print("Vecteur 3:", vec3, "(orthogonal)")
    print()

    sim_12 = explainer.cosine_similarity(vec1, vec2)
    sim_13 = explainer.cosine_similarity(vec1, vec3)

    print(f"Cosine sim(vec1, vec2): {sim_12:.3f}  (identiques → 1.0)")
    print(f"Cosine sim(vec1, vec3): {sim_13:.3f}  (orthogonaux → ~0.0)")
    print()

    print("="*60 + "\n")

    # Modèles d'embedding
    print("=== Modèles d'Embedding Populaires ===\n")

    models = explainer.explain_embedding_models()

    for name, info in list(models.items())[:3]:
        print(f"### {name}")
        print(f"Description: {info['description']}")
        print(f"Dimension: {info['dimension']}")
        print(f"Performance: {info['performance']}")
        print(f"Use cases: {info['use_cases']}")
        print()
```

*[Suite avec Vector Stores et chunking dans la partie 2...]*
