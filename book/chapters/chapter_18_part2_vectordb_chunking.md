# Chapitre 18 (Partie 2): Vector Databases et Chunking Strategies

## 3. Vector Databases

```python
"""
Vector Databases = Stockage et recherche efficace d'embeddings

Problème sans vector DB:
  - 1M documents × 768 dimensions = 3GB en mémoire
  - Recherche linéaire: O(n) - trop lent!
  - Pas de persistance

Vector DB solutions:
  • Pinecone (cloud, managed)
  • Weaviate (open-source, GraphQL)
  • Chroma (embedded, simple)
  • Qdrant (Rust, performance)
  • Milvus (scalable, production)
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import chromadb
from chromadb.config import Settings


@dataclass
class Document:
    """Document with metadata"""
    id: str
    text: str
    metadata: Dict
    embedding: Optional[np.ndarray] = None


class VectorDBComparison:
    """
    Compare different vector databases
    """

    @staticmethod
    def print_comparison():
        """Print comparison table"""
        print("="*100)
        print("VECTOR DATABASE COMPARISON")
        print("="*100)

        databases = [
            {
                "name": "Pinecone",
                "type": "Cloud (managed)",
                "pricing": "Free tier + $70/month",
                "scalability": "Excellent (millions+)",
                "features": "Production-ready, auto-scaling",
                "best_for": "Production apps, no ops",
                "code": "from pinecone import Pinecone"
            },
            {
                "name": "Weaviate",
                "type": "Self-hosted / Cloud",
                "pricing": "Open-source + paid cloud",
                "scalability": "Very good",
                "features": "GraphQL, multi-tenancy, hybrid search",
                "best_for": "Complex queries, graph features",
                "code": "import weaviate"
            },
            {
                "name": "Chroma",
                "type": "Embedded / Server",
                "pricing": "Open-source (free)",
                "scalability": "Good (100k-1M docs)",
                "features": "Simple API, Python-first",
                "best_for": "Development, prototyping, small apps",
                "code": "import chromadb"
            },
            {
                "name": "Qdrant",
                "type": "Self-hosted / Cloud",
                "pricing": "Open-source + paid cloud",
                "scalability": "Excellent",
                "features": "Rust performance, filtering, payload",
                "best_for": "High performance, filtering",
                "code": "from qdrant_client import QdrantClient"
            },
            {
                "name": "Milvus",
                "type": "Self-hosted / Cloud",
                "pricing": "Open-source + Zilliz cloud",
                "scalability": "Excellent (billions)",
                "features": "Highly scalable, GPU support",
                "best_for": "Large scale, production",
                "code": "from pymilvus import connections"
            },
            {
                "name": "FAISS (Meta)",
                "type": "Library (not DB)",
                "pricing": "Open-source (free)",
                "scalability": "Very good",
                "features": "CPU/GPU, many index types",
                "best_for": "Research, custom solutions",
                "code": "import faiss"
            }
        ]

        for db in databases:
            print(f"\n{db['name']}")
            print(f"  Type: {db['type']}")
            print(f"  Pricing: {db['pricing']}")
            print(f"  Scalability: {db['scalability']}")
            print(f"  Features: {db['features']}")
            print(f"  Best for: {db['best_for']}")
            print(f"  Code: {db['code']}")


class ChromaDBExample:
    """
    Complete Chroma DB example
    """

    def __init__(self, persist_directory: str = "./chroma_db"):
        """Initialize Chroma DB"""
        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))

        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}  # Similarity metric
        )

    def add_documents(
        self,
        documents: List[Document],
        embeddings: Optional[List[np.ndarray]] = None
    ):
        """
        Add documents to collection

        Args:
            documents: List of documents
            embeddings: Pre-computed embeddings (optional, Chroma can compute)
        """
        if embeddings is None:
            # Chroma will compute embeddings with default model
            self.collection.add(
                ids=[doc.id for doc in documents],
                documents=[doc.text for doc in documents],
                metadatas=[doc.metadata for doc in documents]
            )
        else:
            # Use pre-computed embeddings
            self.collection.add(
                ids=[doc.id for doc in documents],
                documents=[doc.text for doc in documents],
                embeddings=embeddings,
                metadatas=[doc.metadata for doc in documents]
            )

        print(f"Added {len(documents)} documents to Chroma DB")

    def search(
        self,
        query: str,
        n_results: int = 5,
        where: Optional[Dict] = None
    ) -> Dict:
        """
        Search similar documents

        Args:
            query: Query text
            n_results: Number of results to return
            where: Metadata filter (e.g., {"source": "wikipedia"})

        Returns:
            Results with documents, distances, metadata
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where
        )

        return results

    def delete_collection(self):
        """Delete collection"""
        self.client.delete_collection(name="documents")
        print("Collection deleted")


class PineconeExample:
    """
    Pinecone example (production-grade)
    """

    def __init__(self, api_key: str, environment: str = "us-west1-gcp"):
        """
        Initialize Pinecone

        Args:
            api_key: Pinecone API key
            environment: Pinecone environment (region)
        """
        print("""
# Pinecone Setup

1. Sign up: https://www.pinecone.io/
2. Get API key
3. Install: pip install pinecone-client

Usage:
        """)

        print('''
from pinecone import Pinecone, ServerlessSpec

# Initialize
pc = Pinecone(api_key="YOUR_API_KEY")

# Create index
index_name = "rag-index"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=768,  # embedding dimension
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-west-2"
        )
    )

# Connect to index
index = pc.Index(index_name)

# Upsert vectors
vectors = [
    {
        "id": "doc1",
        "values": embedding1,  # 768-dim vector
        "metadata": {"text": "...", "source": "wikipedia"}
    },
    {
        "id": "doc2",
        "values": embedding2,
        "metadata": {"text": "...", "source": "arxiv"}
    }
]

index.upsert(vectors=vectors)

# Query
results = index.query(
    vector=query_embedding,
    top_k=5,
    include_metadata=True,
    filter={"source": "wikipedia"}  # Metadata filtering
)

for match in results['matches']:
    print(f"Score: {match['score']:.4f}")
    print(f"Text: {match['metadata']['text']}")

# Stats
stats = index.describe_index_stats()
print(f"Total vectors: {stats['total_vector_count']}")
        ''')


# Demo
if __name__ == "__main__":
    print("="*80)
    print("VECTOR DATABASES FOR RAG")
    print("="*80)

    # Comparison
    VectorDBComparison.print_comparison()

    print("\n" + "="*80)
    print("CHROMA DB EXAMPLE")
    print("="*80)

    # Initialize Chroma
    chroma_db = ChromaDBExample(persist_directory="./demo_chroma")

    # Sample documents
    documents = [
        Document(
            id="1",
            text="Paris is the capital of France. It is known for the Eiffel Tower.",
            metadata={"source": "geography", "country": "France"}
        ),
        Document(
            id="2",
            text="Machine learning is a subset of artificial intelligence.",
            metadata={"source": "technology", "topic": "AI"}
        ),
        Document(
            id="3",
            text="The Eiffel Tower was built in 1889 for the World's Fair.",
            metadata={"source": "history", "country": "France"}
        )
    ]

    # Add documents (Chroma will compute embeddings)
    chroma_db.add_documents(documents)

    # Search
    query = "What is the capital of France?"
    results = chroma_db.search(query, n_results=2)

    print(f"\nQuery: '{query}'")
    print(f"\nTop results:")
    for i, (doc, distance) in enumerate(zip(
        results['documents'][0],
        results['distances'][0]
    )):
        print(f"\n{i+1}. (distance: {distance:.4f})")
        print(f"   {doc}")

    # Filter search
    results_france = chroma_db.search(
        query,
        n_results=2,
        where={"country": "France"}
    )

    print(f"\nFiltered search (country=France):")
    print(f"Found {len(results_france['documents'][0])} results")

    # Pinecone example
    print("\n" + "="*80)
    print("PINECONE EXAMPLE")
    print("="*80)

    pinecone_example = PineconeExample(api_key="demo")
```

## 4. Chunking Strategies

```python
"""
Chunking = Découper documents en morceaux pour RAG

Pourquoi chunker?
  • LLMs ont context windows limités (4k-200k tokens)
  • Embeddings plus précis sur petits chunks
  • Retrieval plus granulaire

Trade-offs:
  • Chunks trop petits: Perd contexte
  • Chunks trop grands: Retrieval moins précis

Stratégies:
  1. Fixed size (simple)
  2. Sentence-based (semantic)
  3. Paragraph-based (structure)
  4. Recursive (hierarchical)
  5. Semantic (embeddings)
"""

from typing import List, Tuple
import re
from dataclasses import dataclass


@dataclass
class Chunk:
    """Text chunk with metadata"""
    text: str
    start_idx: int
    end_idx: int
    chunk_id: int
    metadata: dict


class FixedSizeChunker:
    """
    Chunk by fixed token/character count
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        by_tokens: bool = True
    ):
        """
        Args:
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks (for context continuity)
            by_tokens: Chunk by tokens (True) or characters (False)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.by_tokens = by_tokens

        if by_tokens:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")

    def chunk(self, text: str) -> List[Chunk]:
        """
        Chunk text into fixed-size pieces

        Args:
            text: Input text

        Returns:
            List of chunks
        """
        if self.by_tokens:
            return self._chunk_by_tokens(text)
        else:
            return self._chunk_by_chars(text)

    def _chunk_by_tokens(self, text: str) -> List[Chunk]:
        """Chunk by token count"""
        # Tokenize
        tokens = self.tokenizer.encode(text)

        chunks = []
        chunk_id = 0

        for i in range(0, len(tokens), self.chunk_size - self.chunk_overlap):
            chunk_tokens = tokens[i:i + self.chunk_size]

            # Decode
            chunk_text = self.tokenizer.decode(chunk_tokens)

            chunks.append(Chunk(
                text=chunk_text,
                start_idx=i,
                end_idx=min(i + self.chunk_size, len(tokens)),
                chunk_id=chunk_id,
                metadata={"tokens": len(chunk_tokens)}
            ))

            chunk_id += 1

        return chunks

    def _chunk_by_chars(self, text: str) -> List[Chunk]:
        """Chunk by character count"""
        chunks = []
        chunk_id = 0

        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk_text = text[i:i + self.chunk_size]

            chunks.append(Chunk(
                text=chunk_text,
                start_idx=i,
                end_idx=min(i + self.chunk_size, len(text)),
                chunk_id=chunk_id,
                metadata={"chars": len(chunk_text)}
            ))

            chunk_id += 1

        return chunks


class SemanticChunker:
    """
    Chunk by sentences (semantic boundaries)
    """

    def __init__(
        self,
        max_chunk_size: int = 512,
        sentence_splitter: str = "nltk"
    ):
        """
        Args:
            max_chunk_size: Maximum tokens per chunk
            sentence_splitter: 'nltk' or 'regex'
        """
        self.max_chunk_size = max_chunk_size
        self.sentence_splitter = sentence_splitter

        if sentence_splitter == "nltk":
            import nltk
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt')
            from nltk.tokenize import sent_tokenize
            self.sent_tokenize = sent_tokenize
        else:
            self.sent_tokenize = self._regex_sentence_split

        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")

    def _regex_sentence_split(self, text: str) -> List[str]:
        """Split by sentence using regex"""
        # Split on . ! ? followed by space or end
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def chunk(self, text: str) -> List[Chunk]:
        """
        Chunk by sentences, respecting max size

        Args:
            text: Input text

        Returns:
            List of chunks
        """
        # Split into sentences
        sentences = self.sent_tokenize(text)

        chunks = []
        current_chunk = []
        current_tokens = 0
        chunk_id = 0
        start_idx = 0

        for sentence in sentences:
            # Count tokens
            sentence_tokens = len(self.tokenizer.encode(sentence))

            if current_tokens + sentence_tokens > self.max_chunk_size and current_chunk:
                # Save current chunk
                chunk_text = " ".join(current_chunk)
                chunks.append(Chunk(
                    text=chunk_text,
                    start_idx=start_idx,
                    end_idx=start_idx + len(chunk_text),
                    chunk_id=chunk_id,
                    metadata={"sentences": len(current_chunk), "tokens": current_tokens}
                ))

                # Start new chunk
                chunk_id += 1
                start_idx += len(chunk_text) + 1
                current_chunk = [sentence]
                current_tokens = sentence_tokens
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens

        # Add last chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(Chunk(
                text=chunk_text,
                start_idx=start_idx,
                end_idx=start_idx + len(chunk_text),
                chunk_id=chunk_id,
                metadata={"sentences": len(current_chunk), "tokens": current_tokens}
            ))

        return chunks


class RecursiveChunker:
    """
    Recursive chunking with hierarchy

    Tries to split by:
      1. Paragraphs (\n\n)
      2. Sentences
      3. Words
      4. Characters (last resort)
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")

        # Separators (in priority order)
        self.separators = [
            "\n\n",  # Paragraphs
            "\n",    # Lines
            ". ",    # Sentences
            "! ",
            "? ",
            " ",     # Words
            ""       # Characters
        ]

    def chunk(self, text: str) -> List[Chunk]:
        """
        Recursively chunk text

        Args:
            text: Input text

        Returns:
            List of chunks
        """
        return self._recursive_split(text, self.separators)

    def _recursive_split(
        self,
        text: str,
        separators: List[str]
    ) -> List[Chunk]:
        """Recursive splitting logic"""
        if not separators:
            # Base case: split by characters
            return self._split_by_size(text)

        # Try first separator
        separator = separators[0]
        remaining_separators = separators[1:]

        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)

        # Process splits
        chunks = []
        chunk_id = 0
        start_idx = 0

        for split in splits:
            # Check if split is small enough
            tokens = len(self.tokenizer.encode(split))

            if tokens <= self.chunk_size:
                chunks.append(Chunk(
                    text=split,
                    start_idx=start_idx,
                    end_idx=start_idx + len(split),
                    chunk_id=chunk_id,
                    metadata={"tokens": tokens, "separator": separator}
                ))
                chunk_id += 1
            else:
                # Too large, split recursively
                sub_chunks = self._recursive_split(split, remaining_separators)
                chunks.extend(sub_chunks)

            start_idx += len(split) + len(separator)

        return chunks

    def _split_by_size(self, text: str) -> List[Chunk]:
        """Split by fixed size (fallback)"""
        chunks = []
        for i in range(0, len(text), self.chunk_size):
            chunk_text = text[i:i + self.chunk_size]
            tokens = len(self.tokenizer.encode(chunk_text))

            chunks.append(Chunk(
                text=chunk_text,
                start_idx=i,
                end_idx=i + len(chunk_text),
                chunk_id=i // self.chunk_size,
                metadata={"tokens": tokens, "method": "fixed_size"}
            ))

        return chunks


# Demo
if __name__ == "__main__":
    print("="*80)
    print("CHUNKING STRATEGIES")
    print("="*80)

    # Sample text
    sample_text = """
    Paris is the capital and largest city of France. It is located on the Seine River,
    in northern France, at the heart of the Île-de-France region. The city is known
    for its art, fashion, gastronomy, and culture.

    The Eiffel Tower is one of the most iconic landmarks in Paris. It was built in 1889
    for the World's Fair and has become a symbol of France. The tower is 330 meters tall
    and attracts millions of visitors every year.

    Machine learning is a subset of artificial intelligence that focuses on building
    systems that can learn from data. Deep learning, a subfield of machine learning,
    uses neural networks with multiple layers to model complex patterns.
    """

    # Test 1: Fixed size chunking
    print("\n--- Fixed Size Chunking ---")
    fixed_chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=20, by_tokens=True)
    fixed_chunks = fixed_chunker.chunk(sample_text)

    print(f"Generated {len(fixed_chunks)} chunks")
    for i, chunk in enumerate(fixed_chunks[:3]):  # Show first 3
        print(f"\nChunk {i+1} ({chunk.metadata['tokens']} tokens):")
        print(f"  {chunk.text[:100]}...")

    # Test 2: Semantic chunking
    print("\n--- Semantic Chunking ---")
    semantic_chunker = SemanticChunker(max_chunk_size=150)
    semantic_chunks = semantic_chunker.chunk(sample_text)

    print(f"Generated {len(semantic_chunks)} chunks")
    for i, chunk in enumerate(semantic_chunks):
        print(f"\nChunk {i+1} ({chunk.metadata['sentences']} sentences, {chunk.metadata['tokens']} tokens):")
        print(f"  {chunk.text[:100]}...")

    # Test 3: Recursive chunking
    print("\n--- Recursive Chunking ---")
    recursive_chunker = RecursiveChunker(chunk_size=100, chunk_overlap=20)
    recursive_chunks = recursive_chunker.chunk(sample_text)

    print(f"Generated {len(recursive_chunks)} chunks")

    print("\n" + "="*80)
    print("CHUNKING RECOMMENDATIONS")
    print("="*80)
    print("""
1. General purpose: Recursive chunking
   • Respects document structure
   • Good balance

2. Technical docs: Semantic chunking
   • Preserves complete thoughts
   • Better for Q&A

3. Long documents: Fixed size with overlap
   • Predictable chunk sizes
   • Good for embeddings

4. Code: AST-based chunking
   • Respect function/class boundaries
   • Not covered here (specialized)

Typical parameters:
  • chunk_size: 256-512 tokens
  • overlap: 10-20% of chunk_size
  • Experiment with your data!
    """)
```

*[Suite avec Reranking et projet RAG complet dans la partie 3...]*
