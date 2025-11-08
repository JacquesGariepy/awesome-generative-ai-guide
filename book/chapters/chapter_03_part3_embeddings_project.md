# Chapitre 3 (Partie 3): Token Embeddings et Projet Complet

## 5. Token Embeddings

```python
"""
Token Embeddings = Conversion tokens → vecteurs denses

Problème:
  Token IDs sont des entiers discrets: 0, 1, 2, 3, ...
  → Pas de notion de similarité
  → 5 n'est pas "plus proche" de 6 que de 100

Solution: Embeddings
  Chaque token → vecteur de dimension d (e.g., 768, 1024)
  → Vecteurs similaires pour tokens sémantiquement proches

Example:
  "king" → [0.2, -0.5, 0.8, ..., 0.1]  (768 dimensions)
  "queen" → [0.3, -0.4, 0.7, ..., 0.2]  (proche de king!)
  "car" → [-0.8, 0.9, -0.1, ..., 0.5]  (éloigné de king)

Implémentation:
  Simple lookup table (Embedding layer in PyTorch)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple
import math


class TokenEmbedding(nn.Module):
    """
    Token Embedding Layer

    Simple lookup table: token_id → embedding vector

    Args:
        vocab_size: Size of vocabulary
        d_model: Embedding dimension
        padding_idx: Index of padding token (optional)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        padding_idx: Optional[int] = None
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model

        # Embedding lookup table
        self.embedding = nn.Embedding(
            vocab_size,
            d_model,
            padding_idx=padding_idx
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """
        Initialize embedding weights

        Common methods:
          - Random normal: N(0, 1/√d_model)
          - Xavier: uniform(-√(6/(vocab+d)), √(6/(vocab+d)))
        """
        # Xavier uniform initialization
        nn.init.xavier_uniform_(self.embedding.weight)

        # Set padding embedding to zero
        if self.embedding.padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[self.embedding.padding_idx].fill_(0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Convert token IDs to embeddings

        Args:
            token_ids: Token IDs (batch_size, seq_len)

        Returns:
            Embeddings (batch_size, seq_len, d_model)
        """
        # Lookup embeddings
        embeddings = self.embedding(token_ids)

        # Scale by √d_model (as in Transformer paper)
        # Reason: Prevents embeddings from being too small vs positional encoding
        embeddings = embeddings * math.sqrt(self.d_model)

        return embeddings


class EmbeddingAnalyzer:
    """
    Tools for analyzing embeddings
    """

    @staticmethod
    def cosine_similarity(emb1: torch.Tensor, emb2: torch.Tensor) -> float:
        """
        Compute cosine similarity between two embeddings

        Args:
            emb1, emb2: Embedding vectors

        Returns:
            Similarity score [-1, 1]
        """
        cos = nn.CosineSimilarity(dim=0)
        return cos(emb1, emb2).item()

    @staticmethod
    def find_nearest_neighbors(
        embedding: torch.Tensor,
        embedding_matrix: torch.Tensor,
        k: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Find k nearest neighbors in embedding space

        Args:
            embedding: Query embedding (d_model,)
            embedding_matrix: All embeddings (vocab_size, d_model)
            k: Number of neighbors

        Returns:
            List of (token_id, similarity) tuples
        """
        # Compute cosine similarities
        cos = nn.CosineSimilarity(dim=1)
        similarities = cos(
            embedding.unsqueeze(0),
            embedding_matrix
        )

        # Get top k
        top_k_vals, top_k_ids = torch.topk(similarities, k)

        return list(zip(top_k_ids.tolist(), top_k_vals.tolist()))

    @staticmethod
    def demonstrate_embedding_properties():
        """
        Demonstrate key properties of embeddings
        """
        print("="*80)
        print("TOKEN EMBEDDING PROPERTIES")
        print("="*80)

        print("""
1. Dimensionality
   - Typical: 768 (BERT), 1024 (GPT-2), 4096 (GPT-3)
   - Trade-off: Higher dim = more capacity but more parameters

2. Lookup Table
   - Simply indexes into weight matrix
   - embedding[token_id] → vector
   - Extremely fast (O(1))

3. Learned During Training
   - Random initialization
   - Learned via backpropagation
   - Captures semantic similarity

4. Scaling
   - Transformer paper: multiply by √d_model
   - Prevents embeddings from being too small

5. Padding
   - Padding token embedding set to zero
   - Doesn't contribute to gradients

6. Vocabulary Size Impact
   - Vocab: 50k, d_model: 768
   - Parameters: 50k × 768 = 38.4M
   - Significant portion of model!
        """)


# Demo
if __name__ == "__main__":
    print("="*80)
    print("TOKEN EMBEDDINGS DEMO")
    print("="*80)

    # Configuration
    vocab_size = 1000
    d_model = 128
    seq_len = 10
    batch_size = 2

    print(f"\nConfiguration:")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Embedding dimension: {d_model}")

    # Create embedding layer
    embedding_layer = TokenEmbedding(
        vocab_size=vocab_size,
        d_model=d_model,
        padding_idx=0
    )

    # Count parameters
    params = sum(p.numel() for p in embedding_layer.parameters())
    print(f"  Parameters: {params:,}")
    print(f"  Memory (FP32): {params * 4 / 1e6:.2f} MB")

    # Example tokens
    token_ids = torch.randint(1, vocab_size, (batch_size, seq_len))

    print(f"\nToken IDs shape: {token_ids.shape}")
    print(f"Example IDs: {token_ids[0, :5].tolist()}")

    # Get embeddings
    embeddings = embedding_layer(token_ids)

    print(f"\nEmbeddings shape: {embeddings.shape}")
    print(f"  → (batch={batch_size}, seq_len={seq_len}, d_model={d_model})")

    # Analyze embeddings
    print("\n--- Embedding Analysis ---")

    # Get embedding matrix
    emb_matrix = embedding_layer.embedding.weight

    # Check padding embedding
    padding_emb = emb_matrix[0]
    print(f"Padding embedding norm: {padding_emb.norm().item():.6f}")
    print(f"  (should be ~0)")

    # Similarity between random tokens
    token_a = 10
    token_b = 11
    token_c = 500

    emb_a = emb_matrix[token_a]
    emb_b = emb_matrix[token_b]
    emb_c = emb_matrix[token_c]

    analyzer = EmbeddingAnalyzer()

    sim_ab = analyzer.cosine_similarity(emb_a, emb_b)
    sim_ac = analyzer.cosine_similarity(emb_a, emb_c)

    print(f"\nCosine similarities:")
    print(f"  Token {token_a} ↔ Token {token_b}: {sim_ab:.4f}")
    print(f"  Token {token_a} ↔ Token {token_c}: {sim_ac:.4f}")

    # Nearest neighbors
    neighbors = analyzer.find_nearest_neighbors(emb_a, emb_matrix, k=6)
    print(f"\nNearest neighbors of token {token_a}:")
    for token_id, similarity in neighbors:
        print(f"  Token {token_id}: similarity = {similarity:.4f}")

    # Demonstrate properties
    analyzer.demonstrate_embedding_properties()
```

## 6. Special Tokens

```python
"""
Special Tokens = Tokens spéciaux avec fonctions spécifiques

Tokens communs:

[PAD]  - Padding (remplir séquences à même longueur)
[UNK]  - Unknown (mots hors vocabulaire)
[CLS]  - Classification (début de séquence pour BERT)
[SEP]  - Separator (séparer segments)
[MASK] - Masking (MLM dans BERT)
[BOS]  - Beginning of sequence (GPT)
[EOS]  - End of sequence (GPT)

Usage model-specific:
  - BERT: [CLS] text [SEP]
  - GPT: [BOS] text [EOS]
  - T5: text </s>
"""

from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Optional


class SpecialToken(Enum):
    """
    Standard special tokens
    """
    PAD = "[PAD]"    # Padding
    UNK = "[UNK]"    # Unknown
    CLS = "[CLS]"    # Classification token (BERT)
    SEP = "[SEP]"    # Separator
    MASK = "[MASK]"  # Masked token (BERT MLM)
    BOS = "<s>"      # Beginning of sequence
    EOS = "</s>"     # End of sequence


@dataclass
class SpecialTokenConfig:
    """
    Configuration for special tokens
    """
    pad_token: str = "[PAD]"
    unk_token: str = "[UNK]"
    bos_token: str = "<s>"
    eos_token: str = "</s>"
    cls_token: Optional[str] = "[CLS]"
    sep_token: Optional[str] = "[SEP]"
    mask_token: Optional[str] = "[MASK]"

    def get_all_tokens(self) -> List[str]:
        """Get all defined special tokens"""
        tokens = [
            self.pad_token,
            self.unk_token,
            self.bos_token,
            self.eos_token
        ]

        if self.cls_token:
            tokens.append(self.cls_token)
        if self.sep_token:
            tokens.append(self.sep_token)
        if self.mask_token:
            tokens.append(self.mask_token)

        return tokens


class VocabularyBuilder:
    """
    Build vocabulary with special tokens
    """

    def __init__(self, special_token_config: SpecialTokenConfig):
        self.special_config = special_token_config
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}

    def build_vocab(
        self,
        corpus_tokens: List[str],
        max_vocab_size: int = 10000
    ):
        """
        Build vocabulary from corpus

        Args:
            corpus_tokens: All tokens from corpus
            max_vocab_size: Maximum vocabulary size
        """
        # Step 1: Add special tokens first (they get lowest IDs)
        special_tokens = self.special_config.get_all_tokens()

        for idx, token in enumerate(special_tokens):
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token

        # Step 2: Count token frequencies
        from collections import Counter
        token_counts = Counter(corpus_tokens)

        # Step 3: Add most frequent tokens
        remaining_slots = max_vocab_size - len(special_tokens)
        most_common = token_counts.most_common(remaining_slots)

        current_id = len(special_tokens)
        for token, _ in most_common:
            if token not in self.token_to_id:
                self.token_to_id[token] = current_id
                self.id_to_token[current_id] = token
                current_id += 1

        print(f"Built vocabulary: {len(self.token_to_id)} tokens")
        print(f"  Special tokens: {len(special_tokens)}")
        print(f"  Regular tokens: {len(self.token_to_id) - len(special_tokens)}")

    def get_special_token_ids(self) -> Dict[str, int]:
        """Get IDs of all special tokens"""
        special_ids = {}

        for token in self.special_config.get_all_tokens():
            if token in self.token_to_id:
                special_ids[token] = self.token_to_id[token]

        return special_ids


# Demo
if __name__ == "__main__":
    print("="*80)
    print("SPECIAL TOKENS DEMO")
    print("="*80)

    # Configuration
    special_config = SpecialTokenConfig(
        pad_token="[PAD]",
        unk_token="[UNK]",
        bos_token="<s>",
        eos_token="</s>",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]"
    )

    print("\nSpecial tokens:")
    for token in special_config.get_all_tokens():
        print(f"  {token}")

    # Build vocabulary
    print("\n--- Building Vocabulary ---")

    # Mock corpus tokens
    corpus_tokens = ["hello", "world", "hello", "foo", "bar", "world"] * 100

    vocab_builder = VocabularyBuilder(special_config)
    vocab_builder.build_vocab(corpus_tokens, max_vocab_size=100)

    # Show special token IDs
    print("\nSpecial token IDs:")
    special_ids = vocab_builder.get_special_token_ids()
    for token, idx in special_ids.items():
        print(f"  {token}: {idx}")

    # Show usage examples
    print("\n" + "="*80)
    print("USAGE EXAMPLES")
    print("="*80)

    examples = [
        {
            "model": "BERT",
            "format": "[CLS] sentence A [SEP] sentence B [SEP]",
            "example": "[CLS] I love NLP [SEP] It is amazing [SEP]",
            "purpose": "Classification/sentence pair tasks"
        },
        {
            "model": "GPT",
            "format": "<s> text </s>",
            "example": "<s> Once upon a time </s>",
            "purpose": "Text generation"
        },
        {
            "model": "T5",
            "format": "task: input </s>",
            "example": "translate English to French: Hello </s>",
            "purpose": "Text-to-text tasks"
        },
        {
            "model": "BERT (MLM)",
            "format": "[CLS] text with [MASK] [SEP]",
            "example": "[CLS] I [MASK] NLP [SEP]",
            "purpose": "Masked language modeling"
        }
    ]

    for ex in examples:
        print(f"\n{ex['model']}")
        print(f"  Format: {ex['format']}")
        print(f"  Example: {ex['example']}")
        print(f"  Purpose: {ex['purpose']}")

    print("\n" + "="*80)
    print("SPECIAL TOKEN FUNCTIONS")
    print("="*80)

    functions = {
        "[PAD]": "Padding - Fill sequences to same length in batch",
        "[UNK]": "Unknown - Replace out-of-vocabulary words",
        "[CLS]": "Classification - Aggregate representation (BERT)",
        "[SEP]": "Separator - Separate segments/sentences",
        "[MASK]": "Mask - Masked language modeling (BERT)",
        "<s> / [BOS]": "Beginning of sequence - Start marker (GPT)",
        "</s> / [EOS]": "End of sequence - Stop generation"
    }

    for token, function in functions.items():
        print(f"\n{token}")
        print(f"  → {function}")
```

## 7. Projet Complet: Custom Tokenizer + Embeddings

```python
"""
PROJET COMPLET: Tokenizer et Embeddings from Scratch

Implémente:
  1. BPE tokenizer
  2. Vocabulary with special tokens
  3. Token embeddings
  4. Complete text processing pipeline
"""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional
from collections import Counter
import pickle


class CompleteTokenizationPipeline:
    """
    Complete tokenization pipeline

    Includes:
      - BPE tokenizer
      - Special tokens
      - Vocabulary management
      - Token embeddings
    """

    def __init__(
        self,
        vocab_size: int = 10000,
        d_model: int = 768,
        max_seq_len: int = 512
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Special tokens
        self.pad_token = "[PAD]"
        self.unk_token = "[UNK]"
        self.bos_token = "<s>"
        self.eos_token = "</s>"

        # Vocabulary
        self.token_to_id: Dict[str, int] = {
            self.pad_token: 0,
            self.unk_token: 1,
            self.bos_token: 2,
            self.eos_token: 3
        }
        self.id_to_token: Dict[int, str] = {
            0: self.pad_token,
            1: self.unk_token,
            2: self.bos_token,
            3: self.eos_token
        }

        # BPE merges
        self.merges: List[Tuple[str, str]] = []

        # Embedding layer (created after training)
        self.embedding_layer: Optional[TokenEmbedding] = None

    def train(self, corpus: List[str]):
        """
        Train tokenizer on corpus

        Args:
            corpus: List of training texts
        """
        print("="*80)
        print("TRAINING COMPLETE TOKENIZATION PIPELINE")
        print("="*80)

        # Simple BPE training (simplified from Part 1)
        # In practice, use HuggingFace tokenizers library

        print("\n[1/3] Training BPE tokenizer...")

        # Pre-tokenize
        words = []
        for text in corpus:
            words.extend(text.lower().split())

        word_counts = Counter(words)

        # Initialize with characters
        word_freqs = {}
        for word, count in word_counts.items():
            chars = tuple(word) + ('</w>',)
            word_freqs[chars] = count

        # Get initial vocab
        vocab = set()
        for word in word_freqs.keys():
            vocab.update(word)

        # Perform merges
        num_merges = self.vocab_size - len(vocab) - 4  # 4 special tokens

        for i in range(num_merges):
            # Get pair frequencies
            pairs = Counter()
            for word, freq in word_freqs.items():
                for j in range(len(word) - 1):
                    pairs[(word[j], word[j+1])] += freq

            if not pairs:
                break

            # Best pair
            best_pair = max(pairs, key=pairs.get)
            self.merges.append(best_pair)

            # Merge
            new_word_freqs = {}
            for word, freq in word_freqs.items():
                new_word = []
                k = 0
                while k < len(word):
                    if k < len(word) - 1 and (word[k], word[k+1]) == best_pair:
                        new_word.append(''.join(best_pair))
                        k += 2
                    else:
                        new_word.append(word[k])
                        k += 1
                new_word_freqs[tuple(new_word)] = freq

            word_freqs = new_word_freqs

            # Add to vocab
            vocab.add(''.join(best_pair))

        print(f"  Learned {len(self.merges)} merges")

        print("\n[2/3] Building vocabulary...")

        # Build vocab
        current_id = 4  # After special tokens
        for token in sorted(vocab):
            if token not in self.token_to_id:
                self.token_to_id[token] = current_id
                self.id_to_token[current_id] = token
                current_id += 1

        print(f"  Vocabulary size: {len(self.token_to_id)}")

        print("\n[3/3] Initializing embeddings...")

        # Create embedding layer
        self.embedding_layer = TokenEmbedding(
            vocab_size=len(self.token_to_id),
            d_model=self.d_model,
            padding_idx=0
        )

        params = sum(p.numel() for p in self.embedding_layer.parameters())
        print(f"  Embedding parameters: {params:,}")

        print("\n✅ Pipeline trained successfully!")

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Encode text to token IDs

        Args:
            text: Input text
            add_special_tokens: Add BOS/EOS
            max_length: Maximum sequence length
            padding: Pad to max_length

        Returns:
            Dictionary with 'input_ids' and 'attention_mask'
        """
        if max_length is None:
            max_length = self.max_seq_len

        # Tokenize words
        words = text.lower().split()

        # Apply BPE
        tokens = []
        for word in words:
            word_tokens = self._tokenize_word(word)
            tokens.extend(word_tokens)

        # Convert to IDs
        token_ids = [
            self.token_to_id.get(t, self.token_to_id[self.unk_token])
            for t in tokens
        ]

        # Add special tokens
        if add_special_tokens:
            token_ids = [self.token_to_id[self.bos_token]] + \
                       token_ids + \
                       [self.token_to_id[self.eos_token]]

        # Truncate
        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]

        # Padding
        attention_mask = [1] * len(token_ids)

        if padding:
            padding_length = max_length - len(token_ids)
            token_ids = token_ids + [self.token_to_id[self.pad_token]] * padding_length
            attention_mask = attention_mask + [0] * padding_length

        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
        }

    def _tokenize_word(self, word: str) -> List[str]:
        """Tokenize single word with BPE"""
        word = tuple(word) + ('</w>',)

        for pair in self.merges:
            if len(word) < 2:
                break

            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i+1]) == pair:
                    new_word.append(''.join(pair))
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            word = tuple(new_word)

        return list(word)

    def decode(self, token_ids: torch.Tensor) -> str:
        """Decode token IDs to text"""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        tokens = [self.id_to_token.get(id, self.unk_token) for id in token_ids]

        # Remove special tokens
        tokens = [t for t in tokens if t not in [
            self.pad_token, self.bos_token, self.eos_token
        ]]

        # Join
        text = ''.join(tokens).replace('</w>', ' ').strip()

        return text

    def get_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Get embeddings for token IDs"""
        assert self.embedding_layer is not None, "Must train pipeline first"
        return self.embedding_layer(token_ids)

    def save(self, path: str):
        """Save tokenizer"""
        with open(path, 'wb') as f:
            pickle.dump({
                'vocab_size': self.vocab_size,
                'd_model': self.d_model,
                'token_to_id': self.token_to_id,
                'id_to_token': self.id_to_token,
                'merges': self.merges
            }, f)
        print(f"Saved tokenizer to {path}")

    def load(self, path: str):
        """Load tokenizer"""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        self.vocab_size = data['vocab_size']
        self.d_model = data['d_model']
        self.token_to_id = data['token_to_id']
        self.id_to_token = data['id_to_token']
        self.merges = data['merges']

        # Recreate embedding layer
        self.embedding_layer = TokenEmbedding(
            vocab_size=len(self.token_to_id),
            d_model=self.d_model,
            padding_idx=0
        )

        print(f"Loaded tokenizer from {path}")


# Demo / Main
if __name__ == "__main__":
    print("="*80)
    print("COMPLETE TOKENIZATION PIPELINE - PROJECT")
    print("="*80)

    # Training corpus
    corpus = [
        "the quick brown fox jumps over the lazy dog",
        "the dog was really lazy and brown",
        "the fox was very quick",
        "natural language processing is amazing",
        "language models are powerful",
        "tokenization is the first step"
    ] * 10  # Repeat for more data

    print(f"\nCorpus: {len(corpus)} texts")

    # Train pipeline
    pipeline = CompleteTokenizationPipeline(
        vocab_size=500,
        d_model=128,
        max_seq_len=50
    )

    pipeline.train(corpus)

    # Test encoding
    print("\n" + "="*80)
    print("ENCODING EXAMPLES")
    print("="*80)

    test_texts = [
        "the quick fox",
        "language processing",
        "amazing models"
    ]

    for text in test_texts:
        # Encode
        encoded = pipeline.encode(text, padding=True, max_length=20)

        # Get embeddings
        embeddings = pipeline.get_embeddings(encoded['input_ids'].unsqueeze(0))

        # Decode
        decoded = pipeline.decode(encoded['input_ids'])

        print(f"\nText: '{text}'")
        print(f"  Token IDs: {encoded['input_ids'].tolist()}")
        print(f"  Attention mask: {encoded['attention_mask'].tolist()}")
        print(f"  Embeddings shape: {embeddings.shape}")
        print(f"  Decoded: '{decoded}'")

    print("\n" + "="*80)
    print("✅ CHAPTER 3 COMPLETE!")
    print("="*80)
    print("""
You've learned:
  ✅ Tokenization approaches (character, word, subword)
  ✅ BPE algorithm (GPT tokenizer)
  ✅ WordPiece algorithm (BERT tokenizer)
  ✅ SentencePiece (T5, Llama tokenizer)
  ✅ Token embeddings
  ✅ Special tokens and their usage
  ✅ Complete tokenization pipeline from scratch

Next chapter:
  → Chapter 4: Mathematics for LLMs
    (Backpropagation, optimization, loss functions)
    """)
```
