# Chapitre 3: Tokenization et Embeddings

## Introduction

La **tokenization** est LA première étape cruciale dans tout pipeline NLP/LLM. Les modèles ne comprennent pas le texte brut - ils travaillent avec des nombres. La tokenization transforme le texte en séquence de tokens (unités de base), puis en nombres.

### Pourquoi la tokenization est critique?

```python
"""
Problème: Les ordinateurs ne comprennent pas le texte

Texte: "Hello, world!"
→ Comment le donner à un modèle neural?

Solutions (évolution):

1. Character-level (1 char = 1 token)
   "Hello" → ['H', 'e', 'l', 'l', 'o']
   ❌ Séquences trop longues
   ❌ Perte de sémantique des mots
   ✅ Petit vocabulaire (~256 pour ASCII)

2. Word-level (1 mot = 1 token)
   "Hello, world!" → ['Hello', ',', 'world', '!']
   ✅ Sémantique préservée
   ❌ Vocabulaire énorme (100k-1M+ mots)
   ❌ OOV (out-of-vocabulary) problem
   ❌ Morphology: "run", "running", "runs" = 3 tokens différents

3. Subword tokenization (OPTIMAL)
   "Hello, world!" → ['Hello', ',', 'world', '!']
   "running" → ['run', 'ning']
   "unhappiness" → ['un', 'happiness']

   ✅ Balance vocabulaire (30k-50k tokens)
   ✅ Gère OOV (décompose en sous-mots connus)
   ✅ Capture morphology
   ✅ Efficace pour toutes langues

Résultat: Tous les LLMs modernes utilisent subword tokenization
  - GPT: BPE (Byte Pair Encoding)
  - BERT: WordPiece
  - T5, Llama: SentencePiece (Unigram)
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from collections import Counter, defaultdict
import re


@dataclass
class TokenizationComparison:
    """
    Comparaison des approches de tokenization
    """
    name: str
    example_input: str
    example_output: List[str]
    vocab_size: str
    pros: List[str]
    cons: List[str]
    used_in: List[str]


TOKENIZATION_APPROACHES = [
    TokenizationComparison(
        name="Character-level",
        example_input="Hello world!",
        example_output=['H','e','l','l','o',' ','w','o','r','l','d','!'],
        vocab_size="Small (~256-1000)",
        pros=[
            "Tiny vocabulary",
            "No OOV problem",
            "Simple implementation"
        ],
        cons=[
            "Very long sequences",
            "Loses word semantics",
            "Hard to learn patterns",
            "Inefficient"
        ],
        used_in=["Early RNN models", "Character-level CNNs"]
    ),

    TokenizationComparison(
        name="Word-level",
        example_input="Hello world!",
        example_output=['Hello', 'world', '!'],
        vocab_size="Large (100k-1M+)",
        pros=[
            "Preserves word semantics",
            "Short sequences",
            "Intuitive"
        ],
        cons=[
            "Huge vocabulary",
            "OOV problem (unknown words)",
            "Doesn't capture morphology",
            "Language-dependent"
        ],
        used_in=["Word2Vec", "GloVe", "Early NMT"]
    ),

    TokenizationComparison(
        name="Subword (BPE)",
        example_input="Hello world!",
        example_output=['Hello', 'Ġworld', '!'],
        vocab_size="Medium (30k-50k)",
        pros=[
            "Balance vocab size & sequence length",
            "Handles OOV (decomposes unknown words)",
            "Captures morphology",
            "Language-agnostic"
        ],
        cons=[
            "Less intuitive",
            "Tokenization may split words oddly",
            "Requires training"
        ],
        used_in=["GPT-2/3/4", "RoBERTa", "BART", "Most modern LLMs"]
    ),

    TokenizationComparison(
        name="Subword (WordPiece)",
        example_input="Hello world!",
        example_output=['Hello', 'world', '!'],
        vocab_size="Medium (30k)",
        pros=[
            "Similar to BPE",
            "Good for BERT-style models",
            "Handles OOV"
        ],
        cons=[
            "Slightly different algorithm",
            "May split differently than BPE"
        ],
        used_in=["BERT", "DistilBERT", "ELECTRA"]
    ),

    TokenizationComparison(
        name="Subword (SentencePiece)",
        example_input="Hello world!",
        example_output=['▁Hello', '▁world', '!'],
        vocab_size="Medium (32k)",
        pros=[
            "Language-agnostic",
            "Treats text as byte stream",
            "No pre-tokenization needed",
            "Reversible"
        ],
        cons=[
            "More complex implementation"
        ],
        used_in=["T5", "mT5", "ALBERT", "Llama 2", "XLNet"]
    )
]


def print_tokenization_comparison():
    """
    Print comparison of tokenization approaches
    """
    print("="*100)
    print("TOKENIZATION APPROACHES COMPARISON")
    print("="*100)

    for approach in TOKENIZATION_APPROACHES:
        print(f"\n{'='*100}")
        print(f"{approach.name}")
        print('='*100)
        print(f"Example: '{approach.example_input}'")
        print(f"Tokens: {approach.example_output}")
        print(f"Vocab Size: {approach.vocab_size}")

        print("\nPros:")
        for pro in approach.pros:
            print(f"  ✅ {pro}")

        print("\nCons:")
        for con in approach.cons:
            print(f"  ❌ {con}")

        print(f"\nUsed in: {', '.join(approach.used_in)}")


if __name__ == "__main__":
    print_tokenization_comparison()

    print("\n" + "="*100)
    print("WHICH TOKENIZATION TO USE?")
    print("="*100)
    print("""
For Modern LLMs:
  → Use SUBWORD tokenization (BPE, WordPiece, or SentencePiece)

Specific recommendations:
  • GPT-style models: BPE (Byte Pair Encoding)
  • BERT-style models: WordPiece
  • Multilingual models: SentencePiece (handles all languages uniformly)
  • Production LLMs: All use subword (GPT-4, Claude, Llama all use variants)

Vocabulary size sweet spot:
  • Too small (< 10k): Sequences too long
  • Too large (> 100k): Embedding matrix huge
  • Optimal: 30k-50k tokens
    """)
```

## 1. BPE (Byte Pair Encoding) - GPT Tokenizer

```python
"""
BPE (Byte Pair Encoding)
========================

Algorithm (greedy):
  1. Start with character vocabulary
  2. Find most frequent pair of tokens
  3. Merge this pair into a new token
  4. Repeat until vocab_size reached

Example:
  Corpus: "low low low lower lowest"

  Initial vocab: ['l', 'o', 'w', 'e', 'r', 's', 't']

  Iteration 1: Most frequent pair = ('l', 'o')
    → Merge to 'lo'
    → Corpus: "lo w lo w lo w lo wer lo west"

  Iteration 2: Most frequent pair = ('lo', 'w')
    → Merge to 'low'
    → Corpus: "low low low lower lowest"

  Iteration 3: Most frequent pair = ('low', 'e')
    → Merge to 'lowe'
    → Corpus: "low low low lower lowest"

  ... continue until vocab_size reached

Used by: GPT-2, GPT-3, RoBERTa, BART
"""

import re
from collections import Counter
from typing import List, Dict, Tuple


class BytePairEncodingTokenizer:
    """
    BPE Tokenizer (simplified implementation)

    Based on GPT-2 tokenizer, but simplified for clarity.

    Args:
        vocab_size: Target vocabulary size
    """

    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.vocab: Dict[str, int] = {}
        self.merges: List[Tuple[str, str]] = []
        self.byte_encoder = self._bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

    @staticmethod
    def _bytes_to_unicode() -> Dict[int, str]:
        """
        Create mapping from bytes to unicode characters

        GPT-2 uses this to handle any byte sequence.
        """
        # Standard printable ASCII
        bs = list(range(ord("!"), ord("~") + 1)) + \
             list(range(ord("¡"), ord("¬") + 1)) + \
             list(range(ord("®"), ord("ÿ") + 1))

        cs = bs[:]
        n = 0

        # Add non-printable bytes
        for b in range(2**8):
            if b not in bs:
                bs.append(b)
                cs.append(2**8 + n)
                n += 1

        cs = [chr(n) for n in cs]
        return dict(zip(bs, cs))

    def _get_stats(self, word_freqs: Dict[Tuple[str, ...], int]) -> Counter:
        """
        Count frequency of adjacent pairs

        Args:
            word_freqs: Dictionary of (word_tuple → frequency)

        Returns:
            Counter of (pair → frequency)
        """
        pairs = Counter()

        for word, freq in word_freqs.items():
            symbols = list(word)
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq

        return pairs

    def _merge_pair(
        self,
        pair: Tuple[str, str],
        word_freqs: Dict[Tuple[str, ...], int]
    ) -> Dict[Tuple[str, ...], int]:
        """
        Merge all occurrences of a pair

        Args:
            pair: Pair to merge (a, b)
            word_freqs: Current word frequencies

        Returns:
            Updated word frequencies after merge
        """
        new_word_freqs = {}

        bigram = ' '.join(pair)
        replacement = ''.join(pair)

        for word, freq in word_freqs.items():
            # Convert word to string, merge, convert back to tuple
            word_str = ' '.join(word)
            word_str = word_str.replace(bigram, replacement)
            new_word = tuple(word_str.split())
            new_word_freqs[new_word] = freq

        return new_word_freqs

    def train(self, corpus: List[str]):
        """
        Train BPE on corpus

        Args:
            corpus: List of texts to train on
        """
        print("="*80)
        print("TRAINING BPE TOKENIZER")
        print("="*80)

        # Step 1: Pre-tokenize (split on spaces, punctuation)
        # For simplicity, just split on spaces
        words = []
        for text in corpus:
            words.extend(text.split())

        # Step 2: Count word frequencies
        word_counts = Counter(words)

        # Step 3: Split words into characters
        # Add end-of-word marker (like GPT-2's Ġ for space)
        word_freqs = {}
        for word, count in word_counts.items():
            # Split into characters
            chars = tuple(word) + ('</w>',)
            word_freqs[chars] = count

        # Step 4: Initialize vocabulary with characters
        vocab = set()
        for word in word_freqs.keys():
            vocab.update(word)

        print(f"\nInitial vocabulary size: {len(vocab)}")
        print(f"Target vocabulary size: {self.vocab_size}")

        # Step 5: Iteratively merge most frequent pairs
        num_merges = self.vocab_size - len(vocab)

        print(f"\nPerforming {num_merges} merges...")

        for i in range(num_merges):
            # Get pair frequencies
            pairs = self._get_stats(word_freqs)

            if not pairs:
                break

            # Get most frequent pair
            best_pair = max(pairs, key=pairs.get)

            # Merge this pair
            word_freqs = self._merge_pair(best_pair, word_freqs)

            # Save merge operation
            self.merges.append(best_pair)

            # Add to vocabulary
            new_token = ''.join(best_pair)
            vocab.add(new_token)

            # Print progress
            if (i + 1) % 100 == 0 or i < 10:
                print(f"  Merge {i+1}/{num_merges}: {best_pair} → '{new_token}' "
                      f"(freq: {pairs[best_pair]})")

        # Step 6: Create vocab dictionary
        self.vocab = {token: idx for idx, token in enumerate(sorted(vocab))}

        print(f"\nFinal vocabulary size: {len(self.vocab)}")
        print(f"Number of merges: {len(self.merges)}")

    def _tokenize_word(self, word: str) -> List[str]:
        """
        Tokenize a single word using learned merges

        Args:
            word: Word to tokenize

        Returns:
            List of tokens
        """
        # Split into characters
        word = tuple(word) + ('</w>',)

        # Apply merges in order
        for pair in self.merges:
            if len(word) < 2:
                break

            # Find all occurrences of this pair
            i = 0
            new_word = []
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i+1]) == pair:
                    # Merge
                    new_word.append(''.join(pair))
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            word = tuple(new_word)

        return list(word)

    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs

        Args:
            text: Text to encode

        Returns:
            List of token IDs
        """
        # Pre-tokenize (split on spaces)
        words = text.split()

        # Tokenize each word
        tokens = []
        for word in words:
            tokens.extend(self._tokenize_word(word))

        # Convert to IDs
        token_ids = [self.vocab.get(t, self.vocab.get('<unk>', 0)) for t in tokens]

        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs to text

        Args:
            token_ids: List of token IDs

        Returns:
            Decoded text
        """
        # Create reverse vocab
        id_to_token = {v: k for k, v in self.vocab.items()}

        # Convert IDs to tokens
        tokens = [id_to_token.get(id, '<unk>') for id in token_ids]

        # Join and remove end-of-word markers
        text = ''.join(tokens).replace('</w>', ' ').strip()

        return text


# Demo
if __name__ == "__main__":
    print("\n" + "="*80)
    print("BPE TOKENIZER DEMO")
    print("="*80)

    # Training corpus
    corpus = [
        "low low low lower lowest",
        "new new newer newest",
        "wide wide wider widest",
    ]

    print("\nTraining corpus:")
    for text in corpus:
        print(f"  '{text}'")

    # Train tokenizer
    tokenizer = BytePairEncodingTokenizer(vocab_size=50)
    tokenizer.train(corpus)

    # Test encoding
    print("\n" + "="*80)
    print("ENCODING EXAMPLES")
    print("="*80)

    test_texts = [
        "low",
        "lower",
        "lowest",
        "new",
        "newer",
        "wide wider"
    ]

    for text in test_texts:
        tokens = tokenizer._tokenize_word(text) if ' ' not in text else text.split()
        token_ids = tokenizer.encode(text)
        decoded = tokenizer.decode(token_ids)

        print(f"\nText: '{text}'")
        print(f"  Tokens: {tokens}")
        print(f"  IDs: {token_ids}")
        print(f"  Decoded: '{decoded}'")

    # Show learned merges
    print("\n" + "="*80)
    print("LEARNED MERGES (first 20)")
    print("="*80)
    for i, (a, b) in enumerate(tokenizer.merges[:20]):
        merged = ''.join([a, b])
        print(f"  {i+1}. ({a}, {b}) → '{merged}'")
```

*[Suite avec WordPiece et SentencePiece dans la partie 2...]*
