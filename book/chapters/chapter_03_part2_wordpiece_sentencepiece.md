# Chapitre 3 (Partie 2): WordPiece et SentencePiece

## 2. WordPiece Tokenization (BERT)

```python
"""
WordPiece Tokenization
======================

Similar to BPE but with key differences:

Difference from BPE:
  - BPE: Merges most FREQUENT pair
  - WordPiece: Merges pair with highest LIKELIHOOD (based on language model)

Formula:
  score(a, b) = freq(ab) / (freq(a) × freq(b))

→ Chooses merge that maximizes likelihood of training data

Characteristics:
  - Uses ## prefix for subword continuation
  - Example: "unwanted" → ["un", "##want", "##ed"]
  - Vocabulary ~30k tokens

Used by: BERT, DistilBERT, ELECTRA, ALBERT (some variants)

Note: In practice, WordPiece and BPE give similar results.
Most modern libraries use BPE for simplicity.
"""

from typing import List, Dict, Tuple
from collections import Counter, defaultdict
import math


class WordPieceTokenizer:
    """
    WordPiece Tokenizer (BERT-style)

    Simplified implementation for educational purposes.

    Args:
        vocab_size: Target vocabulary size
        unk_token: Unknown token
        continuing_subword_prefix: Prefix for continuing subwords (default: ##)
    """

    def __init__(
        self,
        vocab_size: int = 30000,
        unk_token: str = "[UNK]",
        continuing_subword_prefix: str = "##"
    ):
        self.vocab_size = vocab_size
        self.unk_token = unk_token
        self.prefix = continuing_subword_prefix
        self.vocab: Dict[str, int] = {}
        self.inv_vocab: Dict[int, str] = {}

    def _compute_pair_scores(
        self,
        word_freqs: Dict[Tuple[str, ...], int]
    ) -> Dict[Tuple[str, str], float]:
        """
        Compute scores for all pairs

        Score formula: P(ab) / (P(a) × P(b))
        Higher score = better merge

        Args:
            word_freqs: Word frequencies

        Returns:
            Scores for each pair
        """
        # Count token frequencies
        token_freqs = Counter()
        pair_freqs = Counter()

        for word, freq in word_freqs.items():
            for token in word:
                token_freqs[token] += freq

            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pair_freqs[pair] += freq

        # Compute scores
        scores = {}
        for pair, pair_freq in pair_freqs.items():
            token_a, token_b = pair
            # Avoid division by zero
            score = pair_freq / (token_freqs[token_a] * token_freqs[token_b] + 1e-10)
            scores[pair] = score

        return scores

    def train(self, corpus: List[str]):
        """
        Train WordPiece tokenizer

        Args:
            corpus: Training texts
        """
        print("="*80)
        print("TRAINING WORDPIECE TOKENIZER")
        print("="*80)

        # Step 1: Pre-tokenize (split on whitespace)
        words = []
        for text in corpus:
            # Simple whitespace tokenization
            words.extend(text.lower().split())

        # Count word frequencies
        word_counts = Counter(words)

        # Step 2: Split into characters, mark continuations
        word_freqs = {}
        for word, count in word_counts.items():
            # First char: no prefix
            # Remaining chars: add ## prefix
            chars = [word[0]] + [f"{self.prefix}{c}" for c in word[1:]]
            word_freqs[tuple(chars)] = count

        # Step 3: Initialize vocabulary with characters
        vocab = set()
        for word in word_freqs.keys():
            vocab.update(word)

        # Add special tokens
        vocab.add(self.unk_token)

        print(f"\nInitial vocabulary size: {len(vocab)}")
        print(f"Target vocabulary size: {self.vocab_size}")

        # Step 4: Iteratively merge best pairs
        num_merges = self.vocab_size - len(vocab)

        print(f"\nPerforming {num_merges} merges...")

        for i in range(num_merges):
            # Compute scores for all pairs
            scores = self._compute_pair_scores(word_freqs)

            if not scores:
                break

            # Get best pair (highest score)
            best_pair = max(scores, key=scores.get)

            # Merge this pair
            new_word_freqs = {}
            for word, freq in word_freqs.items():
                new_word = []
                j = 0
                while j < len(word):
                    if j < len(word) - 1 and (word[j], word[j+1]) == best_pair:
                        # Merge
                        merged = word[j] + word[j+1].replace(self.prefix, "")
                        # Keep ## prefix if second token had it
                        if word[j+1].startswith(self.prefix):
                            merged = word[j] + word[j+1][len(self.prefix):]
                        new_word.append(merged)
                        j += 2
                    else:
                        new_word.append(word[j])
                        j += 1

                new_word_freqs[tuple(new_word)] = freq

            word_freqs = new_word_freqs

            # Add to vocabulary
            new_token = ''.join(best_pair).replace(self.prefix, "")
            if best_pair[1].startswith(self.prefix):
                new_token = best_pair[0] + best_pair[1][len(self.prefix):]

            vocab.add(new_token)

            if (i + 1) % 100 == 0 or i < 10:
                print(f"  Merge {i+1}: {best_pair} → '{new_token}' "
                      f"(score: {scores[best_pair]:.6f})")

        # Step 5: Create vocabulary mappings
        self.vocab = {token: idx for idx, token in enumerate(sorted(vocab))}
        self.inv_vocab = {idx: token for token, idx in self.vocab.items()}

        print(f"\nFinal vocabulary size: {len(self.vocab)}")

    def _tokenize_word(self, word: str) -> List[str]:
        """
        Tokenize a single word

        Uses greedy longest-match algorithm.

        Args:
            word: Word to tokenize

        Returns:
            List of subword tokens
        """
        if not word:
            return []

        tokens = []
        start = 0

        while start < len(word):
            end = len(word)
            found = False

            # Try to find longest matching substring
            while start < end:
                substr = word[start:end]

                # Add ## prefix for continuation
                if start > 0:
                    substr = f"{self.prefix}{substr}"

                if substr in self.vocab:
                    tokens.append(substr)
                    found = True
                    break

                end -= 1

            if not found:
                # Unknown token
                tokens.append(self.unk_token)
                start += 1
            else:
                start = end

        return tokens

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        # Pre-tokenize
        words = text.lower().split()

        # Tokenize each word
        tokens = []
        for word in words:
            tokens.extend(self._tokenize_word(word))

        return tokens

    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs

        Args:
            text: Input text

        Returns:
            List of token IDs
        """
        tokens = self.tokenize(text)
        return [self.vocab.get(t, self.vocab[self.unk_token]) for t in tokens]

    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs to text

        Args:
            token_ids: List of token IDs

        Returns:
            Decoded text
        """
        tokens = [self.inv_vocab.get(id, self.unk_token) for id in token_ids]

        # Join tokens, removing ## prefixes
        text = ""
        for token in tokens:
            if token.startswith(self.prefix):
                text += token[len(self.prefix):]
            else:
                text += " " + token

        return text.strip()


# Demo
if __name__ == "__main__":
    print("\n" + "="*80)
    print("WORDPIECE TOKENIZER DEMO")
    print("="*80)

    # Training corpus (small example)
    corpus = [
        "the quick brown fox jumps over the lazy dog",
        "the dog was really lazy",
        "the fox was very quick and brown"
    ]

    print("\nTraining corpus:")
    for text in corpus:
        print(f"  '{text}'")

    # Train tokenizer
    tokenizer = WordPieceTokenizer(vocab_size=100)
    tokenizer.train(corpus)

    # Test
    print("\n" + "="*80)
    print("TOKENIZATION EXAMPLES")
    print("="*80)

    test_texts = [
        "the quick fox",
        "the lazy dog",
        "really quick",
        "unknown"  # OOV word
    ]

    for text in test_texts:
        tokens = tokenizer.tokenize(text)
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)

        print(f"\nText: '{text}'")
        print(f"  Tokens: {tokens}")
        print(f"  IDs: {ids}")
        print(f"  Decoded: '{decoded}'")

    # Show sample vocabulary
    print("\n" + "="*80)
    print("SAMPLE VOCABULARY (first 30)")
    print("="*80)
    for i, (token, idx) in enumerate(sorted(tokenizer.vocab.items(), key=lambda x: x[1])[:30]):
        print(f"  {idx:3d}: '{token}'")
```

## 3. SentencePiece (T5, Llama)

```python
"""
SentencePiece Tokenization
===========================

Key Innovation: Treats text as raw byte sequence
  - No language-specific pre-tokenization
  - Works for ALL languages uniformly
  - Reversible (can perfectly reconstruct original text)

Features:
  - Uses ▁ (U+2581) to represent space
  - Two algorithms: BPE or Unigram
  - Language-agnostic

Example:
  Input: "Hello world"
  → ["▁Hello", "▁world"]  (space becomes ▁)

Advantages:
  ✅ No pre-tokenization needed
  ✅ Language-agnostic (same code for EN, ZH, AR, etc.)
  ✅ Perfectly reversible
  ✅ Handles any text (even binary data)

Used by:
  - T5, mT5 (multilingual)
  - ALBERT
  - XLNet
  - Llama 2 (modified BPE variant)

In practice: Use official SentencePiece library
  pip install sentencepiece

Here: Simplified implementation for understanding
"""

from typing import List, Dict, Tuple
from collections import Counter
import math


class SentencePieceTokenizer:
    """
    Simplified SentencePiece Tokenizer

    Implements Unigram Language Model algorithm.

    Note: This is educational. Use official sentencepiece library in production.

    Args:
        vocab_size: Target vocabulary size
    """

    def __init__(self, vocab_size: int = 8000):
        self.vocab_size = vocab_size
        self.vocab: Dict[str, float] = {}  # token → score
        self.space_token = "▁"  # U+2581 (lower one eighth block)

    def _get_all_substrings(
        self,
        word: str,
        max_len: int = 10
    ) -> List[str]:
        """
        Get all substrings of a word

        Args:
            word: Input word
            max_len: Maximum substring length

        Returns:
            List of substrings
        """
        substrings = set()

        for i in range(len(word)):
            for j in range(i + 1, min(i + max_len + 1, len(word) + 1)):
                substrings.add(word[i:j])

        return list(substrings)

    def _compute_loss(
        self,
        words: List[str],
        word_counts: Counter
    ) -> float:
        """
        Compute likelihood (negative log likelihood)

        Lower is better.

        Args:
            words: Corpus words
            word_counts: Word frequencies

        Returns:
            Total loss
        """
        total_loss = 0

        for word, count in word_counts.items():
            # Tokenize word with current vocab
            tokens = self._viterbi_tokenize(word)

            # Compute log likelihood
            word_loss = 0
            for token in tokens:
                if token in self.vocab:
                    word_loss += self.vocab[token]
                else:
                    word_loss += -10  # Large penalty for unknown

            total_loss += word_loss * count

        return total_loss

    def _viterbi_tokenize(self, word: str) -> List[str]:
        """
        Tokenize word using Viterbi algorithm (dynamic programming)

        Finds best tokenization according to vocab scores.

        Args:
            word: Word to tokenize

        Returns:
            List of tokens
        """
        n = len(word)

        # DP: best_score[i] = best score for word[0:i]
        best_score = [float('-inf')] * (n + 1)
        best_score[0] = 0

        # Backpointer
        backpointer = [0] * (n + 1)

        # Fill DP table
        for i in range(1, n + 1):
            for j in range(i):
                substr = word[j:i]

                if substr in self.vocab:
                    score = best_score[j] + self.vocab[substr]

                    if score > best_score[i]:
                        best_score[i] = score
                        backpointer[i] = j

        # Backtrack to get tokens
        tokens = []
        pos = n
        while pos > 0:
            start = backpointer[pos]
            tokens.append(word[start:pos])
            pos = start

        return list(reversed(tokens))

    def train(self, corpus: List[str], seed_size: int = 10000):
        """
        Train SentencePiece tokenizer

        Simplified Unigram algorithm:
          1. Initialize with all possible substrings
          2. Iteratively prune vocabulary based on likelihood

        Args:
            corpus: Training texts
            seed_size: Initial vocabulary size
        """
        print("="*80)
        print("TRAINING SENTENCEPIECE TOKENIZER")
        print("="*80)

        # Step 1: Add space prefix to simulate SentencePiece
        words = []
        for text in corpus:
            # Split on space, add ▁ prefix
            for word in text.split():
                words.append(f"{self.space_token}{word}")

        word_counts = Counter(words)

        print(f"\nCorpus statistics:")
        print(f"  Unique words: {len(word_counts)}")
        print(f"  Total words: {sum(word_counts.values())}")

        # Step 2: Build initial vocabulary (all substrings)
        print(f"\nBuilding initial vocabulary (target: {seed_size})...")

        substrings = set()
        for word in word_counts.keys():
            substrings.update(self._get_all_substrings(word, max_len=10))

        # Limit to seed_size most frequent
        substring_counts = Counter()
        for word, count in word_counts.items():
            for substr in self._get_all_substrings(word):
                if substr in substrings:
                    substring_counts[substr] += count

        # Keep top seed_size
        top_substrings = [s for s, _ in substring_counts.most_common(seed_size)]

        # Initialize scores (log probability)
        total_count = sum(substring_counts.values())
        for substr in top_substrings:
            prob = substring_counts[substr] / total_count
            self.vocab[substr] = math.log(prob)

        print(f"Initial vocabulary size: {len(self.vocab)}")

        # Step 3: Prune vocabulary (simplified)
        # In full SentencePiece: iteratively remove tokens and check if loss increases
        # Here: just keep top vocab_size by frequency

        print(f"\nPruning to target size ({self.vocab_size})...")

        # Sort by frequency
        sorted_vocab = sorted(
            self.vocab.items(),
            key=lambda x: substring_counts[x[0]],
            reverse=True
        )

        # Keep top vocab_size
        self.vocab = dict(sorted_vocab[:self.vocab_size])

        print(f"Final vocabulary size: {len(self.vocab)}")

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        # Add space prefix to words
        words = [f"{self.space_token}{word}" for word in text.split()]

        # Tokenize each word
        tokens = []
        for word in words:
            tokens.extend(self._viterbi_tokenize(word))

        return tokens

    def encode(self, text: str) -> List[int]:
        """
        Encode text to IDs

        Args:
            text: Input text

        Returns:
            List of token IDs
        """
        tokens = self.tokenize(text)

        # Create ID mapping (sorted for consistency)
        sorted_vocab = sorted(self.vocab.keys())
        token_to_id = {token: idx for idx, token in enumerate(sorted_vocab)}

        return [token_to_id.get(t, 0) for t in tokens]

    def decode(self, token_ids: List[int]) -> str:
        """
        Decode IDs to text

        Args:
            token_ids: List of token IDs

        Returns:
            Decoded text
        """
        # Create ID mapping
        sorted_vocab = sorted(self.vocab.keys())
        id_to_token = {idx: token for idx, token in enumerate(sorted_vocab)}

        # Convert to tokens
        tokens = [id_to_token.get(id, "") for id in token_ids]

        # Join and remove ▁
        text = "".join(tokens).replace(self.space_token, " ")

        return text.strip()


# Demo
if __name__ == "__main__":
    print("\n" + "="*80)
    print("SENTENCEPIECE TOKENIZER DEMO")
    print("="*80)

    # Training corpus
    corpus = [
        "the quick brown fox",
        "the lazy dog",
        "quick brown animals"
    ]

    print("\nTraining corpus:")
    for text in corpus:
        print(f"  '{text}'")

    # Train
    tokenizer = SentencePieceTokenizer(vocab_size=100)
    tokenizer.train(corpus, seed_size=500)

    # Test
    print("\n" + "="*80)
    print("TOKENIZATION EXAMPLES")
    print("="*80)

    test_texts = [
        "the quick fox",
        "lazy dog",
        "brown animals"
    ]

    for text in test_texts:
        tokens = tokenizer.tokenize(text)
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)

        print(f"\nText: '{text}'")
        print(f"  Tokens: {tokens}")
        print(f"  IDs: {ids}")
        print(f"  Decoded: '{decoded}'")

    # Show space token
    print("\n" + "="*80)
    print("SPACE REPRESENTATION")
    print("="*80)
    print(f"Space token: '{tokenizer.space_token}' (U+2581)")
    print(f"Example: 'hello world' → {tokenizer.tokenize('hello world')}")
```

## 4. Comparaison BPE vs WordPiece vs SentencePiece

```python
"""
Résumé des différences
"""

@dataclass
class TokenizerComparison:
    """Comparison of tokenizer algorithms"""
    name: str
    algorithm: str
    merge_criterion: str
    space_handling: str
    reversible: bool
    used_in: List[str]
    pros: List[str]
    cons: List[str]


TOKENIZER_ALGORITHMS = [
    TokenizerComparison(
        name="BPE (Byte Pair Encoding)",
        algorithm="Greedy merging of most frequent pairs",
        merge_criterion="Frequency",
        space_handling="Special token (e.g., Ġ in GPT-2)",
        reversible=True,
        used_in=["GPT-2", "GPT-3", "GPT-4", "RoBERTa", "BART"],
        pros=[
            "Simple algorithm",
            "Works well in practice",
            "Fast training and inference"
        ],
        cons=[
            "May split common words oddly",
            "Frequency-based (not optimal for likelihood)"
        ]
    ),

    TokenizerComparison(
        name="WordPiece",
        algorithm="Merging based on likelihood",
        merge_criterion="Likelihood: P(ab) / (P(a) × P(b))",
        space_handling="## prefix for continuations",
        reversible=True,
        used_in=["BERT", "DistilBERT", "ELECTRA"],
        pros=[
            "Likelihood-based (theoretically better)",
            "Good for BERT-style models"
        ],
        cons=[
            "Slightly more complex than BPE",
            "In practice, similar to BPE"
        ]
    ),

    TokenizerComparison(
        name="SentencePiece (Unigram)",
        algorithm="Probabilistic model, iterative pruning",
        merge_criterion="Maximum likelihood",
        space_handling="▁ (U+2581) character",
        reversible=True,
        used_in=["T5", "mT5", "ALBERT", "XLNet", "Llama 2"],
        pros=[
            "Language-agnostic (no pre-tokenization)",
            "Perfectly reversible",
            "Works for any language",
            "Can use BPE or Unigram algorithm"
        ],
        cons=[
            "More complex implementation",
            "Requires external library in practice"
        ]
    )
]


def print_algorithm_comparison():
    """
    Print detailed comparison of tokenization algorithms
    """
    print("="*100)
    print("TOKENIZATION ALGORITHMS COMPARISON")
    print("="*100)

    for algo in TOKENIZER_ALGORITHMS:
        print(f"\n{'='*100}")
        print(f"{algo.name}")
        print('='*100)
        print(f"Algorithm: {algo.algorithm}")
        print(f"Merge Criterion: {algo.merge_criterion}")
        print(f"Space Handling: {algo.space_handling}")
        print(f"Reversible: {algo.reversible}")

        print("\nPros:")
        for pro in algo.pros:
            print(f"  ✅ {pro}")

        print("\nCons:")
        for con in algo.cons:
            print(f"  ⚠️  {con}")

        print(f"\nUsed in: {', '.join(algo.used_in)}")


if __name__ == "__main__":
    print_algorithm_comparison()

    print("\n" + "="*100)
    print("PRACTICAL RECOMMENDATIONS")
    print("="*100)
    print("""
For Production:

1. Use Pre-trained Tokenizers
   → Don't train from scratch unless you have very specific needs
   → Use from transformers import AutoTokenizer

2. Choose Based on Model:
   • GPT-style: Use GPT-2 BPE tokenizer
   • BERT-style: Use BERT WordPiece tokenizer
   • T5-style: Use T5 SentencePiece tokenizer
   • Llama: Use Llama SentencePiece tokenizer

3. For New Projects:
   → SentencePiece (most flexible, language-agnostic)
   → Vocabulary size: 32k-50k for most tasks

4. Special Tokens:
   Always include: [PAD], [UNK], [CLS], [SEP], [MASK]
   For generation: [BOS], [EOS]

Example (HuggingFace):
  from transformers import AutoTokenizer

  # Load pre-trained tokenizer
  tokenizer = AutoTokenizer.from_pretrained("gpt2")

  # Tokenize
  tokens = tokenizer.tokenize("Hello world")
  ids = tokenizer.encode("Hello world")

  # Decode
  text = tokenizer.decode(ids)
    """)
```

*[Suite avec Token Embeddings et projet complet dans la partie 3...]*
