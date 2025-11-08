# Chapitre 11 (Partie 2): Deduplication et Quality Filtering

## 3. Deduplication

```python
"""
Deduplication = Remove duplicate documents

Pourquoi dédupliquer?
  • Common Crawl: ~30% duplicates exactes
  • Near-duplicates: ~50% total
  • Impact sur training:
    - Overfitting sur contenus répétés
    - Waste de compute
    - Biased models

Méthodes:
  1. Exact deduplication (hash-based)
     • Fast: O(n)
     • Simple: SHA256 hash
     • Trouve: Copies exactes

  2. Near-deduplication (MinHash LSH)
     • Slower: O(n log n)
     • Complex: Jaccard similarity
     • Trouve: Similar documents (90%+ overlap)

  3. Line-level deduplication
     • Remove répétitions within documents
     • Used by C4, RefinedWeb

Résultats:
  • GPT-3: Removed ~13% duplicates
  • Llama 2: Aggressive dedup (30%+ removed)
  • Improves quality significantly
"""

import hashlib
from typing import Set, List, Dict, Tuple
from collections import defaultdict
import mmh3  # MurmurHash3 for MinHash
import numpy as np
from dataclasses import dataclass
import re


class ExactDeduplicator:
    """
    Exact deduplication using hashing

    Example:
        >>> dedup = ExactDeduplicator()
        >>> docs = ["hello world", "hello world", "goodbye"]
        >>> unique = dedup.deduplicate(docs)
        >>> print(len(unique))  # 2
    """

    def __init__(self, hash_algorithm: str = 'sha256'):
        """
        Args:
            hash_algorithm: Hash algorithm (sha256, md5, etc.)
        """
        self.hash_algorithm = hash_algorithm
        self.seen_hashes: Set[str] = set()

    def get_hash(self, text: str) -> str:
        """
        Get hash of text

        Args:
            text: Input text

        Returns:
            Hash string
        """
        if self.hash_algorithm == 'sha256':
            return hashlib.sha256(text.encode('utf-8')).hexdigest()
        elif self.hash_algorithm == 'md5':
            return hashlib.md5(text.encode('utf-8')).hexdigest()
        else:
            raise ValueError(f"Unknown hash algorithm: {self.hash_algorithm}")

    def is_duplicate(self, text: str) -> bool:
        """
        Check if text is duplicate

        Args:
            text: Input text

        Returns:
            True if duplicate
        """
        text_hash = self.get_hash(text)

        if text_hash in self.seen_hashes:
            return True

        self.seen_hashes.add(text_hash)
        return False

    def deduplicate(self, texts: List[str]) -> List[str]:
        """
        Deduplicate list of texts

        Args:
            texts: Input texts

        Returns:
            Unique texts
        """
        unique = []

        for text in texts:
            if not self.is_duplicate(text):
                unique.append(text)

        return unique

    def deduplicate_jsonl(
        self,
        input_path: str,
        output_path: str,
        text_field: str = 'text'
    ):
        """
        Deduplicate JSONL file

        Args:
            input_path: Input JSONL
            output_path: Output JSONL
            text_field: Field containing text
        """
        print(f"Exact deduplication: {input_path} → {output_path}")

        total = 0
        unique = 0

        with open(input_path) as f_in, open(output_path, 'w') as f_out:
            for line in f_in:
                total += 1

                doc = json.loads(line)
                text = doc.get(text_field, '')

                if not self.is_duplicate(text):
                    f_out.write(line)
                    unique += 1

                if total % 10000 == 0:
                    dup_rate = (total - unique) / total * 100
                    print(f"  Processed {total:,} ({dup_rate:.1f}% duplicates)")

        dup_rate = (total - unique) / total * 100

        print(f"\n{'='*60}")
        print(f"Total: {total:,}")
        print(f"Unique: {unique:,}")
        print(f"Duplicates: {total-unique:,} ({dup_rate:.1f}%)")
        print(f"{'='*60}")


class MinHashDeduplicator:
    """
    Near-deduplication using MinHash + LSH

    MinHash = Estimate Jaccard similarity efficiently
    LSH = Locality Sensitive Hashing (find similar docs fast)

    Example:
        >>> dedup = MinHashDeduplicator(num_perm=128, threshold=0.8)
        >>> docs = ["hello world", "hello earth", "goodbye"]
        >>> unique = dedup.deduplicate(docs)
    """

    def __init__(
        self,
        num_perm: int = 128,
        threshold: float = 0.8,
        ngram_size: int = 5
    ):
        """
        Args:
            num_perm: Number of permutations (higher = more accurate)
            threshold: Jaccard similarity threshold (0.8 = 80% overlap)
            ngram_size: N-gram size for shingling
        """
        self.num_perm = num_perm
        self.threshold = threshold
        self.ngram_size = ngram_size

        # LSH buckets
        self.lsh_buckets: Dict[int, List[int]] = defaultdict(list)

    def get_shingles(self, text: str) -> Set[str]:
        """
        Get character n-grams (shingles)

        Args:
            text: Input text

        Returns:
            Set of n-grams
        """
        # Clean and lowercase
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)

        # Generate n-grams
        shingles = set()
        for i in range(len(text) - self.ngram_size + 1):
            shingle = text[i:i + self.ngram_size]
            shingles.add(shingle)

        return shingles

    def get_minhash(self, shingles: Set[str]) -> np.ndarray:
        """
        Compute MinHash signature

        Args:
            shingles: Set of shingles

        Returns:
            MinHash signature (num_perm,)
        """
        # Initialize signature with max values
        signature = np.full(self.num_perm, np.iinfo(np.int32).max, dtype=np.int32)

        # For each shingle
        for shingle in shingles:
            # Hash with different seeds
            for i in range(self.num_perm):
                hash_value = mmh3.hash(shingle, seed=i, signed=False)
                signature[i] = min(signature[i], hash_value)

        return signature

    def estimate_jaccard(self, sig1: np.ndarray, sig2: np.ndarray) -> float:
        """
        Estimate Jaccard similarity from MinHash signatures

        Args:
            sig1: First signature
            sig2: Second signature

        Returns:
            Estimated Jaccard similarity
        """
        return np.sum(sig1 == sig2) / len(sig1)

    def deduplicate(self, texts: List[str]) -> List[str]:
        """
        Deduplicate texts using MinHash + LSH

        Args:
            texts: Input texts

        Returns:
            Unique texts
        """
        print(f"MinHash deduplication (threshold={self.threshold})")

        # Compute signatures
        signatures = []
        for i, text in enumerate(texts):
            shingles = self.get_shingles(text)
            signature = self.get_minhash(shingles)
            signatures.append(signature)

            if (i + 1) % 1000 == 0:
                print(f"  Computed {i+1}/{len(texts)} signatures")

        # Find duplicates using LSH
        is_duplicate = np.zeros(len(texts), dtype=bool)

        # Simple LSH: Use first few hashes as bucket key
        bands = 16
        rows = self.num_perm // bands

        for band_idx in range(bands):
            buckets = defaultdict(list)

            # Hash signatures into buckets
            for doc_idx, sig in enumerate(signatures):
                if is_duplicate[doc_idx]:
                    continue

                # Use band of signature as bucket key
                start = band_idx * rows
                end = start + rows
                band = tuple(sig[start:end])
                bucket_hash = hash(band)

                buckets[bucket_hash].append(doc_idx)

            # Check candidates in same bucket
            for bucket_docs in buckets.values():
                if len(bucket_docs) <= 1:
                    continue

                # Compare all pairs in bucket
                for i in range(len(bucket_docs)):
                    if is_duplicate[bucket_docs[i]]:
                        continue

                    for j in range(i + 1, len(bucket_docs)):
                        if is_duplicate[bucket_docs[j]]:
                            continue

                        # Compute exact similarity
                        sim = self.estimate_jaccard(
                            signatures[bucket_docs[i]],
                            signatures[bucket_docs[j]]
                        )

                        if sim >= self.threshold:
                            # Mark later document as duplicate
                            is_duplicate[bucket_docs[j]] = True

        # Return unique documents
        unique = [text for i, text in enumerate(texts) if not is_duplicate[i]]

        dup_count = np.sum(is_duplicate)
        print(f"\n  Found {dup_count} duplicates ({dup_count/len(texts)*100:.1f}%)")

        return unique


# Demo
if __name__ == "__main__":
    print("="*80)
    print("DEDUPLICATION")
    print("="*80)

    # Test exact deduplication
    print("\n1. EXACT DEDUPLICATION")
    print("-"*80)

    docs_exact = [
        "The quick brown fox jumps over the lazy dog",
        "The quick brown fox jumps over the lazy dog",  # Exact duplicate
        "A different sentence entirely",
        "The quick brown fox jumps over the lazy dog",  # Another duplicate
    ]

    exact_dedup = ExactDeduplicator()
    unique_exact = exact_dedup.deduplicate(docs_exact)

    print(f"Original: {len(docs_exact)} documents")
    print(f"Unique: {len(unique_exact)} documents")

    # Test MinHash deduplication
    print("\n2. MINHASH NEAR-DEDUPLICATION")
    print("-"*80)

    docs_fuzzy = [
        "The quick brown fox jumps over the lazy dog",
        "The quick brown fox jumps over a lazy dog",  # Very similar
        "A completely different sentence here",
        "The quick brown fox leaps over the lazy dog",  # Similar
        "Unrelated content goes here",
    ]

    minhash_dedup = MinHashDeduplicator(num_perm=128, threshold=0.8)
    unique_fuzzy = minhash_dedup.deduplicate(docs_fuzzy)

    print(f"Original: {len(docs_fuzzy)} documents")
    print(f"Unique: {len(unique_fuzzy)} documents")
    print(f"\nKept documents:")
    for i, doc in enumerate(unique_fuzzy):
        print(f"  {i+1}. {doc[:60]}...")
```

## 4. PII Removal

```python
"""
PII (Personally Identifiable Information) = Data that identifies individuals

Types de PII:
  • Names: John Smith
  • Email: john@example.com
  • Phone: (555) 123-4567
  • SSN: 123-45-6789
  • Credit cards: 4111-1111-1111-1111
  • IP addresses: 192.168.1.1
  • Addresses: 123 Main St

Pourquoi remove PII?
  • Privacy concerns (GDPR, CCPA)
  • Security (prevent leaks)
  • Legal compliance
  • Ethical AI

Méthodes:
  1. Regex-based (fast, simple)
  2. NER-based (spaCy, Flair) - more accurate
  3. Hybrid approach

Example tools:
  • Microsoft Presidio
  • spaCy NER
  • Custom regex patterns
"""

import re
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class PIIPattern:
    """PII pattern definition"""
    name: str
    pattern: str
    replacement: str


class PIIRemover:
    """
    Remove PII from text using regex patterns

    Example:
        >>> remover = PIIRemover()
        >>> text = "Contact John at john@email.com or 555-1234"
        >>> clean = remover.remove_pii(text)
        >>> print(clean)  # "Contact [NAME] at [EMAIL] or [PHONE]"
    """

    # Common PII patterns
    PATTERNS = [
        PIIPattern(
            name="email",
            pattern=r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            replacement='[EMAIL]'
        ),
        PIIPattern(
            name="phone_us",
            pattern=r'\b(?:\+?1[-.]?)?\(?([0-9]{3})\)?[-.]?([0-9]{3})[-.]?([0-9]{4})\b',
            replacement='[PHONE]'
        ),
        PIIPattern(
            name="ssn",
            pattern=r'\b\d{3}-\d{2}-\d{4}\b',
            replacement='[SSN]'
        ),
        PIIPattern(
            name="credit_card",
            pattern=r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            replacement='[CREDIT_CARD]'
        ),
        PIIPattern(
            name="ip_address",
            pattern=r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
            replacement='[IP]'
        ),
        PIIPattern(
            name="url",
            pattern=r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            replacement='[URL]'
        ),
    ]

    def __init__(self, patterns: Optional[List[PIIPattern]] = None):
        """
        Args:
            patterns: Custom PII patterns (uses defaults if None)
        """
        self.patterns = patterns or self.PATTERNS

    def remove_pii(self, text: str, keep_stats: bool = False) -> Tuple[str, Optional[Dict]]:
        """
        Remove PII from text

        Args:
            text: Input text
            keep_stats: Return statistics about removed PII

        Returns:
            (cleaned_text, stats_dict)
        """
        stats = defaultdict(int) if keep_stats else None

        for pattern in self.patterns:
            # Count matches
            if keep_stats:
                matches = re.findall(pattern.pattern, text, re.IGNORECASE)
                stats[pattern.name] = len(matches)

            # Replace
            text = re.sub(pattern.pattern, pattern.replacement, text, flags=re.IGNORECASE)

        return text, dict(stats) if stats else None

    def remove_pii_jsonl(
        self,
        input_path: str,
        output_path: str,
        text_field: str = 'text'
    ):
        """
        Remove PII from JSONL file

        Args:
            input_path: Input JSONL
            output_path: Output JSONL
            text_field: Field containing text
        """
        print(f"PII removal: {input_path} → {output_path}")

        total = 0
        total_pii_removed = defaultdict(int)

        with open(input_path) as f_in, open(output_path, 'w') as f_out:
            for line in f_in:
                total += 1

                doc = json.loads(line)
                text = doc.get(text_field, '')

                # Remove PII
                clean_text, stats = self.remove_pii(text, keep_stats=True)

                # Update stats
                for pii_type, count in stats.items():
                    total_pii_removed[pii_type] += count

                # Save
                doc[text_field] = clean_text
                f_out.write(json.dumps(doc) + '\n')

                if total % 10000 == 0:
                    print(f"  Processed {total:,} documents")

        # Print stats
        print(f"\n{'='*60}")
        print("PII REMOVAL STATS")
        print(f"{'='*60}")
        print(f"Documents processed: {total:,}")
        print(f"\nPII found and removed:")
        for pii_type, count in sorted(total_pii_removed.items()):
            print(f"  {pii_type}: {count:,}")
        print(f"{'='*60}")


class NERPIIRemover:
    """
    Remove PII using Named Entity Recognition (more accurate)

    Requires: spaCy with trained model
    pip install spacy
    python -m spacy download en_core_web_lg

    Example:
        >>> remover = NERPIIRemover()
        >>> text = "John Smith works at Microsoft in Seattle"
        >>> clean = remover.remove_pii(text)
    """

    def __init__(self, model_name: str = "en_core_web_lg"):
        """
        Args:
            model_name: spaCy model name
        """
        try:
            import spacy
            self.nlp = spacy.load(model_name)
        except ImportError:
            raise ImportError("Install spaCy: pip install spacy")
        except OSError:
            raise OSError(f"Download model: python -m spacy download {model_name}")

        # Entity types to remove
        self.pii_entity_types = {
            'PERSON',      # People names
            'ORG',         # Organizations (optional)
            'GPE',         # Geo-political entities (optional)
            'LOC',         # Locations (optional)
            'DATE',        # Dates (optional - might want to keep)
        }

    def remove_pii(self, text: str, entity_types: Optional[Set[str]] = None) -> str:
        """
        Remove PII using NER

        Args:
            text: Input text
            entity_types: Set of entity types to remove (uses defaults if None)

        Returns:
            Cleaned text
        """
        entity_types = entity_types or self.pii_entity_types

        # Parse text
        doc = self.nlp(text)

        # Replace entities
        result = text
        offset = 0

        for ent in doc.ents:
            if ent.label_ in entity_types:
                # Calculate position with offset
                start = ent.start_char + offset
                end = ent.end_char + offset

                # Replacement
                replacement = f"[{ent.label_}]"

                # Replace
                result = result[:start] + replacement + result[end:]

                # Update offset
                offset += len(replacement) - (end - start)

        return result


# Demo
if __name__ == "__main__":
    print("="*80)
    print("PII REMOVAL")
    print("="*80)

    # Test text with PII
    test_text = """
    Contact John Smith at john.smith@email.com or call (555) 123-4567.
    His SSN is 123-45-6789 and credit card is 4111-1111-1111-1111.
    You can also reach him at http://example.com or IP 192.168.1.1.
    He lives at 123 Main Street, Seattle, WA 98101.
    """

    print("ORIGINAL TEXT:")
    print(test_text)

    # Regex-based removal
    print("\n" + "="*80)
    print("REGEX-BASED PII REMOVAL")
    print("="*80)

    remover = PIIRemover()
    clean_text, stats = remover.remove_pii(test_text, keep_stats=True)

    print("\nCLEANED TEXT:")
    print(clean_text)

    print("\nPII FOUND:")
    for pii_type, count in stats.items():
        print(f"  {pii_type}: {count}")

    # NER-based removal (if spaCy available)
    print("\n" + "="*80)
    print("NER-BASED PII REMOVAL")
    print("="*80)

    try:
        ner_remover = NERPIIRemover()
        ner_clean = ner_remover.remove_pii(test_text)
        print("\nCLEANED TEXT:")
        print(ner_clean)
    except (ImportError, OSError) as e:
        print(f"Skipped (install spaCy): {e}")
```

## 5. Quality Filtering

```python
"""
Quality Filtering = Keep only high-quality documents

Heuristics (fast):
  • Length (too short / too long)
  • Word count
  • Average word length
  • Special character ratio
  • Uppercase ratio
  • Bullet point ratio
  • Line length statistics
  • Repeated n-gram ratio

ML-based (slow but accurate):
  • Classifier trained on good/bad examples
  • Language model perplexity
  • Used by RefinedWeb, C4

Results:
  • Can filter 70-90% of data
  • Huge quality improvement
  • Worth the compute cost
"""


class QualityFilter:
    """
    Filter documents by quality heuristics

    Example:
        >>> filter = QualityFilter()
        >>> is_good = filter.is_high_quality("This is a good document...")
        >>> print(is_good)
    """

    def __init__(
        self,
        min_words: int = 50,
        max_words: int = 100000,
        min_avg_word_length: float = 3.0,
        max_avg_word_length: float = 10.0,
        max_special_char_ratio: float = 0.3,
        max_uppercase_ratio: float = 0.3,
        max_ellipsis_ratio: float = 0.1,
        max_bullet_ratio: float = 0.5,
        min_alpha_ratio: float = 0.6
    ):
        """
        Args:
            min_words: Minimum word count
            max_words: Maximum word count
            min_avg_word_length: Minimum average word length
            max_avg_word_length: Maximum average word length
            max_special_char_ratio: Maximum ratio of special characters
            max_uppercase_ratio: Maximum ratio of uppercase letters
            max_ellipsis_ratio: Maximum ratio of ellipsis (...)
            max_bullet_ratio: Maximum ratio of lines starting with bullets
            min_alpha_ratio: Minimum ratio of alphabetic characters
        """
        self.min_words = min_words
        self.max_words = max_words
        self.min_avg_word_length = min_avg_word_length
        self.max_avg_word_length = max_avg_word_length
        self.max_special_char_ratio = max_special_char_ratio
        self.max_uppercase_ratio = max_uppercase_ratio
        self.max_ellipsis_ratio = max_ellipsis_ratio
        self.max_bullet_ratio = max_bullet_ratio
        self.min_alpha_ratio = min_alpha_ratio

    def is_high_quality(self, text: str) -> Tuple[bool, Dict[str, any]]:
        """
        Check if document is high quality

        Args:
            text: Input text

        Returns:
            (is_high_quality, metrics_dict)
        """
        metrics = self.compute_metrics(text)

        # Check all criteria
        checks = {
            'word_count': self.min_words <= metrics['word_count'] <= self.max_words,
            'avg_word_length': self.min_avg_word_length <= metrics['avg_word_length'] <= self.max_avg_word_length,
            'special_char_ratio': metrics['special_char_ratio'] <= self.max_special_char_ratio,
            'uppercase_ratio': metrics['uppercase_ratio'] <= self.max_uppercase_ratio,
            'ellipsis_ratio': metrics['ellipsis_ratio'] <= self.max_ellipsis_ratio,
            'bullet_ratio': metrics['bullet_ratio'] <= self.max_bullet_ratio,
            'alpha_ratio': metrics['alpha_ratio'] >= self.min_alpha_ratio,
        }

        is_good = all(checks.values())

        return is_good, {'metrics': metrics, 'checks': checks}

    def compute_metrics(self, text: str) -> Dict[str, float]:
        """
        Compute quality metrics

        Args:
            text: Input text

        Returns:
            Metrics dict
        """
        # Word-level metrics
        words = text.split()
        word_count = len(words)

        if word_count > 0:
            avg_word_length = sum(len(w) for w in words) / word_count
        else:
            avg_word_length = 0

        # Character-level metrics
        char_count = len(text)

        if char_count > 0:
            alpha_count = sum(c.isalpha() for c in text)
            upper_count = sum(c.isupper() for c in text)
            special_count = sum(not c.isalnum() and not c.isspace() for c in text)

            alpha_ratio = alpha_count / char_count
            uppercase_ratio = upper_count / max(alpha_count, 1)
            special_char_ratio = special_count / char_count
        else:
            alpha_ratio = 0
            uppercase_ratio = 0
            special_char_ratio = 0

        # Line-level metrics
        lines = text.split('\n')
        line_count = len(lines)

        if line_count > 0:
            bullet_lines = sum(
                1 for line in lines
                if line.strip() and line.strip()[0] in '•·-*►▪■'
            )
            bullet_ratio = bullet_lines / line_count
        else:
            bullet_ratio = 0

        # Other metrics
        ellipsis_count = text.count('...')
        ellipsis_ratio = ellipsis_count / max(word_count, 1)

        return {
            'word_count': word_count,
            'char_count': char_count,
            'avg_word_length': avg_word_length,
            'alpha_ratio': alpha_ratio,
            'uppercase_ratio': uppercase_ratio,
            'special_char_ratio': special_char_ratio,
            'bullet_ratio': bullet_ratio,
            'ellipsis_ratio': ellipsis_ratio,
        }

    def filter_jsonl(
        self,
        input_path: str,
        output_path: str,
        text_field: str = 'text'
    ):
        """
        Filter JSONL file by quality

        Args:
            input_path: Input JSONL
            output_path: Output JSONL
            text_field: Field containing text
        """
        print(f"Quality filtering: {input_path} → {output_path}")

        total = 0
        kept = 0
        rejection_reasons = defaultdict(int)

        with open(input_path) as f_in, open(output_path, 'w') as f_out:
            for line in f_in:
                total += 1

                doc = json.loads(line)
                text = doc.get(text_field, '')

                # Check quality
                is_good, details = self.is_high_quality(text)

                if is_good:
                    f_out.write(line)
                    kept += 1
                else:
                    # Track rejection reasons
                    for check, passed in details['checks'].items():
                        if not passed:
                            rejection_reasons[check] += 1

                if total % 10000 == 0:
                    keep_rate = kept / total * 100
                    print(f"  Processed {total:,} ({keep_rate:.1f}% kept)")

        # Print stats
        keep_rate = kept / total * 100

        print(f"\n{'='*60}")
        print("QUALITY FILTERING STATS")
        print(f"{'='*60}")
        print(f"Total: {total:,}")
        print(f"Kept: {kept:,} ({keep_rate:.1f}%)")
        print(f"Rejected: {total-kept:,} ({100-keep_rate:.1f}%)")
        print(f"\nTop rejection reasons:")
        for reason, count in sorted(rejection_reasons.items(), key=lambda x: -x[1])[:5]:
            print(f"  {reason}: {count:,}")
        print(f"{'='*60}")


# Demo
if __name__ == "__main__":
    print("="*80)
    print("QUALITY FILTERING")
    print("="*80)

    # Test documents
    docs = [
        {
            'text': "This is a high-quality document with proper sentences and good structure. " * 10,
            'label': 'GOOD'
        },
        {
            'text': "SPAM SPAM SPAM !!! CLICK HERE !!! FREE MONEY !!!",
            'label': 'BAD (uppercase, special chars)'
        },
        {
            'text': "a b c d e",
            'label': 'BAD (too short)'
        },
        {
            'text': "Word " * 10000,
            'label': 'BAD (too long)'
        },
        {
            'text': "This has... way too many... ellipses... everywhere... " * 5,
            'label': 'BAD (ellipsis)'
        },
    ]

    filter = QualityFilter()

    for doc in docs:
        is_good, details = filter.is_high_quality(doc['text'])

        print(f"\n{'-'*80}")
        print(f"Document: {doc['text'][:60]}...")
        print(f"Expected: {doc['label']}")
        print(f"Result: {'✅ PASS' if is_good else '❌ REJECT'}")

        if not is_good:
            print(f"Failed checks:")
            for check, passed in details['checks'].items():
                if not passed:
                    metric_value = details['metrics'].get(check.replace('_ratio', ''))
                    print(f"  • {check}: {metric_value}")

    print("\n" + "="*80)
    print("READY FOR BATCH FILTERING")
    print("="*80)
    print("""
Usage:
    filter = QualityFilter(
        min_words=50,
        max_special_char_ratio=0.3,
        max_uppercase_ratio=0.3
    )

    filter.filter_jsonl(
        input_path="preprocessed.jsonl",
        output_path="high_quality.jsonl"
    )
    """)
```

*[Suite avec Synthetic Data et Pipeline Complet dans la partie 3...]*
