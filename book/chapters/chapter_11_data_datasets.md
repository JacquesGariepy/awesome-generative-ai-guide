# Chapitre 11: Données et Datasets pour Pré-entraînement

## Introduction

La **qualité des données** est l'élément le plus critique pour le succès d'un LLM. Un modèle entraîné sur des données de mauvaise qualité ne pourra jamais atteindre de bonnes performances, peu importe l'architecture ou les ressources compute.

### Le Pipeline de Données

```python
"""
Pipeline de Données = Data Sources → Cleaning → Quality → Training

Le paradoxe de la qualité des données:
  • GPT-3: Entraîné sur ~570GB de texte (filtré de 45TB!)
  • Llama 2: 2TB de données (filtré de ~15TB)
  • Ratio signal/noise: Seulement 5-10% des données brutes sont gardées

Étapes critiques:
  1. Collection: Collecter données brutes (web, livres, code)
  2. Extraction: Extraire texte propre (HTML → text)
  3. Filtering: Filtrer qualité (heuristiques, ML)
  4. Deduplication: Enlever doublons (exact, fuzzy)
  5. PII Removal: Enlever infos personnelles
  6. Mixing: Mélanger sources avec bons ratios
  7. Tokenization: Préparer pour entraînement

Coût caché:
  • Collection: Plusieurs semaines
  • Preprocessing: Plusieurs mois de compute
  • Storage: Plusieurs TB
  • Total: Souvent plus cher que l'entraînement lui-même!
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Iterator
import re
import json
from pathlib import Path
from collections import Counter
import hashlib
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
import ftfy
import unicodedata


@dataclass
class DatasetStats:
    """Statistics for a dataset"""
    name: str
    size_gb: float
    num_documents: int
    avg_doc_length: int
    languages: List[str]
    domains: List[str]
    quality_score: float


# Famous pretraining datasets
FAMOUS_DATASETS = [
    DatasetStats(
        name="Common Crawl (raw)",
        size_gb=250000,  # 250TB per monthly dump
        num_documents=3_000_000_000,
        avg_doc_length=5000,
        languages=["all"],
        domains=["web"],
        quality_score=0.3  # Lots of spam, ads, etc.
    ),
    DatasetStats(
        name="C4 (Colossal Clean Crawl Corpus)",
        size_gb=750,
        num_documents=365_000_000,
        avg_doc_length=2000,
        languages=["en"],
        domains=["web"],
        quality_score=0.7  # Cleaned Common Crawl
    ),
    DatasetStats(
        name="The Pile",
        size_gb=825,
        num_documents=210_000_000,
        avg_doc_length=4000,
        languages=["en", "code"],
        domains=["web", "books", "github", "arxiv", "pubmed"],
        quality_score=0.8  # High quality, diverse
    ),
    DatasetStats(
        name="RedPajama",
        size_gb=5000,
        num_documents=1_000_000_000,
        avg_doc_length=5000,
        languages=["en", "code"],
        domains=["web", "books", "github", "arxiv", "wikipedia"],
        quality_score=0.75  # Llama-style dataset
    ),
    DatasetStats(
        name="RefinedWeb",
        size_gb=2800,
        num_documents=968_000_000,
        avg_doc_length=3000,
        languages=["en"],
        domains=["web"],
        quality_score=0.85  # Very high quality web data
    ),
]


def print_dataset_comparison():
    """Print comparison of famous pretraining datasets"""
    print("="*100)
    print("FAMOUS PRETRAINING DATASETS")
    print("="*100)

    print(f"\n{'Dataset':<30} {'Size (GB)':<12} {'Documents':<15} {'Quality':<10}")
    print("-"*100)

    for ds in FAMOUS_DATASETS:
        docs_str = f"{ds.num_documents:,}" if ds.num_documents < 1_000_000_000 else f"{ds.num_documents/1e9:.1f}B"
        print(f"{ds.name:<30} {ds.size_gb:<12,.0f} {docs_str:<15} {ds.quality_score:<10.2f}")

    print("\n" + "="*100)
    print("KEY INSIGHTS")
    print("="*100)
    print("""
1. Common Crawl = Source principale
   • 250TB par mois (snapshot mensuel)
   • Mais: 70% spam, ads, low quality
   • Solution: Aggressive filtering

2. Cleaned Datasets (C4, RefinedWeb)
   • 100x plus petits que Common Crawl raw
   • Mais: 10x meilleure qualité
   • Résultat: Meilleurs modèles avec moins de données

3. Diverse Mix (The Pile, RedPajama)
   • Combine web + books + code + scientific
   • Diversité = Robustesse
   • Llama 2, Falcon, MPT utilisent ce pattern

4. Quality > Quantity
   • 1TB de données clean > 10TB de données dirty
   • Filtering peut prendre 90% du temps
   • Mais critical pour performance finale
    """)


if __name__ == "__main__":
    print_dataset_comparison()
```

## 1. Sources de Données

### 1.1 Common Crawl

```python
"""
Common Crawl = Archive du web public

Snapshots:
  • Publiés mensuellement depuis 2008
  • ~250TB par snapshot (format WARC)
  • ~3 milliards de pages web
  • Gratuit et open-source

Format WARC (Web ARChive):
  • Container pour web data
  • Inclut: HTML, HTTP headers, metadata
  • Compressed avec GZIP

Accès:
  • AWS S3: s3://commoncrawl/
  • HTTPS: https://data.commoncrawl.org/
  • Registry: https://index.commoncrawl.org/
"""

import gzip
from warcio.archiveiterator import ArchiveIterator
from typing import Iterator, Dict


class CommonCrawlDownloader:
    """
    Download and process Common Crawl data

    Example:
        >>> downloader = CommonCrawlDownloader()
        >>> # Download single WARC file
        >>> records = downloader.download_warc(
        ...     "crawl-data/CC-MAIN-2023-06/segments/.../warc/..."
        ... )
        >>> for record in records:
        ...     print(record['text'][:100])
    """

    BASE_URL = "https://data.commoncrawl.org/"

    def __init__(self):
        self.session = requests.Session()

    def get_warc_paths(self, crawl_id: str = "CC-MAIN-2023-06", limit: int = 10) -> List[str]:
        """
        Get list of WARC file paths for a crawl

        Args:
            crawl_id: Crawl identifier (e.g., "CC-MAIN-2023-06")
            limit: Maximum number of paths to return

        Returns:
            List of WARC file paths
        """
        warc_paths_url = f"{self.BASE_URL}crawl-data/{crawl_id}/warc.paths.gz"

        print(f"Fetching WARC paths from {warc_paths_url}")

        response = self.session.get(warc_paths_url)
        response.raise_for_status()

        # Decompress and parse
        paths = gzip.decompress(response.content).decode('utf-8').strip().split('\n')

        print(f"Found {len(paths)} WARC files")

        return paths[:limit]

    def download_warc(self, warc_path: str) -> Iterator[Dict]:
        """
        Download and parse a WARC file

        Args:
            warc_path: Path to WARC file (relative to base URL)

        Yields:
            Records with extracted text
        """
        warc_url = f"{self.BASE_URL}{warc_path}"

        print(f"Downloading {warc_url}")

        response = self.session.get(warc_url, stream=True)
        response.raise_for_status()

        # Parse WARC
        for record in ArchiveIterator(response.raw, arc2warc=True):
            if record.rec_type == 'response':
                # Get URL and content
                url = record.rec_headers.get_header('WARC-Target-URI')
                content_type = record.http_headers.get_header('Content-Type', '')

                # Only process HTML
                if 'html' in content_type.lower():
                    # Extract HTML
                    html = record.content_stream().read()

                    # Parse with BeautifulSoup
                    text = self._extract_text_from_html(html)

                    if text:
                        yield {
                            'url': url,
                            'text': text,
                            'content_type': content_type
                        }

    def _extract_text_from_html(self, html: bytes) -> str:
        """
        Extract clean text from HTML

        Args:
            html: Raw HTML bytes

        Returns:
            Extracted text
        """
        try:
            soup = BeautifulSoup(html, 'lxml')

            # Remove script and style elements
            for script in soup(["script", "style", "header", "footer", "nav"]):
                script.decompose()

            # Get text
            text = soup.get_text()

            # Clean whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)

            return text

        except Exception as e:
            print(f"Error extracting text: {e}")
            return ""

    def process_crawl(
        self,
        crawl_id: str = "CC-MAIN-2023-06",
        num_warcs: int = 2,
        output_path: str = "common_crawl_data.jsonl"
    ):
        """
        Process a crawl and save to JSONL

        Args:
            crawl_id: Crawl identifier
            num_warcs: Number of WARC files to process
            output_path: Output JSONL file path
        """
        print("="*80)
        print(f"PROCESSING COMMON CRAWL: {crawl_id}")
        print("="*80)

        # Get WARC paths
        warc_paths = self.get_warc_paths(crawl_id, limit=num_warcs)

        total_docs = 0

        with open(output_path, 'w') as f:
            for warc_path in warc_paths:
                print(f"\nProcessing {warc_path}")

                for record in self.download_warc(warc_path):
                    # Save to JSONL
                    f.write(json.dumps(record) + '\n')
                    total_docs += 1

                    if total_docs % 100 == 0:
                        print(f"  Processed {total_docs} documents")

        print(f"\n✅ Saved {total_docs} documents to {output_path}")


# Demo (requires warcio, beautifulsoup4, lxml)
def demo_common_crawl():
    """Demo Common Crawl download"""
    print("""
To use Common Crawl:

1. Install dependencies:
   pip install warcio beautifulsoup4 lxml requests

2. Download sample:
   downloader = CommonCrawlDownloader()
   downloader.process_crawl(
       crawl_id="CC-MAIN-2023-06",
       num_warcs=1,  # Start small!
       output_path="cc_sample.jsonl"
   )

3. Warning:
   • Each WARC ~1GB compressed
   • Full crawl = 250TB
   • Use AWS S3 or local filtering first

4. Better alternative:
   • Use pre-filtered datasets (C4, RefinedWeb)
   • Or Common Crawl News (smaller, cleaner)
    """)
    print("Common Crawl downloader ready (install deps first)")


if __name__ == "__main__":
    demo_common_crawl()
```

### 1.2 Books et Wikipedia

```python
"""
Books = High-quality long-form text
Wikipedia = Clean, factual, multilingual

Sources:
  Books:
    • Project Gutenberg: 70k free books (public domain)
    • Books1/Books2: Used by GPT-3 (not public)
    • Books3: The Pile component (controversial copyright)

  Wikipedia:
    • Dumps: https://dumps.wikimedia.org/
    • ~6M articles (English)
    • ~300M articles (all languages)
    • Updated monthly
    • Very clean, well-structured
"""

import requests
import bz2
import xml.etree.ElementTree as ET
from typing import Iterator


class WikipediaDownloader:
    """
    Download and process Wikipedia dumps

    Example:
        >>> wiki = WikipediaDownloader()
        >>> articles = wiki.download_latest_dump(language='en', max_articles=1000)
        >>> for article in articles:
        ...     print(article['title'], len(article['text']))
    """

    DUMPS_URL = "https://dumps.wikimedia.org/{lang}wiki/latest/"

    def __init__(self):
        self.session = requests.Session()

    def get_dump_url(self, language: str = 'en') -> str:
        """
        Get URL for latest Wikipedia dump

        Args:
            language: Language code (en, fr, es, etc.)

        Returns:
            URL to latest dump
        """
        # Use articles-multistream (compressed XML)
        dump_file = f"{language}wiki-latest-pages-articles-multistream.xml.bz2"
        url = self.DUMPS_URL.format(lang=language) + dump_file

        return url

    def download_latest_dump(
        self,
        language: str = 'en',
        max_articles: Optional[int] = None,
        output_path: Optional[str] = None
    ) -> Iterator[Dict]:
        """
        Download and parse Wikipedia dump

        Args:
            language: Language code
            max_articles: Maximum number of articles to process
            output_path: Save to JSONL file (optional)

        Yields:
            Article dictionaries
        """
        url = self.get_dump_url(language)

        print(f"Downloading Wikipedia dump: {url}")
        print("Warning: Full dump is ~20GB compressed, ~90GB uncompressed")
        print("Consider using streaming or pre-processed datasets")

        # Stream download
        response = self.session.get(url, stream=True)
        response.raise_for_status()

        # Decompress on-the-fly
        decompressor = bz2.BZ2Decompressor()

        count = 0
        buffer = b""

        file_handle = None
        if output_path:
            file_handle = open(output_path, 'w')

        try:
            for chunk in response.iter_content(chunk_size=1024*1024):  # 1MB chunks
                buffer += decompressor.decompress(chunk)

                # Parse complete pages
                while b'</page>' in buffer:
                    page_end = buffer.index(b'</page>') + len(b'</page>')
                    page_xml = buffer[:page_end]
                    buffer = buffer[page_end:]

                    # Parse page
                    article = self._parse_page(page_xml)

                    if article:
                        count += 1

                        if file_handle:
                            file_handle.write(json.dumps(article) + '\n')

                        yield article

                        if count % 1000 == 0:
                            print(f"Processed {count} articles")

                        if max_articles and count >= max_articles:
                            break

                if max_articles and count >= max_articles:
                    break

        finally:
            if file_handle:
                file_handle.close()

        print(f"\n✅ Processed {count} Wikipedia articles")

    def _parse_page(self, page_xml: bytes) -> Optional[Dict]:
        """
        Parse a Wikipedia page XML

        Args:
            page_xml: Page XML bytes

        Returns:
            Article dict or None
        """
        try:
            # Wrap in root element
            xml_str = b'<root>' + page_xml + b'</root>'
            root = ET.fromstring(xml_str)

            # Extract fields
            page = root.find('page')
            if page is None:
                return None

            title_elem = page.find('title')
            text_elem = page.find('revision/text')

            if title_elem is None or text_elem is None:
                return None

            title = title_elem.text
            text = text_elem.text

            if not text:
                return None

            # Skip redirects and special pages
            if text.startswith('#REDIRECT') or ':' in title:
                return None

            # Clean wiki markup (basic)
            text = self._clean_wiki_markup(text)

            return {
                'title': title,
                'text': text,
                'source': 'wikipedia'
            }

        except Exception as e:
            return None

    def _clean_wiki_markup(self, text: str) -> str:
        """
        Clean Wikipedia markup (basic implementation)

        For production, use mwparserfromhell or wikitextparser

        Args:
            text: Raw wiki text

        Returns:
            Cleaned text
        """
        # Remove common markup (very basic)
        # In production, use proper parser!

        # Remove templates
        text = re.sub(r'\{\{[^}]+\}\}', '', text)

        # Remove references
        text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL)
        text = re.sub(r'<ref[^>]*/>', '', text)

        # Remove comments
        text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)

        # Remove categories
        text = re.sub(r'\[\[Category:[^\]]+\]\]', '', text)

        # Clean links [[link|text]] -> text
        text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', text)

        # Remove bold/italic
        text = re.sub(r"'{2,}", '', text)

        return text.strip()


class GutenbergDownloader:
    """
    Download books from Project Gutenberg

    Example:
        >>> gutenberg = GutenbergDownloader()
        >>> book = gutenberg.download_book(1342)  # Pride and Prejudice
        >>> print(book['title'], len(book['text']))
    """

    BASE_URL = "https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"

    def download_book(self, book_id: int) -> Dict:
        """
        Download a book by ID

        Args:
            book_id: Gutenberg book ID

        Returns:
            Book dict
        """
        url = self.BASE_URL.format(book_id=book_id)

        response = requests.get(url)
        response.raise_for_status()

        # Decode text (Gutenberg uses UTF-8)
        text = response.content.decode('utf-8-sig')

        # Extract title (first non-empty line after header)
        lines = text.split('\n')
        title = "Unknown"
        for line in lines[:50]:
            if line.strip() and not line.startswith('*'):
                title = line.strip()
                break

        return {
            'id': book_id,
            'title': title,
            'text': text,
            'source': 'gutenberg'
        }

    def download_popular_books(self, num_books: int = 100, output_path: str = "gutenberg.jsonl"):
        """
        Download popular books

        Args:
            num_books: Number of books
            output_path: Output path
        """
        # Popular book IDs (top 100)
        popular_ids = [
            1342,  # Pride and Prejudice
            84,    # Frankenstein
            1661,  # Sherlock Holmes
            11,    # Alice in Wonderland
            98,    # Tale of Two Cities
            # ... add more
        ]

        print(f"Downloading {len(popular_ids)} books from Gutenberg...")

        with open(output_path, 'w') as f:
            for book_id in popular_ids[:num_books]:
                try:
                    book = self.download_book(book_id)
                    f.write(json.dumps(book) + '\n')
                    print(f"  ✅ Downloaded: {book['title']}")
                except Exception as e:
                    print(f"  ❌ Failed {book_id}: {e}")

        print(f"\n✅ Saved to {output_path}")


# Demo
if __name__ == "__main__":
    print("="*80)
    print("BOOKS & WIKIPEDIA DOWNLOADERS")
    print("="*80)

    print("""
Wikipedia:
  • Use HuggingFace datasets (easier):
    from datasets import load_dataset
    wiki = load_dataset("wikipedia", "20220301.en", split="train")

  • Or download dumps directly (this class)

Gutenberg:
  • 70k free books (public domain)
  • Good for literary text
  • But: Old books (pre-1928)

Modern Books:
  • Books1/2 used by GPT-3 (not public)
  • Books3 (The Pile) - copyright issues
  • Alternative: Use instruction datasets instead
    """)
```

## 2. Preprocessing et Nettoyage

```python
"""
Text Preprocessing = Clean raw text

Étapes:
  1. Unicode normalization (fix encoding issues)
  2. HTML/XML removal
  3. URL removal
  4. Whitespace normalization
  5. Language detection
  6. Quality filtering

Libraries utiles:
  • ftfy: Fix text encoding issues
  • unicodedata: Unicode normalization
  • langdetect: Language detection
  • trafilatura: HTML extraction
"""


class TextPreprocessor:
    """
    Comprehensive text preprocessing pipeline

    Example:
        >>> preprocessor = TextPreprocessor()
        >>> clean = preprocessor.preprocess("Raw   text\x00 with issues…")
        >>> print(clean)
    """

    def __init__(
        self,
        fix_unicode: bool = True,
        remove_urls: bool = True,
        remove_emails: bool = True,
        normalize_whitespace: bool = True,
        min_length: int = 100
    ):
        self.fix_unicode = fix_unicode
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.normalize_whitespace = normalize_whitespace
        self.min_length = min_length

    def preprocess(self, text: str) -> Optional[str]:
        """
        Preprocess text through full pipeline

        Args:
            text: Raw text

        Returns:
            Cleaned text or None if filtered out
        """
        if not text:
            return None

        # 1. Fix unicode issues
        if self.fix_unicode:
            text = ftfy.fix_text(text)

        # 2. Unicode normalization (NFC = canonical composition)
        text = unicodedata.normalize('NFC', text)

        # 3. Remove null bytes and control characters
        text = self._remove_control_characters(text)

        # 4. Remove URLs
        if self.remove_urls:
            text = self._remove_urls(text)

        # 5. Remove emails
        if self.remove_emails:
            text = self._remove_emails(text)

        # 6. Normalize whitespace
        if self.normalize_whitespace:
            text = self._normalize_whitespace(text)

        # 7. Check minimum length
        if len(text) < self.min_length:
            return None

        return text

    def _remove_control_characters(self, text: str) -> str:
        """Remove control characters except newline and tab"""
        return ''.join(
            char for char in text
            if unicodedata.category(char)[0] != 'C' or char in '\n\t'
        )

    def _remove_urls(self, text: str) -> str:
        """Remove URLs"""
        # Simple regex (not perfect but fast)
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return re.sub(url_pattern, '', text)

    def _remove_emails(self, text: str) -> str:
        """Remove email addresses"""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return re.sub(email_pattern, '', text)

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace"""
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)

        # Replace multiple newlines with double newline
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Remove leading/trailing whitespace per line
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)

        return text.strip()

    def batch_preprocess(
        self,
        input_path: str,
        output_path: str,
        show_stats: bool = True
    ):
        """
        Preprocess a JSONL file

        Args:
            input_path: Input JSONL file
            output_path: Output JSONL file
            show_stats: Print statistics
        """
        print(f"Preprocessing {input_path} → {output_path}")

        total = 0
        kept = 0
        total_chars_before = 0
        total_chars_after = 0

        with open(input_path) as f_in, open(output_path, 'w') as f_out:
            for line in f_in:
                total += 1

                doc = json.loads(line)
                text = doc.get('text', '')

                total_chars_before += len(text)

                # Preprocess
                cleaned = self.preprocess(text)

                if cleaned:
                    kept += 1
                    total_chars_after += len(cleaned)

                    doc['text'] = cleaned
                    f_out.write(json.dumps(doc) + '\n')

                if total % 10000 == 0:
                    print(f"  Processed {total:,} documents ({kept:,} kept)")

        if show_stats:
            print(f"\n{'='*60}")
            print("PREPROCESSING STATS")
            print(f"{'='*60}")
            print(f"Total documents: {total:,}")
            print(f"Kept: {kept:,} ({kept/total*100:.1f}%)")
            print(f"Filtered: {total-kept:,} ({(total-kept)/total*100:.1f}%)")
            print(f"Chars before: {total_chars_before:,}")
            print(f"Chars after: {total_chars_after:,}")
            print(f"Compression: {total_chars_after/total_chars_before*100:.1f}%")


# Demo
if __name__ == "__main__":
    print("="*80)
    print("TEXT PREPROCESSING")
    print("="*80)

    # Example text with issues
    dirty_text = """
    This   has    weird    spacing



    And too many newlines

    And a URL: https://example.com/page
    And email: user@example.com
    And unicode issues: CafÃ©
    """

    preprocessor = TextPreprocessor()
    clean = preprocessor.preprocess(dirty_text)

    print("BEFORE:")
    print(repr(dirty_text))
    print("\nAFTER:")
    print(repr(clean))

    print("\n" + "="*80)
    print("READY FOR BATCH PROCESSING")
    print("="*80)
    print("""
Usage:
    preprocessor = TextPreprocessor(
        fix_unicode=True,
        remove_urls=True,
        remove_emails=True,
        min_length=100
    )

    preprocessor.batch_preprocess(
        input_path="raw_data.jsonl",
        output_path="clean_data.jsonl"
    )
    """)
```

*[Suite avec Deduplication, PII Removal et Quality Filtering dans la partie 2...]*
