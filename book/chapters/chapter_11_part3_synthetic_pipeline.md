# Chapitre 11 (Partie 3): Synthetic Data et Pipeline Complet

## 6. Synthetic Data Generation

```python
"""
Synthetic Data = Generate training data using LLMs

Pourquoi synthetic data?
  • Manque de données pour certain tasks
  • Bootstrapping (self-improvement)
  • Data augmentation
  • Privacy-preserving

Méthodes populaires:
  1. Self-Instruct (Stanford, 2022)
     • Generate instructions + responses
     • Used to create Alpaca dataset
     • Cost: ~$500 with GPT-3

  2. Evol-Instruct (WizardLM, 2023)
     • Evolve simple → complex instructions
     • In-depth, breadth, reasoning
     • Better quality

  3. Self-Rewarding (Meta, 2024)
     • Model generates and judges own data
     • Iterative improvement

Exemples:
  • Alpaca: 52k synthetic instructions
  • WizardLM: Evol-Instruct dataset
  • Orca: Explanation tuning with GPT-4
  • Phi-2: Textbook-quality synthetic data

Résultats:
  • Llama 7B + Alpaca → ChatGPT-like performance
  • Quality > Quantity (10k good > 100k bad)
"""

from typing import List, Dict, Optional
import random
import openai
from dataclasses import dataclass
import asyncio
from tqdm import tqdm


@dataclass
class SyntheticExample:
    """Synthetic training example"""
    instruction: str
    input: str
    output: str
    source: str = "synthetic"
    metadata: Optional[Dict] = None


class SelfInstructGenerator:
    """
    Generate synthetic instruction-following data

    Based on Self-Instruct paper (Wang et al., 2022)

    Example:
        >>> generator = SelfInstructGenerator(api_key="your-key")
        >>> examples = generator.generate_batch(num_examples=100)
    """

    # Seed tasks for bootstrapping
    SEED_TASKS = [
        {
            "instruction": "Write a short poem about {topic}",
            "topic_pool": ["nature", "love", "technology", "friendship", "time"]
        },
        {
            "instruction": "Explain {concept} in simple terms",
            "topic_pool": ["quantum mechanics", "blockchain", "photosynthesis", "democracy"]
        },
        {
            "instruction": "Generate a creative name for a {type}",
            "topic_pool": ["startup", "product", "band", "restaurant", "app"]
        },
        {
            "instruction": "List 5 tips for {activity}",
            "topic_pool": ["studying", "exercising", "saving money", "cooking", "time management"]
        },
    ]

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7
    ):
        """
        Args:
            api_key: OpenAI API key
            model: Model to use for generation
            temperature: Sampling temperature
        """
        openai.api_key = api_key
        self.model = model
        self.temperature = temperature

    def generate_instruction(self, seed_task: Optional[Dict] = None) -> str:
        """
        Generate new instruction from seed task

        Args:
            seed_task: Seed task template (random if None)

        Returns:
            Generated instruction
        """
        if seed_task is None:
            seed_task = random.choice(self.SEED_TASKS)

        # Fill template
        topic = random.choice(seed_task["topic_pool"])
        instruction = seed_task["instruction"].format(topic=topic)

        return instruction

    def generate_response(self, instruction: str, input_text: str = "") -> str:
        """
        Generate response for instruction

        Args:
            instruction: The instruction
            input_text: Additional input context

        Returns:
            Generated response
        """
        # Build prompt
        if input_text:
            prompt = f"{instruction}\n\nInput: {input_text}\n\nOutput:"
        else:
            prompt = f"{instruction}\n\nOutput:"

        # Call API
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that follows instructions carefully."},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            max_tokens=512
        )

        output = response.choices[0].message.content.strip()
        return output

    def generate_example(self) -> SyntheticExample:
        """
        Generate single synthetic example

        Returns:
            Synthetic example
        """
        # Generate instruction
        instruction = self.generate_instruction()

        # Generate response
        output = self.generate_response(instruction)

        return SyntheticExample(
            instruction=instruction,
            input="",
            output=output,
            source="self_instruct"
        )

    def generate_batch(
        self,
        num_examples: int = 100,
        show_progress: bool = True
    ) -> List[SyntheticExample]:
        """
        Generate batch of examples

        Args:
            num_examples: Number of examples to generate
            show_progress: Show progress bar

        Returns:
            List of synthetic examples
        """
        examples = []

        iterator = tqdm(range(num_examples)) if show_progress else range(num_examples)

        for _ in iterator:
            try:
                example = self.generate_example()
                examples.append(example)
            except Exception as e:
                print(f"Error generating example: {e}")
                continue

        return examples

    def save_to_json(self, examples: List[SyntheticExample], output_path: str):
        """
        Save examples to JSON file

        Args:
            examples: Examples to save
            output_path: Output path
        """
        data = [
            {
                'instruction': ex.instruction,
                'input': ex.input,
                'output': ex.output,
                'source': ex.source
            }
            for ex in examples
        ]

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"✅ Saved {len(examples)} examples to {output_path}")


class EvolInstructGenerator:
    """
    Evolve instructions to be more complex (WizardLM approach)

    Evolution types:
      • In-depth: Add constraints, requirements
      • Breadth: Broaden scope
      • Concretize: Make more specific
      • Reasoning: Add reasoning steps

    Example:
        >>> generator = EvolInstructGenerator(api_key="your-key")
        >>> evolved = generator.evolve_instruction("Write a poem")
    """

    EVOLUTION_PROMPTS = {
        "in_depth": """
Make the following instruction more complex by adding constraints or requirements:

Original: {instruction}

Evolved instruction:
""",
        "breadth": """
Broaden the following instruction to cover more aspects:

Original: {instruction}

Evolved instruction:
""",
        "concretize": """
Make the following instruction more specific and concrete:

Original: {instruction}

Evolved instruction:
""",
        "reasoning": """
Add reasoning or step-by-step requirements to the following instruction:

Original: {instruction}

Evolved instruction:
"""
    }

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4",  # GPT-4 better for evolution
        temperature: float = 0.7
    ):
        openai.api_key = api_key
        self.model = model
        self.temperature = temperature

    def evolve_instruction(
        self,
        instruction: str,
        evolution_type: Optional[str] = None
    ) -> str:
        """
        Evolve instruction to be more complex

        Args:
            instruction: Original instruction
            evolution_type: Type of evolution (random if None)

        Returns:
            Evolved instruction
        """
        # Random evolution type if not specified
        if evolution_type is None:
            evolution_type = random.choice(list(self.EVOLUTION_PROMPTS.keys()))

        # Get evolution prompt
        prompt_template = self.EVOLUTION_PROMPTS[evolution_type]
        prompt = prompt_template.format(instruction=instruction)

        # Call API
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert at creating complex, detailed instructions."},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            max_tokens=256
        )

        evolved = response.choices[0].message.content.strip()
        return evolved

    def evolve_dataset(
        self,
        instructions: List[str],
        num_evolutions: int = 2
    ) -> List[str]:
        """
        Evolve entire dataset

        Args:
            instructions: Original instructions
            num_evolutions: Number of evolution iterations

        Returns:
            Evolved instructions
        """
        evolved = instructions.copy()

        for iter in range(num_evolutions):
            print(f"Evolution iteration {iter+1}/{num_evolutions}")

            new_evolved = []
            for instruction in tqdm(evolved):
                try:
                    evolved_inst = self.evolve_instruction(instruction)
                    new_evolved.append(evolved_inst)
                except Exception as e:
                    print(f"Error evolving: {e}")
                    new_evolved.append(instruction)  # Keep original

            evolved = new_evolved

        return evolved


# Demo
def demo_synthetic_generation():
    """Demo synthetic data generation"""
    print("="*80)
    print("SYNTHETIC DATA GENERATION")
    print("="*80)

    print("""
SELF-INSTRUCT (Alpaca-style):

  # Generate 52k instructions like Alpaca
  generator = SelfInstructGenerator(api_key="your-key")

  examples = generator.generate_batch(num_examples=52000)

  generator.save_to_json(examples, "alpaca_synthetic.json")

  # Cost: ~$500 with GPT-3.5-turbo
  # Time: ~10 hours


EVOL-INSTRUCT (WizardLM-style):

  # Start with simple instructions
  simple_instructions = [
      "Write a poem",
      "Explain photosynthesis",
      "Write a function to sort a list"
  ]

  # Evolve 2-3 times
  evol_gen = EvolInstructGenerator(api_key="your-key")
  complex_instructions = evol_gen.evolve_dataset(
      simple_instructions,
      num_evolutions=2
  )

  # Results:
  # "Write a poem" →
  # "Write a sonnet about climate change that uses metaphors from
  #  quantum physics and follows iambic pentameter"


TIPS:
  ✅ Start with high-quality seed tasks
  ✅ Use GPT-4 for better quality (worth the cost)
  ✅ Filter low-quality generations (check length, coherence)
  ✅ Mix with real data (50% synthetic, 50% real)
  ✅ Iterate: Generate → Train → Evaluate → Refine
    """)


if __name__ == "__main__":
    demo_synthetic_generation()
```

## 7. Data Mixing

```python
"""
Data Mixing = Combine different data sources with optimal ratios

Pourquoi mixer?
  • Single source = biased model
  • Multiple sources = robust, diverse model
  • Ratios = control model capabilities

Exemples de mix:
  • Llama 2:
    - Web: 50%
    - Books: 15%
    - Code: 10%
    - Scientific: 10%
    - Wikipedia: 5%
    - Other: 10%

  • GPT-3:
    - Common Crawl (filtered): 60%
    - WebText2: 22%
    - Books1+2: 16%
    - Wikipedia: 3%

Stratégies:
  1. Domain-based mixing (web, books, code)
  2. Quality-based mixing (high quality = higher weight)
  3. Task-based mixing (instruction, chat, reasoning)
  4. Curriculum learning (easy → hard)

Implementation:
  • Sampling probability per source
  • Ensure balanced epochs
  • Track source distribution
"""


@dataclass
class DataSource:
    """Data source configuration"""
    name: str
    path: str
    weight: float  # Sampling probability
    num_documents: int = 0


class DataMixer:
    """
    Mix multiple data sources with specified ratios

    Example:
        >>> mixer = DataMixer([
        ...     DataSource("web", "web.jsonl", weight=0.5),
        ...     DataSource("books", "books.jsonl", weight=0.3),
        ...     DataSource("code", "code.jsonl", weight=0.2)
        ... ])
        >>> mixed = mixer.create_mixed_dataset("mixed.jsonl", total_docs=1000000)
    """

    def __init__(self, sources: List[DataSource]):
        """
        Args:
            sources: List of data sources with weights
        """
        self.sources = sources

        # Normalize weights
        total_weight = sum(s.weight for s in sources)
        for source in self.sources:
            source.weight /= total_weight

        # Count documents per source
        for source in self.sources:
            source.num_documents = self._count_documents(source.path)

        print("Data sources:")
        for source in self.sources:
            print(f"  {source.name}: {source.num_documents:,} docs (weight={source.weight:.2%})")

    def _count_documents(self, path: str) -> int:
        """Count documents in JSONL file"""
        count = 0
        with open(path) as f:
            for _ in f:
                count += 1
        return count

    def create_mixed_dataset(
        self,
        output_path: str,
        total_docs: int,
        shuffle: bool = True
    ):
        """
        Create mixed dataset

        Args:
            output_path: Output JSONL path
            total_docs: Total number of documents to sample
            shuffle: Shuffle output
        """
        print(f"\nCreating mixed dataset: {output_path}")
        print(f"Target size: {total_docs:,} documents")

        # Calculate documents per source
        docs_per_source = {}
        for source in self.sources:
            target_docs = int(total_docs * source.weight)
            # Cap at available documents
            actual_docs = min(target_docs, source.num_documents)
            docs_per_source[source.name] = actual_docs

        print("\nSampling plan:")
        for source in self.sources:
            n = docs_per_source[source.name]
            print(f"  {source.name}: {n:,} docs ({n/total_docs:.2%})")

        # Sample from each source
        all_docs = []

        for source in self.sources:
            n = docs_per_source[source.name]
            print(f"\nSampling {n:,} from {source.name}...")

            # Read all documents
            docs = []
            with open(source.path) as f:
                for line in f:
                    docs.append(line)

            # Random sample
            if len(docs) > n:
                sampled = random.sample(docs, n)
            else:
                sampled = docs

            # Add source tag
            for line in sampled:
                doc = json.loads(line)
                doc['_source'] = source.name
                all_docs.append(json.dumps(doc))

            print(f"  ✅ Sampled {len(sampled):,} documents")

        # Shuffle if requested
        if shuffle:
            print("\nShuffling...")
            random.shuffle(all_docs)

        # Write output
        print(f"\nWriting to {output_path}...")
        with open(output_path, 'w') as f:
            for line in all_docs:
                f.write(line + '\n')

        print(f"\n✅ Created mixed dataset: {len(all_docs):,} documents")

        # Print final statistics
        source_counts = defaultdict(int)
        with open(output_path) as f:
            for line in f:
                doc = json.loads(line)
                source_counts[doc.get('_source', 'unknown')] += 1

        print("\nFinal distribution:")
        for source_name, count in sorted(source_counts.items()):
            print(f"  {source_name}: {count:,} ({count/len(all_docs):.2%})")


class CurriculumDataMixer:
    """
    Mix data with curriculum learning strategy

    Start with easy examples, gradually increase difficulty

    Example:
        >>> mixer = CurriculumDataMixer(sources)
        >>> mixer.create_curriculum_dataset(
        ...     output_path="curriculum.jsonl",
        ...     stages=[
        ...         {"difficulty": "easy", "ratio": 0.5},
        ...         {"difficulty": "medium", "ratio": 0.3},
        ...         {"difficulty": "hard", "ratio": 0.2}
        ...     ]
        ... )
    """

    def __init__(self, sources: List[DataSource]):
        self.sources = sources

    def estimate_difficulty(self, text: str) -> float:
        """
        Estimate difficulty of text (simple heuristic)

        Args:
            text: Input text

        Returns:
            Difficulty score (0-1)
        """
        # Simple heuristics:
        # - Longer sentences = harder
        # - Longer words = harder
        # - More rare words = harder

        words = text.split()
        if not words:
            return 0.0

        avg_word_length = sum(len(w) for w in words) / len(words)

        sentences = text.split('.')
        if sentences:
            avg_sentence_length = len(words) / len(sentences)
        else:
            avg_sentence_length = 0

        # Normalize to 0-1
        difficulty = min(1.0, (avg_word_length / 10 + avg_sentence_length / 50) / 2)

        return difficulty

    def create_curriculum_dataset(
        self,
        output_path: str,
        stages: List[Dict],
        total_docs: int = 100000
    ):
        """
        Create curriculum dataset

        Args:
            output_path: Output path
            stages: List of difficulty stages with ratios
            total_docs: Total documents
        """
        print(f"Creating curriculum dataset: {output_path}")

        # Load all documents with difficulty scores
        all_docs = []

        for source in self.sources:
            print(f"Processing {source.name}...")

            with open(source.path) as f:
                for line in f:
                    doc = json.loads(line)
                    text = doc.get('text', '')

                    difficulty = self.estimate_difficulty(text)

                    all_docs.append({
                        'doc': line,
                        'difficulty': difficulty,
                        'source': source.name
                    })

        print(f"Loaded {len(all_docs):,} documents")

        # Sort by difficulty
        all_docs.sort(key=lambda x: x['difficulty'])

        # Sample by stages
        curriculum_docs = []

        start_idx = 0
        for stage in stages:
            stage_size = int(total_docs * stage['ratio'])

            # Find documents in difficulty range
            if stage['difficulty'] == 'easy':
                threshold = 0.33
            elif stage['difficulty'] == 'medium':
                threshold = 0.66
            else:
                threshold = 1.0

            # Sample from appropriate difficulty range
            end_idx = int(len(all_docs) * threshold)
            stage_docs = all_docs[start_idx:end_idx]

            if len(stage_docs) > stage_size:
                sampled = random.sample(stage_docs, stage_size)
            else:
                sampled = stage_docs

            curriculum_docs.extend([d['doc'] for d in sampled])
            start_idx = end_idx

            print(f"  {stage['difficulty']}: {len(sampled):,} docs")

        # Write
        with open(output_path, 'w') as f:
            for line in curriculum_docs:
                f.write(line)

        print(f"\n✅ Created curriculum dataset: {len(curriculum_docs):,} documents")


# Demo
if __name__ == "__main__":
    print("="*80)
    print("DATA MIXING")
    print("="*80)

    print("""
BASIC MIXING:

  sources = [
      DataSource("web", "cc_clean.jsonl", weight=0.5),
      DataSource("books", "books.jsonl", weight=0.2),
      DataSource("code", "github.jsonl", weight=0.15),
      DataSource("wiki", "wikipedia.jsonl", weight=0.1),
      DataSource("papers", "arxiv.jsonl", weight=0.05),
  ]

  mixer = DataMixer(sources)
  mixer.create_mixed_dataset(
      output_path="mixed_pretrain.jsonl",
      total_docs=1_000_000,  # 1M documents
      shuffle=True
  )


CURRICULUM LEARNING:

  curriculum_mixer = CurriculumDataMixer(sources)
  curriculum_mixer.create_curriculum_dataset(
      output_path="curriculum.jsonl",
      stages=[
          {"difficulty": "easy", "ratio": 0.5},
          {"difficulty": "medium", "ratio": 0.3},
          {"difficulty": "hard", "ratio": 0.2}
      ],
      total_docs=100_000
  )


BEST PRACTICES:
  ✅ Start with balanced mix
  ✅ Oversample high-quality sources (books, wiki)
  ✅ Undersample noisy sources (raw web)
  ✅ Track source in metadata (for debugging)
  ✅ Iterate based on eval results
  ✅ Consider task-specific mixes (code model = more code data)
    """)
```

## 8. Complete Production Pipeline

```python
"""
COMPLETE DATA PIPELINE

End-to-end pipeline:
  1. Download raw data (Common Crawl, etc.)
  2. Extract text (HTML → text)
  3. Preprocess (unicode, whitespace)
  4. Filter quality (heuristics)
  5. Deduplicate (exact + fuzzy)
  6. Remove PII
  7. Language detection
  8. Mix sources
  9. Tokenize
  10. Save shards for training

Time estimate: 1-4 weeks for 1TB dataset
Cost estimate: $1k-$10k (compute + storage)
"""

from pathlib import Path
import logging
from typing import Iterator
from datetime import datetime


class ProductionDataPipeline:
    """
    Complete production data pipeline

    Example:
        >>> pipeline = ProductionDataPipeline(
        ...     output_dir="./processed_data",
        ...     num_workers=8
        ... )
        >>> pipeline.run(
        ...     input_sources=[
        ...         {"name": "cc", "path": "cc_raw.jsonl"},
        ...         {"name": "wiki", "path": "wiki_raw.jsonl"}
        ...     ],
        ...     mix_weights={"cc": 0.8, "wiki": 0.2}
        ... )
    """

    def __init__(
        self,
        output_dir: str = "./processed_data",
        num_workers: int = 4,
        log_level: str = "INFO"
    ):
        """
        Args:
            output_dir: Output directory
            num_workers: Number of parallel workers
            log_level: Logging level
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        self.num_workers = num_workers

        # Setup logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'pipeline.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('DataPipeline')

        # Components
        self.preprocessor = TextPreprocessor()
        self.deduplicator = ExactDeduplicator()
        self.minhash_dedup = MinHashDeduplicator()
        self.pii_remover = PIIRemover()
        self.quality_filter = QualityFilter()

    def process_document(self, doc: Dict) -> Optional[Dict]:
        """
        Process single document through pipeline

        Args:
            doc: Input document

        Returns:
            Processed document or None if filtered
        """
        text = doc.get('text', '')

        # 1. Preprocess
        text = self.preprocessor.preprocess(text)
        if not text:
            return None

        # 2. Quality filter
        is_good, _ = self.quality_filter.is_high_quality(text)
        if not is_good:
            return None

        # 3. Deduplication check
        if self.deduplicator.is_duplicate(text):
            return None

        # 4. PII removal
        text, _ = self.pii_remover.remove_pii(text)

        # Update document
        doc['text'] = text
        doc['processed_at'] = datetime.now().isoformat()

        return doc

    def process_source(
        self,
        source_path: str,
        source_name: str
    ) -> str:
        """
        Process single data source

        Args:
            source_path: Input JSONL path
            source_name: Source name

        Returns:
            Output path
        """
        self.logger.info(f"Processing source: {source_name}")

        output_path = self.output_dir / f"{source_name}_processed.jsonl"

        total = 0
        kept = 0

        with open(source_path) as f_in, open(output_path, 'w') as f_out:
            for line in f_in:
                total += 1

                try:
                    doc = json.loads(line)
                    processed = self.process_document(doc)

                    if processed:
                        f_out.write(json.dumps(processed) + '\n')
                        kept += 1

                except Exception as e:
                    self.logger.error(f"Error processing document: {e}")

                if total % 10000 == 0:
                    self.logger.info(f"  Processed {total:,} ({kept/total:.1%} kept)")

        self.logger.info(f"✅ {source_name}: {total:,} → {kept:,} ({kept/total:.1%})")

        return str(output_path)

    def run(
        self,
        input_sources: List[Dict],
        mix_weights: Dict[str, float],
        total_docs: int = 1_000_000
    ):
        """
        Run complete pipeline

        Args:
            input_sources: List of {"name": str, "path": str}
            mix_weights: Dict of source weights
            total_docs: Target total documents
        """
        self.logger.info("="*80)
        self.logger.info("STARTING DATA PIPELINE")
        self.logger.info("="*80)

        start_time = datetime.now()

        # Process each source
        processed_sources = []

        for source in input_sources:
            output_path = self.process_source(
                source_path=source['path'],
                source_name=source['name']
            )

            processed_sources.append(
                DataSource(
                    name=source['name'],
                    path=output_path,
                    weight=mix_weights.get(source['name'], 1.0)
                )
            )

        # Mix sources
        self.logger.info("\nMixing sources...")

        mixer = DataMixer(processed_sources)
        mixed_path = self.output_dir / "final_mixed.jsonl"

        mixer.create_mixed_dataset(
            output_path=str(mixed_path),
            total_docs=total_docs,
            shuffle=True
        )

        # Near-deduplication (optional, expensive)
        self.logger.info("\nNear-deduplication (this may take a while)...")
        # Skip for now, but would go here

        # Final statistics
        elapsed = (datetime.now() - start_time).total_seconds()

        self.logger.info("\n" + "="*80)
        self.logger.info("PIPELINE COMPLETE")
        self.logger.info("="*80)
        self.logger.info(f"Time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
        self.logger.info(f"Output: {mixed_path}")
        self.logger.info("="*80)

        return str(mixed_path)


# Demo
if __name__ == "__main__":
    print("="*80)
    print("COMPLETE PRODUCTION DATA PIPELINE")
    print("="*80)

    print("""
FULL PIPELINE EXAMPLE:

# 1. Setup
pipeline = ProductionDataPipeline(
    output_dir="./processed_data",
    num_workers=8
)

# 2. Define sources
input_sources = [
    {"name": "common_crawl", "path": "cc_raw.jsonl"},
    {"name": "wikipedia", "path": "wiki_raw.jsonl"},
    {"name": "books", "path": "books_raw.jsonl"},
    {"name": "github", "path": "code_raw.jsonl"},
]

# 3. Define mixing ratios (Llama 2 style)
mix_weights = {
    "common_crawl": 0.67,  # 67%
    "wikipedia": 0.045,    # 4.5%
    "books": 0.045,        # 4.5%
    "github": 0.045,       # 4.5%
}

# 4. Run pipeline
final_dataset = pipeline.run(
    input_sources=input_sources,
    mix_weights=mix_weights,
    total_docs=10_000_000  # 10M documents
)

# 5. Result
# → processed_data/final_mixed.jsonl
# Ready for tokenization and training!


WHAT THE PIPELINE DOES:

For each source:
  ✅ Preprocess (unicode, whitespace)
  ✅ Quality filtering (heuristics)
  ✅ Exact deduplication
  ✅ PII removal
  ✅ Save intermediate results

Then:
  ✅ Mix sources with specified weights
  ✅ Shuffle documents
  ✅ Save final dataset
  ✅ Log statistics and metrics


EXPECTED RESULTS:

Input: 100GB raw data (10B tokens)
After filtering: 20GB clean data (2B tokens)
After deduplication: 15GB unique data (1.5B tokens)

Filtering rate: 80-85% is normal!
Quality > Quantity!


NEXT STEPS:

1. Tokenization:
   tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
   # Tokenize all documents

2. Shard creation:
   # Split into shards (1GB each)
   # For distributed training

3. Training:
   # Use Chapter 6 pretraining code
   # Train on clean, deduplicated, mixed data

4. Evaluation:
   # Perplexity on validation set
   # Downstream task performance
    """)

    print("\n✅ CHAPITRE 11 TERMINÉ!")
    print("="*80)
    print("""
Vous maîtrisez maintenant:

1. Sources de données
   ✅ Common Crawl download & processing
   ✅ Wikipedia dumps
   ✅ Books (Gutenberg)
   ✅ Code, scientific papers

2. Preprocessing
   ✅ Unicode normalization
   ✅ HTML extraction
   ✅ Text cleaning
   ✅ Language detection

3. Deduplication
   ✅ Exact (hash-based)
   ✅ Near (MinHash + LSH)
   ✅ Line-level

4. PII Removal
   ✅ Regex-based
   ✅ NER-based
   ✅ Privacy compliance

5. Quality Filtering
   ✅ Heuristic filtering
   ✅ Statistical metrics
   ✅ ML-based filtering

6. Synthetic Data
   ✅ Self-Instruct
   ✅ Evol-Instruct
   ✅ Quality vs quantity

7. Data Mixing
   ✅ Multi-source mixing
   ✅ Optimal ratios
   ✅ Curriculum learning

8. Production Pipeline
   ✅ End-to-end automation
   ✅ Logging & monitoring
   ✅ Scalable processing

Ready to build production data pipelines!

Next: Chapter 12 → Optimized Inference (vLLM, TensorRT)
    """)
```
