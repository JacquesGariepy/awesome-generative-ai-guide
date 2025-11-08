# Chapitre 5: Setup et Environnement de Développement

## Introduction

Maintenant que vous comprenez les **fondations** (Transformers, tokenization, mathématiques), il est temps de configurer votre environnement de développement pour construire et entraîner des LLMs.

### Ce que vous allez configurer

```python
"""
Environnement complet pour développeur LLM:

1. Python Environment
   - Python 3.9+ (3.10 recommandé)
   - Virtual environments (venv, conda)
   - Package managers (pip, poetry)

2. Deep Learning Frameworks
   - PyTorch (framework principal)
   - CUDA (GPU acceleration)
   - cuDNN (optimized primitives)

3. LLM Libraries
   - Transformers (HuggingFace)
   - Accelerate (distributed training)
   - PEFT (parameter-efficient fine-tuning)
   - BitsAndBytes (quantization)

4. Development Tools
   - Jupyter/VSCode
   - Git
   - Weights & Biases (experiment tracking)
   - Docker (containerization)

5. Cloud Platforms
   - Google Colab (free GPUs!)
   - Lambda Labs, RunPod (GPU rental)
   - AWS, GCP, Azure (production)

Tout avec exemples pratiques et best practices!
"""
```

## 1. Python Environment Setup

```python
"""
Python Setup = Fondation de tout projet LLM

Versions:
  - Python 3.9+: minimum requis
  - Python 3.10: recommandé (bon balance compatibilité/features)
  - Python 3.11+: plus rapide mais peut avoir compatibility issues

Virtual Environments:
  - Isole les dépendances par projet
  - Évite les conflits de versions
  - Essentiel pour reproducibilité
"""

# ==================
# Option 1: venv (built-in)
# ==================

print("""
=== Setup avec venv (Python built-in) ===

# Create virtual environment
python3.10 -m venv llm_env

# Activate
# Linux/Mac:
source llm_env/bin/activate

# Windows:
llm_env\\Scripts\\activate

# Verify
which python  # Should point to llm_env/bin/python
python --version  # Should be 3.10.x

# Install packages
pip install --upgrade pip
pip install torch transformers accelerate

# Deactivate
deactivate
""")

# ==================
# Option 2: conda (recommended for beginners)
# ==================

print("""
=== Setup avec conda (Anaconda/Miniconda) ===

# Install Miniconda (if not installed)
# Download from: https://docs.conda.io/en/latest/miniconda.html

# Create environment with Python 3.10
conda create -n llm_env python=3.10 -y

# Activate
conda activate llm_env

# Verify
python --version

# Install PyTorch with CUDA (GPU support)
# Visit: https://pytorch.org/get-started/locally/
# Example for CUDA 12.1:
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install other packages
pip install transformers accelerate datasets evaluate

# List installed packages
conda list

# Export environment (for reproducibility)
conda env export > environment.yml

# Create from environment.yml
conda env create -f environment.yml

# Deactivate
conda deactivate
""")

# ==================
# Option 3: poetry (for production projects)
# ==================

print("""
=== Setup avec poetry (dependency management) ===

# Install poetry
curl -sSL https://install.python-poetry.org | python3 -

# Create new project
poetry new llm_project
cd llm_project

# Or init in existing directory
poetry init

# Add dependencies
poetry add torch transformers accelerate

# Add dev dependencies
poetry add --group dev pytest black mypy

# Install all dependencies
poetry install

# Activate virtual environment
poetry shell

# Run script
poetry run python train.py

# Export requirements.txt (for compatibility)
poetry export -f requirements.txt --output requirements.txt

# pyproject.toml example:
# [tool.poetry.dependencies]
# python = "^3.10"
# torch = "^2.1.0"
# transformers = "^4.35.0"
# accelerate = "^0.24.0"
""")


class EnvironmentVerifier:
    """
    Verify Python environment is correctly set up
    """

    @staticmethod
    def check_python_version():
        """Check Python version"""
        import sys

        print("="*80)
        print("PYTHON VERSION CHECK")
        print("="*80)

        version = sys.version_info
        print(f"\nPython version: {version.major}.{version.minor}.{version.micro}")

        if version.major == 3 and version.minor >= 9:
            print("  ✅ Python version is compatible")
        else:
            print("  ❌ Python 3.9+ required")

        return version.major == 3 and version.minor >= 9

    @staticmethod
    def check_pytorch():
        """Check PyTorch installation"""
        print("\n" + "="*80)
        print("PYTORCH CHECK")
        print("="*80)

        try:
            import torch

            print(f"\nPyTorch version: {torch.__version__}")
            print(f"CUDA available: {torch.cuda.is_available()}")

            if torch.cuda.is_available():
                print(f"CUDA version: {torch.version.cuda}")
                print(f"cuDNN version: {torch.backends.cudnn.version()}")
                print(f"Number of GPUs: {torch.cuda.device_count()}")

                for i in range(torch.cuda.device_count()):
                    print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
                    mem = torch.cuda.get_device_properties(i).total_memory / 1e9
                    print(f"  Memory: {mem:.1f} GB")

                print("\n  ✅ PyTorch with CUDA is properly installed")
            else:
                print("\n  ⚠️  CUDA not available (CPU-only mode)")
                print("     For GPU support, reinstall PyTorch with CUDA")

            return True

        except ImportError:
            print("\n  ❌ PyTorch not installed")
            print("     Install: pip install torch")
            return False

    @staticmethod
    def check_transformers():
        """Check HuggingFace Transformers"""
        print("\n" + "="*80)
        print("TRANSFORMERS CHECK")
        print("="*80)

        try:
            import transformers

            print(f"\nTransformers version: {transformers.__version__}")

            # Try loading a small model
            from transformers import AutoTokenizer

            print("\nTesting model loading...")
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            print("  ✅ Successfully loaded GPT-2 tokenizer")

            # Test tokenization
            text = "Hello, world!"
            tokens = tokenizer(text, return_tensors="pt")
            print(f"\n  Test tokenization: '{text}'")
            print(f"  Token IDs: {tokens['input_ids'][0].tolist()}")
            print(f"\n  ✅ Transformers is properly installed")

            return True

        except ImportError:
            print("\n  ❌ Transformers not installed")
            print("     Install: pip install transformers")
            return False

        except Exception as e:
            print(f"\n  ⚠️  Error loading model: {e}")
            print("     May need internet connection for first download")
            return False

    @staticmethod
    def check_optional_packages():
        """Check optional but useful packages"""
        print("\n" + "="*80)
        print("OPTIONAL PACKAGES")
        print("="*80)

        packages = {
            "accelerate": "Distributed training",
            "peft": "Parameter-efficient fine-tuning",
            "bitsandbytes": "8-bit/4-bit quantization",
            "datasets": "HuggingFace datasets",
            "evaluate": "Evaluation metrics",
            "wandb": "Experiment tracking",
            "tensorboard": "TensorBoard logging",
            "jupyter": "Jupyter notebooks"
        }

        for package, description in packages.items():
            try:
                __import__(package)
                print(f"  ✅ {package:20s} ({description})")
            except ImportError:
                print(f"  ❌ {package:20s} ({description})")

    @staticmethod
    def run_full_check():
        """Run all checks"""
        print("="*80)
        print("ENVIRONMENT VERIFICATION")
        print("="*80)

        checks = [
            EnvironmentVerifier.check_python_version(),
            EnvironmentVerifier.check_pytorch(),
            EnvironmentVerifier.check_transformers()
        ]

        EnvironmentVerifier.check_optional_packages()

        print("\n" + "="*80)
        if all(checks):
            print("✅ ENVIRONMENT IS READY!")
        else:
            print("⚠️  SOME CHECKS FAILED - Review above")
        print("="*80)


# Demo
if __name__ == "__main__":
    # Run verification
    verifier = EnvironmentVerifier()
    verifier.run_full_check()

    print("\n" + "="*80)
    print("RECOMMENDED PACKAGES FOR LLM DEVELOPMENT")
    print("="*80)

    requirements = """
# Core
torch>=2.1.0
transformers>=4.35.0
tokenizers>=0.15.0

# Training
accelerate>=0.24.0
peft>=0.7.0
bitsandbytes>=0.41.0
deepspeed>=0.11.0  # For large-scale training

# Data
datasets>=2.15.0
evaluate>=0.4.0

# Utilities
tqdm>=4.66.0
numpy>=1.24.0
pandas>=2.0.0

# Experiment tracking
wandb>=0.16.0
tensorboard>=2.15.0

# Development
jupyter>=1.0.0
ipython>=8.17.0
black>=23.11.0
pytest>=7.4.0

# Deployment
fastapi>=0.104.0
uvicorn>=0.24.0
onnx>=1.15.0
onnxruntime-gpu>=1.16.0
    """

    print(requirements)

    print("\nInstall all:")
    print("  pip install -r requirements.txt")

    print("\nOr create requirements.txt with above content")
```

*[Suite avec PyTorch CUDA Setup et HuggingFace Ecosystem dans la partie 2...]*
