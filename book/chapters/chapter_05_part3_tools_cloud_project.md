# Chapitre 5 (Partie 3): Development Tools, Cloud Platforms et Projet Complet

## 4. Development Tools

```python
"""
Development Tools = Productivité maximale

Outils essentiels:
  • IDE: VSCode, PyCharm, Jupyter
  • Version control: Git, GitHub
  • Experiment tracking: W&B, TensorBoard
  • Debugging: pdb, PyTorch profiler
  • Documentation: Sphinx, MkDocs
"""

class DevelopmentTools:
    """
    Essential development tools for LLM work
    """

    @staticmethod
    def vscode_setup():
        """
        VSCode setup for LLM development
        """
        print("="*80)
        print("VSCODE SETUP FOR LLM DEVELOPMENT")
        print("="*80)

        print("""
VSCode = Most popular IDE for Python/AI

1. Install VSCode
   • Download: https://code.visualstudio.com/
   • Install extensions

2. Essential extensions:

   Python Development:
   • Python (ms-python.python) - Core Python support
   • Pylance (ms-python.vscode-pylance) - Fast IntelliSense
   • Python Debugger (ms-python.debugpy) - Debugging

   AI/ML Specific:
   • Jupyter (ms-toolsai.jupyter) - Notebooks in VSCode
   • GitHub Copilot (github.copilot) - AI code completion
   • Remote - SSH (ms-vscode-remote.remote-ssh) - Remote GPU servers

   Code Quality:
   • Black Formatter (ms-python.black-formatter) - Auto-formatting
   • Mypy (ms-python.mypy-type-checker) - Type checking
   • Ruff (charliermarsh.ruff) - Fast linter

   Productivity:
   • GitLens (eamodio.gitlens) - Git superpowers
   • Thunder Client (rangav.vscode-thunder-client) - API testing
   • Error Lens (usernamehw.errorlens) - Inline errors

3. Settings (settings.json):
        """)

        print('''
{
    // Python
    "python.defaultInterpreterPath": "./llm_env/bin/python",
    "python.formatting.provider": "black",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": false,
    "python.linting.flake8Enabled": true,

    // Editor
    "editor.formatOnSave": true,
    "editor.rulers": [88, 120],
    "editor.inlineSuggest.enabled": true,

    // Files
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true,
        "**/.pytest_cache": true
    },

    // Jupyter
    "jupyter.askForKernelRestart": false,
    "jupyter.interactiveWindow.textEditor.executeSelection": true
}
        ''')

        print("""
4. Keyboard shortcuts (customize in keybindings.json):
   • Ctrl+Shift+P: Command palette
   • Ctrl+`: Toggle terminal
   • F5: Start debugging
   • Shift+Enter: Run cell (Jupyter)
   • Ctrl+K Ctrl+0: Fold all
   • Ctrl+/: Toggle comment

5. Snippets for common patterns:

   Create: .vscode/python.code-snippets
        """)

        print('''
{
    "PyTorch Model": {
        "prefix": "pytorchmodel",
        "body": [
            "import torch",
            "import torch.nn as nn",
            "",
            "class ${1:ModelName}(nn.Module):",
            "    def __init__(self, ${2:args}):",
            "        super().__init__()",
            "        $3",
            "",
            "    def forward(self, x):",
            "        $4",
            "        return x"
        ]
    },
    "Training Loop": {
        "prefix": "trainloop",
        "body": [
            "for epoch in range(num_epochs):",
            "    model.train()",
            "    for batch in train_loader:",
            "        optimizer.zero_grad()",
            "        outputs = model(batch)",
            "        loss = criterion(outputs, targets)",
            "        loss.backward()",
            "        optimizer.step()",
            "    print(f'Epoch {epoch}: Loss {loss.item():.4f}')"
        ]
    }
}
        ''')

    @staticmethod
    def jupyter_setup():
        """
        Jupyter setup
        """
        print("\n" + "="*80)
        print("JUPYTER NOTEBOOK SETUP")
        print("="*80)

        print("""
Jupyter = Interactive development (great for experimentation)

1. Install
   pip install jupyter jupyterlab ipywidgets

2. Install kernels (for different environments)
   # Activate your environment
   conda activate llm_env

   # Install kernel
   python -m ipykernel install --user --name llm_env --display-name "Python (LLM)"

   # List kernels
   jupyter kernelspec list

3. Useful extensions

   JupyterLab:
   pip install jupyterlab-git  # Git integration
   pip install jupyterlab-lsp  # Language server
   pip install jupyterlab_code_formatter  # Auto-format cells

4. Magic commands (essential!)
   %timeit: Time execution
   %load_ext autoreload: Auto-reload modules
   %autoreload 2: Reload all modules before execution
   %%time: Time cell execution
   %%writefile: Save cell to file
   !command: Run shell command

5. Best practices for notebooks:
   • Use %autoreload for development
   • Clear outputs before committing
   • Convert to .py for production
   • Use papermill for parameterized notebooks

Example notebook setup:
        """)

        print('''
# Cell 1: Setup
%load_ext autoreload
%autoreload 2

import torch
import numpy as np
import matplotlib.pyplot as plt

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

# Check GPU
print(f"CUDA available: {torch.cuda.is_available()}")

# Cell 2: Load model
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Cell 3: Experiment...
        ''')

    @staticmethod
    def experiment_tracking():
        """
        Experiment tracking setup
        """
        print("\n" + "="*80)
        print("EXPERIMENT TRACKING")
        print("="*80)

        print("""
Experiment Tracking = Essential for ML research/development

Why needed:
  • Track hyperparameters
  • Log metrics (loss, accuracy, perplexity)
  • Compare experiments
  • Reproducibility

Two popular options:

1. Weights & Biases (W&B) - Industry standard
2. TensorBoard - Simple, built into PyTorch

=== Weights & Biases ===

Setup:
  # Install
  pip install wandb

  # Login
  wandb login  # Enter API key from wandb.ai

Usage:
        """)

        print('''
import wandb

# Initialize
wandb.init(
    project="llm-finetuning",
    config={
        "learning_rate": 5e-5,
        "batch_size": 32,
        "epochs": 10,
        "model": "gpt2"
    }
)

# Log metrics
for epoch in range(epochs):
    for batch in dataloader:
        loss = train_step(batch)

        # Log to W&B
        wandb.log({
            "train/loss": loss,
            "train/perplexity": np.exp(loss),
            "epoch": epoch
        })

# Log artifacts (model, predictions, etc.)
wandb.save("model.pt")

# Finish
wandb.finish()
        ''')

        print("""
View results: https://wandb.ai/your-username/llm-finetuning

Features:
  ✅ Beautiful dashboards
  ✅ Hyperparameter sweeps
  ✅ Model versioning
  ✅ Team collaboration
  ✅ Free for personal use

=== TensorBoard ===

Setup:
  # Install
  pip install tensorboard

Usage:
        """)

        print('''
from torch.utils.tensorboard import SummaryWriter

# Initialize
writer = SummaryWriter('runs/experiment1')

# Log scalars
for epoch in range(epochs):
    for batch_idx, batch in enumerate(dataloader):
        loss = train_step(batch)

        # Log
        global_step = epoch * len(dataloader) + batch_idx
        writer.add_scalar('Loss/train', loss, global_step)

    # Log histograms
    for name, param in model.named_parameters():
        writer.add_histogram(name, param, epoch)

# Log graph
writer.add_graph(model, input_sample)

# Close
writer.close()
        ''')

        print("""
View results:
  tensorboard --logdir runs

Open: http://localhost:6006

Features:
  ✅ Simple, no account needed
  ✅ Scalars, histograms, graphs
  ✅ Built into PyTorch
  ❌ Less features than W&B
        """)

    @staticmethod
    def git_best_practices():
        """
        Git best practices for ML projects
        """
        print("\n" + "="*80)
        print("GIT BEST PRACTICES FOR ML PROJECTS")
        print("="*80)

        print("""
Git = Version control (essential for any project)

ML-specific challenges:
  • Large model files (100GB+)
  • Large datasets
  • Jupyter notebooks (hard to diff)
  • Experiment artifacts

Solutions:

1. .gitignore for ML projects
        """)

        print('''
# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
*.egg-info/

# Jupyter
.ipynb_checkpoints
*/.ipynb_checkpoints/*

# Models & Data
*.pt
*.pth
*.bin
*.onnx
data/
datasets/
models/
checkpoints/
*.h5
*.pkl
*.pickle

# Logs
logs/
runs/
wandb/
mlruns/
*.log

# OS
.DS_Store
Thumbs.db

# IDEs
.vscode/
.idea/
*.swp
*.swo
        ''')

        print("""
2. Git LFS (Large File Storage) for models
   # Install
   git lfs install

   # Track large files
   git lfs track "*.pt"
   git lfs track "*.pth"
   git lfs track "*.bin"

   # Add .gitattributes
   git add .gitattributes

   # Now large files are handled by LFS
   git add model.pt
   git commit -m "Add trained model"
   git push

3. DVC (Data Version Control) for datasets
   # Install
   pip install dvc

   # Initialize
   dvc init

   # Track dataset
   dvc add data/train.csv

   # Commit .dvc file (not actual data)
   git add data/train.csv.dvc .gitignore
   git commit -m "Add training data"

   # Configure remote storage (S3, GCS, etc.)
   dvc remote add -d storage s3://mybucket/dvc

   # Push data to remote
   dvc push

   # Pull data on another machine
   dvc pull

4. Commit message conventions
   • feat: Add new feature
   • fix: Bug fix
   • perf: Performance improvement
   • docs: Documentation
   • exp: Experiment (for ML experiments)

   Examples:
   git commit -m "exp: Try learning rate 1e-4"
   git commit -m "feat: Add LoRA fine-tuning"
   git commit -m "perf: Optimize data loading (2x faster)"

5. Branches for experiments
   # Create experiment branch
   git checkout -b exp/lora-rank-8

   # After experiment, merge if successful
   git checkout main
   git merge exp/lora-rank-8

   # Or delete if unsuccessful
   git branch -D exp/lora-rank-8
        """)


# Demo
if __name__ == "__main__":
    tools = DevelopmentTools()

    tools.vscode_setup()
    tools.jupyter_setup()
    tools.experiment_tracking()
    tools.git_best_practices()
```

## 5. Cloud Platforms et GPU Access

```python
"""
Cloud Platforms = Access aux GPUs sans investissement matériel

Options:
  • Free: Google Colab, Kaggle
  • Low-cost: Lambda Labs, RunPod, Vast.ai
  • Enterprise: AWS, GCP, Azure
"""

class CloudPlatformsGuide:
    """
    Guide for cloud GPU platforms
    """

    @staticmethod
    def google_colab():
        """
        Google Colab setup
        """
        print("="*80)
        print("GOOGLE COLAB - FREE GPU ACCESS")
        print("="*80)

        print("""
Google Colab = Free Jupyter notebooks with GPU

Advantages:
  ✅ FREE GPU (Tesla T4)
  ✅ FREE TPU
  ✅ No setup required
  ✅ Google Drive integration
  ✅ Pre-installed libraries

Limitations:
  ❌ 12-hour session limit
  ❌ May disconnect if idle
  ❌ Limited GPU memory (15GB)
  ❌ Can't run long training jobs

Best for:
  • Learning, tutorials
  • Small experiments
  • Fine-tuning small models
  • Inference

Setup:

1. Visit: https://colab.research.google.com/

2. Create new notebook

3. Enable GPU
   Runtime → Change runtime type → GPU → T4

4. Verify GPU
        """)

        print('''
!nvidia-smi

import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
        ''')

        print("""
5. Mount Google Drive (for saving models)
        """)

        print('''
from google.colab import drive
drive.mount('/content/drive')

# Save model
model.save_pretrained('/content/drive/MyDrive/models/finetuned-gpt2')
        ''')

        print("""
6. Install packages (reset each session)
        """)

        print('''
!pip install transformers accelerate peft bitsandbytes
        ''')

        print("""
Tips:
  • Save checkpoints frequently
  • Use Google Drive for persistence
  • Keep browser tab active (prevent disconnect)
  • Use Colab Pro ($10/month) for longer sessions (24h) and better GPUs (V100, A100)

Pro version:
  • Colab Pro: $10/month
    - 24h sessions
    - V100 GPU
    - More memory

  • Colab Pro+: $50/month
    - Background execution
    - A100 GPU
    - Even more resources
        """)

    @staticmethod
    def lambda_labs():
        """
        Lambda Labs GPU cloud
        """
        print("\n" + "="*80)
        print("LAMBDA LABS - GPU CLOUD")
        print("="*80)

        print("""
Lambda Labs = Budget-friendly GPU cloud (optimized for ML)

Pricing (as of 2024):
  • RTX 6000 Ada (48GB): $0.50/hour
  • A100 (40GB): $1.10/hour
  • A100 (80GB): $1.29/hour
  • H100 (80GB): $2.49/hour

Advantages:
  ✅ Simple pricing (no hidden fees)
  ✅ Pre-configured for ML (PyTorch, CUDA, etc.)
  ✅ SSH access
  ✅ Persistent storage
  ✅ API for automation

Setup:

1. Sign up: https://lambdalabs.com/

2. Add payment method

3. Launch instance
   • Select GPU type
   • Select region
   • Add SSH key
   • Launch

4. Connect via SSH
   ssh ubuntu@<instance-ip>

5. Instance comes with:
   • Ubuntu 22.04
   • CUDA 12.1
   • PyTorch, TensorFlow
   • Jupyter Lab

6. Start Jupyter
   jupyter lab --ip=0.0.0.0 --no-browser

   Access: http://<instance-ip>:8888

7. Upload data
   scp -r ./data ubuntu@<instance-ip>:~/

8. Download results
   scp ubuntu@<instance-ip>:~/model.pt ./

Tips:
  • Stop instance when not using (only pay for running time)
  • Use persistent storage for data ($0.20/GB/month)
  • Set up auto-shutdown to save costs
        """)

    @staticmethod
    def comparison_table():
        """
        Compare cloud platforms
        """
        print("\n" + "="*80)
        print("CLOUD PLATFORM COMPARISON")
        print("="*80)

        platforms = [
            {
                "name": "Google Colab (Free)",
                "gpu": "T4 (16GB)",
                "price": "Free",
                "session": "12 hours",
                "best_for": "Learning, small experiments"
            },
            {
                "name": "Google Colab Pro+",
                "gpu": "A100 (40GB)",
                "price": "$50/month",
                "session": "24 hours",
                "best_for": "Regular use, medium models"
            },
            {
                "name": "Lambda Labs",
                "gpu": "A100 (40GB)",
                "price": "$1.10/hour",
                "session": "Unlimited",
                "best_for": "Flexible GPU rental"
            },
            {
                "name": "RunPod",
                "gpu": "Various",
                "price": "$0.39-$3/hour",
                "session": "Unlimited",
                "best_for": "Budget GPU access"
            },
            {
                "name": "Vast.ai",
                "gpu": "Various",
                "price": "$0.10-$2/hour",
                "session": "Unlimited",
                "best_for": "Cheapest GPU option"
            },
            {
                "name": "AWS EC2 (p3.2xlarge)",
                "gpu": "V100 (16GB)",
                "price": "$3.06/hour",
                "session": "Unlimited",
                "best_for": "Enterprise, production"
            },
            {
                "name": "GCP (a2-highgpu-1g)",
                "gpu": "A100 (40GB)",
                "price": "$3.67/hour",
                "session": "Unlimited",
                "best_for": "Enterprise, Google ecosystem"
            },
            {
                "name": "Azure (NC24ads_A100_v4)",
                "gpu": "A100 (80GB)",
                "price": "$3.67/hour",
                "session": "Unlimited",
                "best_for": "Enterprise, Microsoft ecosystem"
            }
        ]

        print("\n{:<25} {:<20} {:<20} {:<15} {:<30}".format(
            "Platform", "GPU", "Price", "Session", "Best For"
        ))
        print("="*110)

        for p in platforms:
            print("{:<25} {:<20} {:<20} {:<15} {:<30}".format(
                p["name"], p["gpu"], p["price"], p["session"], p["best_for"]
            ))

        print("\nRecommendations:")
        print("  • Starting out: Google Colab (free)")
        print("  • Regular experimentation: Colab Pro+ or Lambda Labs")
        print("  • Production: AWS/GCP/Azure with spot instances")
        print("  • Budget-conscious: Vast.ai, RunPod")


# Demo
if __name__ == "__main__":
    cloud = CloudPlatformsGuide()

    cloud.google_colab()
    cloud.lambda_labs()
    cloud.comparison_table()

    print("\n" + "="*80)
    print("✅ CHAPITRE 5 TERMINÉ!")
    print("="*80)
    print("""
Vous avez maintenant:

1. Environment Setup
   ✅ Python (venv, conda, poetry)
   ✅ Verification scripts

2. PyTorch & CUDA
   ✅ GPU setup complete
   ✅ Optimization tips
   ✅ Troubleshooting guide

3. HuggingFace Ecosystem
   ✅ Transformers, Datasets, Accelerate
   ✅ Hub access
   ✅ Quick start examples

4. Development Tools
   ✅ VSCode, Jupyter setup
   ✅ Experiment tracking (W&B, TensorBoard)
   ✅ Git best practices for ML

5. Cloud Platforms
   ✅ Free GPUs (Colab)
   ✅ Budget options (Lambda, RunPod)
   ✅ Enterprise (AWS, GCP, Azure)

Vous êtes prêt à développer et entraîner des LLMs!

Prochaine étape:
  → Chapitre 6: Pré-entraînement de LLMs from Scratch
    (Training complet d'un modèle GPT-style)
    """)
```
