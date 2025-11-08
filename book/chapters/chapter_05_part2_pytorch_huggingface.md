# Chapitre 5 (Partie 2): PyTorch, CUDA et HuggingFace Ecosystem

## 2. PyTorch et CUDA Setup

```python
"""
PyTorch + CUDA = Accélération GPU essentielle pour LLMs

Sans GPU:
  - Training GPT-2 (117M params): ~100 jours sur CPU
  - Avec GPU V100: ~1 journée
  - Speedup: 100x+

CUDA Setup:
  1. Vérifier compatibilité GPU (Nvidia uniquement)
  2. Installer CUDA Toolkit
  3. Installer cuDNN
  4. Installer PyTorch avec CUDA support
"""

import subprocess
import sys


class CUDASetupGuide:
    """
    Complete CUDA setup guide
    """

    @staticmethod
    def check_gpu_compatibility():
        """
        Check if GPU is CUDA-compatible
        """
        print("="*80)
        print("GPU COMPATIBILITY CHECK")
        print("="*80)

        print("""
Step 1: Check if you have Nvidia GPU

Linux:
  lspci | grep -i nvidia

Windows:
  nvidia-smi

Mac:
  # Unfortunately, Mac doesn't support CUDA (use MPS backend instead)
  # Or use cloud GPUs (Colab, Lambda Labs, etc.)

Common CUDA-compatible GPUs:

Consumer (Gaming):
  • RTX 4090: 24GB VRAM (excellent for fine-tuning)
  • RTX 4080: 16GB VRAM (good for medium models)
  • RTX 3090: 24GB VRAM (great value)
  • RTX 3080: 10-12GB VRAM (minimum for small LLMs)

Professional (Data Center):
  • H100: 80GB VRAM (SOTA, $30k+)
  • A100: 40-80GB VRAM (production standard)
  • V100: 16-32GB VRAM (older but reliable)
  • T4: 16GB VRAM (cost-effective inference)

Minimum for LLM work:
  • 8GB VRAM: Small models (GPT-2, BERT-base)
  • 16GB VRAM: Medium models (GPT-2 XL, Llama 7B with quantization)
  • 24GB+ VRAM: Large models (Llama 13B, fine-tuning)
  • 40GB+ VRAM: Very large models (Llama 70B)
        """)

        try:
            # Try to run nvidia-smi
            result = subprocess.run(
                ['nvidia-smi'],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                print("\n✅ Nvidia GPU detected!")
                print("\nGPU Information:")
                print(result.stdout)
                return True
            else:
                print("\n❌ No Nvidia GPU detected")
                return False

        except FileNotFoundError:
            print("\n❌ nvidia-smi not found")
            print("   Install Nvidia drivers first")
            return False

        except Exception as e:
            print(f"\n⚠️  Error checking GPU: {e}")
            return False

    @staticmethod
    def cuda_installation_guide():
        """
        Guide for installing CUDA
        """
        print("\n" + "="*80)
        print("CUDA TOOLKIT INSTALLATION")
        print("="*80)

        print("""
CUDA = Parallel computing platform for Nvidia GPUs

Step-by-step installation:

1. Check CUDA compatibility
   • Visit: https://developer.nvidia.com/cuda-gpus
   • Find your GPU's compute capability
   • Minimum: Compute Capability 3.5 (most GPUs since 2012)

2. Choose CUDA version
   • CUDA 12.1: Latest (PyTorch 2.1+)
   • CUDA 11.8: Stable, widely supported
   • CUDA 11.7: Older but very compatible

   Check PyTorch compatibility: https://pytorch.org/

3. Download CUDA Toolkit
   • Visit: https://developer.nvidia.com/cuda-downloads
   • Select your OS
   • Download installer (~3GB)

4. Install CUDA

   Linux (Ubuntu):
     wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
     sudo sh cuda_12.1.0_530.30.02_linux.run

   Windows:
     # Download .exe installer
     # Run installer, follow GUI

5. Install cuDNN (optimized deep learning primitives)
   • Visit: https://developer.nvidia.com/cudnn
   • Download cuDNN for your CUDA version
   • Extract and copy to CUDA directory

   Linux:
     tar -xvf cudnn-linux-x86_64-8.x.x.x_cudaX.Y-archive.tar.xz
     sudo cp cudnn-*-archive/include/cudnn*.h /usr/local/cuda/include
     sudo cp -P cudnn-*-archive/lib/libcudnn* /usr/local/cuda/lib64
     sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*

6. Set environment variables

   Linux (add to ~/.bashrc):
     export PATH=/usr/local/cuda/bin:$PATH
     export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

   Windows (System Environment Variables):
     PATH: C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.1\\bin

7. Verify installation
   nvcc --version
   nvidia-smi
        """)

    @staticmethod
    def pytorch_installation_guide():
        """
        PyTorch installation with CUDA
        """
        print("\n" + "="*80)
        print("PYTORCH INSTALLATION WITH CUDA")
        print("="*80)

        print("""
PyTorch = Deep learning framework (used by GPT, Llama, etc.)

Installation options:

1. Visit: https://pytorch.org/get-started/locally/
   • Select your OS, package manager, Python version, CUDA version
   • Copy the installation command

2. Example installations:

   CUDA 12.1 (pip):
     pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

   CUDA 11.8 (pip):
     pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

   CUDA 12.1 (conda):
     conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

   CPU-only (no GPU):
     pip3 install torch torchvision torchaudio

   Mac (MPS - Apple Silicon):
     pip3 install torch torchvision torchaudio

3. Verify installation:
        """)

        print("""
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"cuDNN version: {torch.backends.cudnn.version()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"GPU 0: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Test GPU computation
x = torch.randn(1000, 1000).cuda()
y = torch.randn(1000, 1000).cuda()
z = x @ y
print(f"\\nGPU computation successful: {z.shape}")
        """)

    @staticmethod
    def troubleshooting():
        """
        Common issues and solutions
        """
        print("\n" + "="*80)
        print("CUDA TROUBLESHOOTING")
        print("="*80)

        issues = [
            {
                "problem": "torch.cuda.is_available() returns False",
                "solutions": [
                    "Check nvidia-smi works",
                    "Reinstall PyTorch with correct CUDA version",
                    "Check CUDA_PATH environment variable",
                    "Verify CUDA and PyTorch versions match"
                ]
            },
            {
                "problem": "CUDA out of memory",
                "solutions": [
                    "Reduce batch size",
                    "Use gradient accumulation",
                    "Use mixed precision (FP16)",
                    "Use gradient checkpointing",
                    "Clear cache: torch.cuda.empty_cache()"
                ]
            },
            {
                "problem": "Slow training on GPU",
                "solutions": [
                    "Check GPU utilization: nvidia-smi",
                    "Increase batch size (utilize GPU memory)",
                    "Use torch.compile() (PyTorch 2.0+)",
                    "Use DataLoader with num_workers > 0",
                    "Use pin_memory=True in DataLoader"
                ]
            },
            {
                "problem": "Version mismatch errors",
                "solutions": [
                    "Create fresh virtual environment",
                    "Uninstall and reinstall PyTorch",
                    "Check compatibility matrix",
                    "Use conda (handles dependencies better)"
                ]
            }
        ]

        for issue in issues:
            print(f"\n{issue['problem']}")
            print("  Solutions:")
            for solution in issue['solutions']:
                print(f"    • {solution}")


class GPUOptimizationTips:
    """
    Tips for optimizing GPU usage
    """

    @staticmethod
    def print_optimization_guide():
        """
        Print GPU optimization guide
        """
        print("\n" + "="*80)
        print("GPU OPTIMIZATION TIPS")
        print("="*80)

        print("""
1. Monitor GPU utilization
   # Terminal 1: Training
   python train.py

   # Terminal 2: Monitor
   watch -n 1 nvidia-smi

   Target: GPU utilization 90-100%

2. Maximize batch size
   • Larger batch = better GPU utilization
   • Find max batch size without OOM
   • Use gradient accumulation if needed

   # Binary search for optimal batch size
   batch_sizes = [64, 32, 16, 8, 4]
   for bs in batch_sizes:
       try:
           train(batch_size=bs)
           print(f"Max batch size: {bs}")
           break
       except RuntimeError:  # OOM
           continue

3. Use mixed precision (AMP)
   • 2x faster training
   • 2x less memory
   • Minimal accuracy loss

   from torch.cuda.amp import autocast, GradScaler

   scaler = GradScaler()
   for data, target in loader:
       with autocast():
           output = model(data)
           loss = criterion(output, target)
       scaler.scale(loss).backward()
       scaler.step(optimizer)
       scaler.update()

4. DataLoader optimization
   • num_workers: CPU cores for data loading
   • pin_memory: Faster CPU→GPU transfer
   • prefetch_factor: Pre-load batches

   loader = DataLoader(
       dataset,
       batch_size=32,
       num_workers=4,      # 4 CPU cores
       pin_memory=True,    # Faster transfer
       prefetch_factor=2   # Pre-load 2 batches
   )

5. Gradient checkpointing
   • Trade computation for memory
   • Useful for large models

   from torch.utils.checkpoint import checkpoint

   def forward_with_checkpoint(x):
       return checkpoint(model.layer, x)

6. Use torch.compile() (PyTorch 2.0+)
   • 30-200% speedup
   • Zero code changes

   model = torch.compile(model)

7. Profile your code
   • Find bottlenecks

   from torch.profiler import profile, ProfilerActivity

   with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
       model(input)
   print(prof.key_averages().table())

8. Clear cache between runs
   torch.cuda.empty_cache()
        """)


# Demo
if __name__ == "__main__":
    cuda_guide = CUDASetupGuide()

    # Check GPU
    cuda_guide.check_gpu_compatibility()

    # Installation guides
    cuda_guide.cuda_installation_guide()
    cuda_guide.pytorch_installation_guide()

    # Troubleshooting
    cuda_guide.troubleshooting()

    # Optimization
    gpu_opt = GPUOptimizationTips()
    gpu_opt.print_optimization_guide()
```

## 3. HuggingFace Ecosystem

```python
"""
HuggingFace = GitHub de l'AI

Composants principaux:
  • Transformers: Pre-trained models
  • Datasets: Ready-to-use datasets
  • Tokenizers: Fast tokenization
  • Accelerate: Distributed training
  • PEFT: Parameter-efficient fine-tuning
  • Hub: Model sharing

L'écosystème le plus utilisé pour LLMs!
"""

class HuggingFaceSetup:
    """
    Complete HuggingFace ecosystem setup
    """

    @staticmethod
    def installation_guide():
        """
        Install HuggingFace libraries
        """
        print("="*80)
        print("HUGGINGFACE ECOSYSTEM INSTALLATION")
        print("="*80)

        print("""
Core libraries:

1. transformers (must-have)
   pip install transformers

2. datasets (training data)
   pip install datasets

3. tokenizers (fast tokenization)
   pip install tokenizers

4. accelerate (distributed training)
   pip install accelerate

5. peft (LoRA, adapters)
   pip install peft

6. bitsandbytes (quantization)
   pip install bitsandbytes

7. evaluate (metrics)
   pip install evaluate

Install all at once:
  pip install transformers[torch] datasets tokenizers accelerate peft bitsandbytes evaluate

Or with specific versions (recommended for reproducibility):
  pip install transformers==4.35.0 datasets==2.15.0 accelerate==0.24.0 peft==0.7.0
        """)

    @staticmethod
    def hub_setup():
        """
        Setup HuggingFace Hub access
        """
        print("\n" + "="*80)
        print("HUGGINGFACE HUB SETUP")
        print("="*80)

        print("""
HuggingFace Hub = Model & dataset repository (like GitHub for AI)

1. Create account
   • Visit: https://huggingface.co/join
   • Free account with generous limits

2. Get access token
   • Go to: https://huggingface.co/settings/tokens
   • Click "New token"
   • Name: "dev-machine" or similar
   • Role: "write" (for uploading models)
   • Copy token (starts with hf_...)

3. Login from CLI
   huggingface-cli login

   # Or programmatically
   from huggingface_hub import login
   login(token="hf_...")

   # Or save to environment variable
   export HF_TOKEN="hf_..."

4. Access gated models (Llama 2, Mistral, etc.)
   • Visit model page (e.g., meta-llama/Llama-2-7b-hf)
   • Click "Request access"
   • Wait for approval (usually instant)
   • Use same token to download

5. Download models

   from transformers import AutoModelForCausalLM, AutoTokenizer

   # Authenticate
   from huggingface_hub import login
   login(token="hf_...")

   # Download
   model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
   tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

6. Upload models

   model.push_to_hub("username/my-finetuned-model")
   tokenizer.push_to_hub("username/my-finetuned-model")

7. Cache directory
   • Default: ~/.cache/huggingface/
   • Can be large (100GB+)
   • Change with: export HF_HOME=/path/to/cache

   # Clear cache
   rm -rf ~/.cache/huggingface/

   # Or use CLI
   huggingface-cli delete-cache
        """)

    @staticmethod
    def transformers_quickstart():
        """
        Quick start with Transformers
        """
        print("\n" + "="*80)
        print("TRANSFORMERS LIBRARY QUICKSTART")
        print("="*80)

        print("""
Transformers = Access 100k+ pre-trained models

Basic usage:

1. Text generation (GPT-style)
        """)

        print('''
from transformers import pipeline

# Create pipeline
generator = pipeline('text-generation', model='gpt2')

# Generate text
output = generator(
    "Once upon a time",
    max_length=50,
    num_return_sequences=1
)

print(output[0]['generated_text'])
        ''')

        print("""
2. Classification
        """)

        print('''
classifier = pipeline('sentiment-analysis')
result = classifier("I love this book!")
print(result)  # [{'label': 'POSITIVE', 'score': 0.999}]
        ''')

        print("""
3. Question answering
        """)

        print('''
qa = pipeline('question-answering')
result = qa(
    question="What is the capital of France?",
    context="Paris is the capital and largest city of France."
)
print(result)  # {'answer': 'Paris', 'score': 0.998}
        ''')

        print("""
4. Manual model loading (more control)
        """)

        print('''
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Tokenize
inputs = tokenizer("Hello, world!", return_tensors="pt")

# Generate
outputs = model.generate(
    inputs["input_ids"],
    max_length=50,
    temperature=0.8,
    top_p=0.95,
    do_sample=True
)

# Decode
text = tokenizer.decode(outputs[0])
print(text)
        ''')

    @staticmethod
    def datasets_quickstart():
        """
        Quick start with Datasets
        """
        print("\n" + "="*80)
        print("DATASETS LIBRARY QUICKSTART")
        print("="*80)

        print("""
Datasets = 50k+ ready-to-use datasets

Basic usage:

1. Load dataset
        """)

        print('''
from datasets import load_dataset

# Load dataset
dataset = load_dataset("squad")

print(dataset)
# DatasetDict({
#     train: Dataset({features: ['id', 'title', 'context', 'question', 'answers'], num_rows: 87599})
#     validation: Dataset({features: ['id', 'title', 'context', 'question', 'answers'], num_rows: 10570})
# })

# Access data
sample = dataset['train'][0]
print(sample['question'])
print(sample['answers'])
        ''')

        print("""
2. Process dataset
        """)

        print('''
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    return tokenizer(
        examples["question"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

# Apply to entire dataset (fast, parallel)
tokenized = dataset.map(
    tokenize_function,
    batched=True,
    num_proc=4  # Use 4 CPU cores
)

print(tokenized['train'][0])
# {'input_ids': [...], 'attention_mask': [...]}
        ''')

        print("""
3. Stream large datasets
        """)

        print('''
# Don't download entire dataset (useful for 100GB+ datasets)
dataset = load_dataset("c4", "en", streaming=True)

for example in dataset['train'].take(10):
    print(example['text'][:100])
        ''')

    @staticmethod
    def accelerate_setup():
        """
        Accelerate setup for distributed training
        """
        print("\n" + "="*80)
        print("ACCELERATE SETUP")
        print("="*80)

        print("""
Accelerate = Simplifies distributed training

Setup:

1. Install
   pip install accelerate

2. Configure
   accelerate config

   Questions:
   - Compute environment? [0] This machine
   - Machine type? [0] No distributed training / [1] Multi-GPU / [2] TPU
   - Mixed precision? [fp16] or [bf16]
   - Number of GPUs? [1, 2, 4, 8...]

   Creates: ~/.cache/huggingface/accelerate/default_config.yaml

3. Example config (2 GPUs, fp16):
        """)

        print('''
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: 2
use_cpu: false
        ''')

        print("""
4. Use in training script
        """)

        print('''
from accelerate import Accelerator

# Initialize
accelerator = Accelerator()

# Prepare model, optimizer, dataloader
model, optimizer, train_loader = accelerator.prepare(
    model, optimizer, train_loader
)

# Training loop (same code for 1 GPU, multi-GPU, TPU!)
for batch in train_loader:
    outputs = model(**batch)
    loss = outputs.loss

    accelerator.backward(loss)
    optimizer.step()
    optimizer.zero_grad()

# Save model
accelerator.wait_for_everyone()
unwrapped_model = accelerator.unwrap_model(model)
unwrapped_model.save_pretrained("output")
        ''')

        print("""
5. Launch training
   # Single GPU
   python train.py

   # Multi-GPU (uses accelerate config)
   accelerate launch train.py

   # Override config
   accelerate launch --num_processes 4 --mixed_precision fp16 train.py
        """)


# Demo
if __name__ == "__main__":
    hf_setup = HuggingFaceSetup()

    hf_setup.installation_guide()
    hf_setup.hub_setup()
    hf_setup.transformers_quickstart()
    hf_setup.datasets_quickstart()
    hf_setup.accelerate_setup()

    print("\n" + "="*80)
    print("KEY TAKEAWAYS - HUGGINGFACE")
    print("="*80)
    print("""
1. Transformers: 100k+ pre-trained models
   • AutoModel*, pipeline
   • GPT, BERT, T5, Llama, etc.

2. Datasets: 50k+ datasets
   • load_dataset, map, filter
   • Streaming for large datasets

3. Hub: Share models & datasets
   • huggingface-cli login
   • push_to_hub, from_pretrained

4. Accelerate: Distributed training made easy
   • Same code for 1 GPU, multi-GPU, TPU
   • accelerate config, accelerate launch

5. PEFT: Parameter-efficient fine-tuning
   • LoRA, adapters (covered in Chapter 8)

Essential for modern LLM development!
    """)
```

*[Suite avec Development Tools et Cloud Platforms dans la partie 3...]*
