# Chapitre 14: Déploiement Cloud de LLMs

## Introduction

Le **déploiement cloud** permet de scaler les LLMs en production, gérer la haute disponibilité et optimiser les coûts.

### Choix de Cloud Provider

```python
"""
Cloud Deployment = Production LLM serving at scale

Challenges:
  • Cost: GPUs are expensive ($1-10/hour)
  • Scale: Handle 1M+ requests/day
  • Latency: < 100ms first token
  • Availability: 99.9%+ uptime
  • Security: Data privacy, compliance

Cloud Providers:
  1. AWS (Amazon Web Services)
     • SageMaker: Managed ML platform
     • EC2 P4/G5: GPU instances
     • Lambda: Serverless (for small models)
     • Pros: Mature, most services
     • Cons: Complex, expensive

  2. GCP (Google Cloud Platform)
     • Vertex AI: Managed ML
     • Compute Engine: GPU VMs
     • Pros: Best for Kubernetes, TPUs
     • Cons: Fewer regions

  3. Azure (Microsoft)
     • Azure ML: Managed ML
     • AKS: Kubernetes
     • Pros: Enterprise features
     • Cons: Less ML-specific

  4. Specialized (for LLMs)
     • Lambda Labs: Cheap GPUs
     • RunPod: GPU cloud
     • Vast.ai: Spot instances
     • Together AI: LLM-specific

Cost Comparison (A100 80GB):
  • AWS: $4.10/hour
  • GCP: $3.67/hour
  • Azure: $3.67/hour
  • Lambda Labs: $1.10/hour
  • Vast.ai: $0.60-1.50/hour (spot)

Decision factors:
  ✅ Budget: Specialized providers for cost
  ✅ Scale: AWS/GCP/Azure for enterprise
  ✅ Compliance: AWS/GCP/Azure (HIPAA, SOC2)
  ✅ Ease: Specialized for simplicity
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
import json


@dataclass
class CloudProvider:
    """Cloud provider configuration"""
    name: str
    gpu_instance: str  # Instance type
    gpu_type: str      # GPU model
    gpu_memory_gb: int
    cost_per_hour: float
    spot_available: bool
    spot_cost_per_hour: Optional[float]
    regions: List[str]
    managed_ml: str    # Managed ML service


# Major cloud providers
CLOUD_PROVIDERS = [
    CloudProvider(
        name="AWS",
        gpu_instance="p4d.24xlarge",
        gpu_type="8x A100 80GB",
        gpu_memory_gb=640,
        cost_per_hour=32.77,
        spot_available=True,
        spot_cost_per_hour=10.0,
        regions=["us-east-1", "us-west-2", "eu-west-1", "ap-south-1"],
        managed_ml="SageMaker"
    ),
    CloudProvider(
        name="GCP",
        gpu_instance="a2-ultragpu-8g",
        gpu_type="8x A100 80GB",
        gpu_memory_gb=640,
        cost_per_hour=29.39,
        spot_available=True,
        spot_cost_per_hour=8.82,
        regions=["us-central1", "us-east4", "europe-west4"],
        managed_ml="Vertex AI"
    ),
    CloudProvider(
        name="Azure",
        gpu_instance="Standard_ND96asr_v4",
        gpu_type="8x A100 80GB",
        gpu_memory_gb=640,
        cost_per_hour=29.40,
        spot_available=True,
        spot_cost_per_hour=8.82,
        regions=["eastus", "westus2", "westeurope"],
        managed_ml="Azure ML"
    ),
    CloudProvider(
        name="Lambda Labs",
        gpu_instance="gpu_8x_a100_80gb",
        gpu_type="8x A100 80GB",
        gpu_memory_gb=640,
        cost_per_hour=8.80,
        spot_available=False,
        spot_cost_per_hour=None,
        regions=["us-west-1", "us-east-1"],
        managed_ml=None
    ),
]


def print_cloud_comparison():
    """Print cloud provider comparison"""
    print("="*120)
    print("CLOUD PROVIDERS COMPARISON (8x A100 80GB)")
    print("="*120)

    print(f"\n{'Provider':<15} {'Instance':<25} {'On-Demand':<15} {'Spot':<15} {'Savings':<15} {'Managed ML':<15}")
    print("-"*120)

    for provider in CLOUD_PROVIDERS:
        spot_str = f"${provider.spot_cost_per_hour:.2f}/hr" if provider.spot_available else "N/A"

        if provider.spot_available:
            savings = (1 - provider.spot_cost_per_hour / provider.cost_per_hour) * 100
            savings_str = f"{savings:.0f}%"
        else:
            savings_str = "N/A"

        ml_str = provider.managed_ml or "Manual"

        print(f"{provider.name:<15} {provider.gpu_instance:<25} ${provider.cost_per_hour:<14.2f}/hr "
              f"{spot_str:<15} {savings_str:<15} {ml_str:<15}")

    print("\n" + "="*120)
    print("COST ANALYSIS (monthly, 24/7)")
    print("="*120)

    print(f"\n{'Provider':<15} {'On-Demand':<20} {'Spot':<20} {'Savings':<15}")
    print("-"*120)

    hours_per_month = 730  # ~30 days

    for provider in CLOUD_PROVIDERS:
        on_demand_monthly = provider.cost_per_hour * hours_per_month

        if provider.spot_available:
            spot_monthly = provider.spot_cost_per_hour * hours_per_month
            savings_monthly = on_demand_monthly - spot_monthly
            spot_str = f"${spot_monthly:,.0f}/mo"
            savings_str = f"${savings_monthly:,.0f}/mo"
        else:
            spot_str = "N/A"
            savings_str = "N/A"

        print(f"{provider.name:<15} ${on_demand_monthly:,.0f}/mo {spot_str:<20} {savings_str:<15}")

    print("\n" + "="*120)
    print("KEY INSIGHTS")
    print("="*120)
    print("""
1. Specialized providers (Lambda Labs, Vast.ai):
   ✅ 3-5x cheaper than AWS/GCP/Azure
   ✅ Simple, LLM-focused
   ✅ Good for: Startups, development
   ❌ Less: Enterprise features, compliance

2. Major cloud providers (AWS, GCP, Azure):
   ✅ Enterprise-grade
   ✅ Compliance (HIPAA, SOC2, etc.)
   ✅ Managed ML services
   ❌ 3-5x more expensive

3. Spot instances:
   ✅ 70% cost reduction
   ✅ Good for: Batch processing, non-critical
   ❌ Can be interrupted (save checkpoints!)

4. Cost optimization strategies:
   ✅ Use spot instances
   ✅ Auto-scale (scale to zero when idle)
   ✅ Use smaller models when possible
   ✅ Quantization (4-bit = 4x cheaper)
   ✅ Multi-region (cheaper regions)
    """)


if __name__ == "__main__":
    print_cloud_comparison()
```

## 1. AWS Deployment

```python
"""
AWS = Largest cloud provider, most mature ML services

Services for LLMs:
  1. SageMaker:
     • Managed ML platform
     • Built-in endpoints
     • Auto-scaling
     • Best for: Production inference

  2. EC2 (Elastic Compute Cloud):
     • Raw GPU instances
     • Full control
     • Best for: Custom setups

  3. Lambda:
     • Serverless
     • Best for: Small models, low traffic

  4. Bedrock:
     • Managed LLM API
     • Pre-trained models (Claude, Llama, etc.)
     • Pay per token

Instance types:
  • p4d.24xlarge: 8x A100 80GB ($32.77/hr)
  • p4de.24xlarge: 8x A100 80GB + NVME ($40.97/hr)
  • g5.xlarge: 1x A10G 24GB ($1.006/hr)
  • g5.12xlarge: 4x A10G 24GB ($5.672/hr)

Deployment patterns:
  1. SageMaker Real-time Endpoint
  2. SageMaker Serverless
  3. EC2 + vLLM + Load Balancer
  4. EKS (Kubernetes)
"""


class AWSSageMakerDeployment:
    """
    Deploy LLM to AWS SageMaker

    Example:
        >>> deployer = AWSSageMakerDeployment(
        ...     model_name="llama-2-7b-awq",
        ...     instance_type="ml.g5.2xlarge"
        ... )
        >>> endpoint = deployer.deploy()
    """

    def __init__(
        self,
        model_name: str,
        instance_type: str = "ml.g5.2xlarge",
        initial_instance_count: int = 1,
        region: str = "us-east-1"
    ):
        """
        Args:
            model_name: Model name or S3 path
            instance_type: SageMaker instance type
            initial_instance_count: Number of instances
            region: AWS region
        """
        self.model_name = model_name
        self.instance_type = instance_type
        self.initial_instance_count = initial_instance_count
        self.region = region

    def deploy(self):
        """Deploy model to SageMaker"""
        print("="*80)
        print(f"DEPLOYING TO AWS SAGEMAKER")
        print("="*80)

        print(f"""
# 1. Install AWS SDK
pip install boto3 sagemaker

# 2. Configure AWS credentials
aws configure
# Enter: Access Key ID, Secret Access Key, Region

# 3. Prepare model for SageMaker
import sagemaker
from sagemaker.huggingface import HuggingFaceModel

sess = sagemaker.Session()
role = sagemaker.get_execution_role()

# Model configuration
hub_config = {{
    'HF_MODEL_ID': '{self.model_name}',
    'HF_TASK': 'text-generation',
    'MAX_INPUT_LENGTH': '2048',
    'MAX_TOTAL_TOKENS': '4096',
}}

# Create HuggingFace model
huggingface_model = HuggingFaceModel(
    env=hub_config,
    role=role,
    transformers_version='4.37',
    pytorch_version='2.1',
    py_version='py310',
)

# 4. Deploy to endpoint
predictor = huggingface_model.deploy(
    initial_instance_count={self.initial_instance_count},
    instance_type='{self.instance_type}',
    endpoint_name='{self.model_name}-endpoint',
)

# 5. Test endpoint
response = predictor.predict({{
    'inputs': 'Write a poem about AI',
    'parameters': {{
        'max_new_tokens': 256,
        'temperature': 0.8,
        'top_p': 0.95,
    }}
}})

print(response[0]['generated_text'])


AUTOSCALING:

# Enable auto-scaling
import boto3

client = boto3.client('application-autoscaling')

# Register scalable target
client.register_scalable_target(
    ServiceNamespace='sagemaker',
    ResourceId=f'endpoint/{self.model_name}-endpoint/variant/AllTraffic',
    ScalableDimension='sagemaker:variant:DesiredInstanceCount',
    MinCapacity=1,
    MaxCapacity=10,
)

# Target tracking scaling policy
client.put_scaling_policy(
    PolicyName='TargetTrackingScaling',
    ServiceNamespace='sagemaker',
    ResourceId=f'endpoint/{self.model_name}-endpoint/variant/AllTraffic',
    ScalableDimension='sagemaker:variant:DesiredInstanceCount',
    PolicyType='TargetTrackingScaling',
    TargetTrackingScalingPolicyConfiguration={{
        'TargetValue': 70.0,  # Target 70% utilization
        'PredefinedMetricSpecification': {{
            'PredefinedMetricType': 'SageMakerVariantInvocationsPerInstance'
        }},
        'ScaleInCooldown': 300,
        'ScaleOutCooldown': 60,
    }}
)


COST OPTIMIZATION:

1. Use Serverless Inference (for low traffic):
   serverless_config = {{
       'MemorySizeInMB': 6144,
       'MaxConcurrency': 10,
   }}

   predictor = huggingface_model.deploy(
       serverless_inference_config=serverless_config
   )

2. Use Spot instances (70% cheaper):
   # Not available for real-time endpoints
   # Use for batch transform

3. Use quantized models:
   # 4-bit model = 1/4 the memory = smaller instance

4. Multi-model endpoints:
   # Host multiple models on same instance
        """)


def demo_aws_ec2_deployment():
    """Demo EC2 deployment with vLLM"""
    print("\n" + "="*80)
    print("AWS EC2 + vLLM DEPLOYMENT")
    print("="*80)

    print("""
MANUAL EC2 DEPLOYMENT:

# 1. Launch EC2 instance
#    Instance type: g5.2xlarge (1x A10G 24GB)
#    AMI: Deep Learning AMI (Ubuntu 22.04)
#    Storage: 100GB EBS
#    Security group: Allow port 8000

# 2. SSH into instance
ssh -i your-key.pem ubuntu@your-instance-ip

# 3. Install vLLM
pip install vllm

# 4. Run inference server
python -m vllm.entrypoints.openai.api_server \\
    --model meta-llama/Llama-2-7b-hf \\
    --dtype float16 \\
    --max-model-len 4096 \\
    --host 0.0.0.0 \\
    --port 8000

# 5. Test from local machine
curl http://your-instance-ip:8000/v1/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "meta-llama/Llama-2-7b-hf",
    "prompt": "Write a poem",
    "max_tokens": 256
  }'


PRODUCTION SETUP WITH LOAD BALANCER:

1. Create Launch Template:
   - AMI: Custom AMI with vLLM pre-installed
   - User data script to start vLLM on boot

2. Auto Scaling Group:
   - Min: 1 instance
   - Max: 10 instances
   - Target: CPU 70%

3. Application Load Balancer:
   - Distribute traffic across instances
   - Health checks on /health endpoint

4. Route 53:
   - DNS for your API
   - e.g., api.your-domain.com


TERRAFORM AUTOMATION:

# main.tf
provider "aws" {
  region = "us-east-1"
}

resource "aws_instance" "llm_server" {
  ami           = "ami-xxxx"  # Deep Learning AMI
  instance_type = "g5.2xlarge"

  user_data = <<-EOF
    #!/bin/bash
    pip install vllm
    python -m vllm.entrypoints.openai.api_server \\
      --model meta-llama/Llama-2-7b-hf \\
      --dtype float16
  EOF

  tags = {
    Name = "LLM-Inference-Server"
  }
}

# Deploy
terraform init
terraform apply


COST ESTIMATE (monthly):

Instance: g5.2xlarge (24/7)
  • On-demand: $1.212/hr × 730hrs = $885/month
  • Spot: ~$0.36/hr × 730hrs = $263/month (70% savings)

Storage: 100GB EBS
  • $10/month

Data transfer:
  • First 1GB: Free
  • Next 10TB: $0.09/GB
  • Assume 1TB/month: $90/month

Total (spot): ~$363/month for single instance
    """)


if __name__ == "__main__":
    print("="*80)
    print("AWS DEPLOYMENT OPTIONS")
    print("="*80)

    deployer = AWSSageMakerDeployment(
        model_name="meta-llama/Llama-2-7b-hf",
        instance_type="ml.g5.2xlarge"
    )

    deployer.deploy()
    demo_aws_ec2_deployment()
```

*[Suite avec GCP et Azure dans la partie 2...]*
