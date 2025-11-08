# Chapitre 14 (Partie 2): GCP et Azure Deployment

## 2. GCP (Google Cloud Platform) Deployment

```python
"""
GCP = Google Cloud Platform

Services for LLMs:
  1. Vertex AI:
     • Managed ML platform (like SageMaker)
     • Prediction endpoints
     • Model Garden (pre-trained models)
     • Best for: Managed deployment

  2. Compute Engine:
     • GPU VMs
     • Full control
     • Best for: Custom setups

  3. GKE (Google Kubernetes Engine):
     • Managed Kubernetes
     • Best for: Scalable production
     • Excellent autoscaling

  4. Cloud Run:
     • Serverless containers
     • Best for: Small models, variable traffic

Instance types:
  • a2-highgpu-1g: 1x A100 40GB ($3.67/hr)
  • a2-ultragpu-1g: 1x A100 80GB ($5.05/hr)
  • a2-ultragpu-8g: 8x A100 80GB ($29.39/hr)
  • g2-standard-4: 1x L4 24GB ($0.96/hr) - Cheapest!

Advantages vs AWS:
  ✅ Better Kubernetes (GKE)
  ✅ TPUs available (for custom models)
  ✅ Simpler pricing
  ✅ Better for data analytics (BigQuery)

Disadvantages:
  ❌ Fewer regions
  ❌ Less mature ML services
  ❌ Smaller ecosystem
"""


class GCPVertexAIDeployment:
    """
    Deploy LLM to GCP Vertex AI

    Example:
        >>> deployer = GCPVertexAIDeployment(
        ...     model_name="llama-2-7b-awq",
        ...     machine_type="g2-standard-4"
        ... )
        >>> endpoint = deployer.deploy()
    """

    def __init__(
        self,
        model_name: str,
        machine_type: str = "g2-standard-4",
        accelerator_type: str = "NVIDIA_L4",
        accelerator_count: int = 1,
        project_id: str = None,
        region: str = "us-central1"
    ):
        """
        Args:
            model_name: Model name or GCS path
            machine_type: GCP machine type
            accelerator_type: GPU type
            accelerator_count: Number of GPUs
            project_id: GCP project ID
            region: GCP region
        """
        self.model_name = model_name
        self.machine_type = machine_type
        self.accelerator_type = accelerator_type
        self.accelerator_count = accelerator_count
        self.project_id = project_id
        self.region = region

    def deploy(self):
        """Deploy model to Vertex AI"""
        print("="*80)
        print(f"DEPLOYING TO GCP VERTEX AI")
        print("="*80)

        print(f"""
# 1. Install GCP SDK
pip install google-cloud-aiplatform

# 2. Authenticate
gcloud auth login
gcloud config set project {self.project_id or 'YOUR_PROJECT_ID'}

# 3. Prepare custom container with vLLM
# Dockerfile
FROM nvcr.io/nvidia/pytorch:23.10-py3

RUN pip install vllm transformers

COPY serve.py /app/serve.py

EXPOSE 8080

CMD ["python", "/app/serve.py"]

# serve.py
from vllm import LLM, SamplingParams
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load model
llm = LLM(model="{self.model_name}", dtype="float16")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    prompt = data.get('prompt', '')

    sampling_params = SamplingParams(
        temperature=data.get('temperature', 0.8),
        top_p=data.get('top_p', 0.95),
        max_tokens=data.get('max_tokens', 256)
    )

    outputs = llm.generate([prompt], sampling_params)

    return jsonify({{
        'text': outputs[0].outputs[0].text
    }})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({{'status': 'healthy'}})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

# 4. Build and push container
gcloud builds submit --tag gcr.io/{self.project_id or 'YOUR_PROJECT'}/llm-server:latest

# 5. Deploy to Vertex AI
from google.cloud import aiplatform

aiplatform.init(
    project='{self.project_id or 'YOUR_PROJECT'}',
    location='{self.region}'
)

# Upload model
model = aiplatform.Model.upload(
    display_name='{self.model_name}',
    serving_container_image_uri='gcr.io/{self.project_id or 'YOUR_PROJECT'}/llm-server:latest',
    serving_container_ports=[8080],
    serving_container_predict_route='/predict',
    serving_container_health_route='/health',
)

# Deploy to endpoint
endpoint = model.deploy(
    deployed_model_display_name='{self.model_name}-endpoint',
    machine_type='{self.machine_type}',
    accelerator_type='{self.accelerator_type}',
    accelerator_count={self.accelerator_count},
    min_replica_count=1,
    max_replica_count=10,
    traffic_percentage=100,
)

# 6. Test endpoint
response = endpoint.predict(instances=[{{
    'prompt': 'Write a poem about AI',
    'max_tokens': 256,
    'temperature': 0.8
}}])

print(response.predictions[0]['text'])


AUTOSCALING:

# Vertex AI handles autoscaling automatically based on:
# - Request rate
# - CPU/GPU utilization
# - Custom metrics

# Configure in deployment:
endpoint = model.deploy(
    min_replica_count=1,      # Minimum instances
    max_replica_count=10,     # Maximum instances
    traffic_split={{"0": 100}},
)


COST OPTIMIZATION:

1. Use L4 GPUs (instead of A100):
   # L4: $0.96/hr (24GB VRAM)
   # A100 40GB: $3.67/hr
   # 4x cheaper, good for 7B-13B models!

2. Preemptible instances:
   # Not available for Vertex AI endpoints
   # Use for batch predictions

3. GKE with spot instances:
   # See Kubernetes section below


MONITORING:

# Vertex AI provides built-in monitoring
# View in Cloud Console:
# - Request rate
# - Latency (p50, p95, p99)
# - Error rate
# - Resource utilization

# Or use Cloud Monitoring API:
from google.cloud import monitoring_v3

client = monitoring_v3.MetricServiceClient()
project_name = f"projects/{self.project_id or 'YOUR_PROJECT'}"

# Query metrics
results = client.list_time_series(
    name=project_name,
    filter='metric.type="aiplatform.googleapis.com/prediction/online/latency"'
)
        """)


def demo_gcp_compute_engine():
    """Demo GCP Compute Engine deployment"""
    print("\n" + "="*80)
    print("GCP COMPUTE ENGINE + vLLM")
    print("="*80)

    print("""
MANUAL DEPLOYMENT:

# 1. Create GPU instance
gcloud compute instances create llm-server \\
    --zone=us-central1-a \\
    --machine-type=g2-standard-4 \\
    --accelerator=type=nvidia-l4,count=1 \\
    --image-family=pytorch-latest-gpu \\
    --image-project=deeplearning-platform-release \\
    --maintenance-policy=TERMINATE \\
    --boot-disk-size=100GB

# 2. SSH into instance
gcloud compute ssh llm-server --zone=us-central1-a

# 3. Install vLLM
pip install vllm

# 4. Run server
python -m vllm.entrypoints.openai.api_server \\
    --model meta-llama/Llama-2-7b-hf \\
    --dtype float16 \\
    --host 0.0.0.0 \\
    --port 8000

# 5. Create firewall rule
gcloud compute firewall-rules create allow-llm-server \\
    --allow=tcp:8000 \\
    --source-ranges=0.0.0.0/0

# 6. Get external IP
gcloud compute instances describe llm-server \\
    --zone=us-central1-a \\
    --format='get(networkInterfaces[0].accessConfigs[0].natIP)'

# 7. Test
curl http://EXTERNAL_IP:8000/v1/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "meta-llama/Llama-2-7b-hf",
    "prompt": "Write a poem",
    "max_tokens": 256
  }'


TERRAFORM DEPLOYMENT:

# main.tf
provider "google" {
  project = "YOUR_PROJECT_ID"
  region  = "us-central1"
}

resource "google_compute_instance" "llm_server" {
  name         = "llm-server"
  machine_type = "g2-standard-4"
  zone         = "us-central1-a"

  boot_disk {
    initialize_params {
      image = "deeplearning-platform-release/pytorch-latest-gpu"
      size  = 100
    }
  }

  network_interface {
    network = "default"
    access_config {}
  }

  guest_accelerator {
    type  = "nvidia-l4"
    count = 1
  }

  scheduling {
    on_host_maintenance = "TERMINATE"
  }

  metadata_startup_script = <<-EOF
    #!/bin/bash
    pip install vllm
    python -m vllm.entrypoints.openai.api_server \\
      --model meta-llama/Llama-2-7b-hf \\
      --dtype float16
  EOF
}

resource "google_compute_firewall" "llm_server" {
  name    = "allow-llm-server"
  network = "default"

  allow {
    protocol = "tcp"
    ports    = ["8000"]
  }

  source_ranges = ["0.0.0.0/0"]
}

# Deploy
terraform init
terraform apply


SPOT INSTANCES (PREEMPTIBLE):

# 70% cheaper but can be terminated
gcloud compute instances create llm-server-spot \\
    --zone=us-central1-a \\
    --machine-type=g2-standard-4 \\
    --accelerator=type=nvidia-l4,count=1 \\
    --preemptible \\
    --image-family=pytorch-latest-gpu \\
    --image-project=deeplearning-platform-release

# Cost: $0.96/hr → $0.29/hr (spot)


MANAGED INSTANCE GROUP (AUTO-SCALING):

# 1. Create instance template
gcloud compute instance-templates create llm-template \\
    --machine-type=g2-standard-4 \\
    --accelerator=type=nvidia-l4,count=1 \\
    --image-family=pytorch-latest-gpu \\
    --image-project=deeplearning-platform-release

# 2. Create managed instance group
gcloud compute instance-groups managed create llm-group \\
    --base-instance-name=llm-server \\
    --template=llm-template \\
    --size=1 \\
    --zone=us-central1-a

# 3. Configure autoscaling
gcloud compute instance-groups managed set-autoscaling llm-group \\
    --max-num-replicas=10 \\
    --min-num-replicas=1 \\
    --target-cpu-utilization=0.7 \\
    --zone=us-central1-a

# 4. Create load balancer
# (Similar to AWS ALB setup)
    """)


if __name__ == "__main__":
    deployer = GCPVertexAIDeployment(
        model_name="meta-llama/Llama-2-7b-hf",
        machine_type="g2-standard-4"
    )
    deployer.deploy()
    demo_gcp_compute_engine()
```

## 3. Azure Deployment

```python
"""
Azure = Microsoft cloud platform

Services for LLMs:
  1. Azure ML:
     • Managed ML platform
     • Online endpoints
     • Best for: Enterprise deployment

  2. Azure VMs:
     • GPU virtual machines
     • Full control
     • Best for: Custom setups

  3. AKS (Azure Kubernetes Service):
     • Managed Kubernetes
     • Best for: Production scale

  4. Azure OpenAI Service:
     • Managed API (GPT-4, GPT-3.5)
     • Pay per token
     • Best for: Using OpenAI models

Instance types:
  • Standard_NC6s_v3: 1x V100 16GB ($3.06/hr)
  • Standard_NC24ads_A100_v4: 1x A100 80GB ($3.67/hr)
  • Standard_ND96asr_v4: 8x A100 80GB ($29.40/hr)
  • Standard_NC4as_T4_v3: 1x T4 16GB ($0.526/hr)

Advantages:
  ✅ Best enterprise features
  ✅ Strong security/compliance
  ✅ Good Windows integration
  ✅ Azure OpenAI Service

Disadvantages:
  ❌ Complex pricing
  ❌ Less ML-specific than AWS/GCP
  ❌ Smaller ML community
"""


class AzureMLDeployment:
    """
    Deploy LLM to Azure ML

    Example:
        >>> deployer = AzureMLDeployment(
        ...     model_name="llama-2-7b-awq",
        ...     instance_type="Standard_NC4as_T4_v3"
        ... )
        >>> endpoint = deployer.deploy()
    """

    def __init__(
        self,
        model_name: str,
        instance_type: str = "Standard_NC4as_T4_v3",
        instance_count: int = 1,
        subscription_id: str = None,
        resource_group: str = "llm-rg",
        workspace_name: str = "llm-workspace"
    ):
        self.model_name = model_name
        self.instance_type = instance_type
        self.instance_count = instance_count
        self.subscription_id = subscription_id
        self.resource_group = resource_group
        self.workspace_name = workspace_name

    def deploy(self):
        """Deploy model to Azure ML"""
        print("="*80)
        print(f"DEPLOYING TO AZURE ML")
        print("="*80)

        print(f"""
# 1. Install Azure ML SDK
pip install azure-ai-ml azure-identity

# 2. Authenticate
az login
az account set --subscription {self.subscription_id or 'YOUR_SUBSCRIPTION_ID'}

# 3. Create workspace (if not exists)
az ml workspace create \\
    --name {self.workspace_name} \\
    --resource-group {self.resource_group}

# 4. Prepare deployment files

# conda.yml
name: vllm-env
channels:
  - defaults
dependencies:
  - python=3.10
  - pip:
    - vllm
    - transformers
    - flask

# score.py
import os
from vllm import LLM, SamplingParams

def init():
    global llm
    model_path = os.getenv('AZUREML_MODEL_DIR')
    llm = LLM(model="{self.model_name}", dtype="float16")

def run(raw_data):
    import json
    data = json.loads(raw_data)

    prompt = data.get('prompt', '')
    sampling_params = SamplingParams(
        temperature=data.get('temperature', 0.8),
        max_tokens=data.get('max_tokens', 256)
    )

    outputs = llm.generate([prompt], sampling_params)

    return json.dumps({{
        'text': outputs[0].outputs[0].text
    }})

# 5. Deploy with Python SDK
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Model,
    Environment,
    CodeConfiguration,
)
from azure.identity import DefaultAzureCredential

# Connect to workspace
ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id="{self.subscription_id or 'YOUR_SUBSCRIPTION'}",
    resource_group_name="{self.resource_group}",
    workspace_name="{self.workspace_name}",
)

# Create endpoint
endpoint = ManagedOnlineEndpoint(
    name="{self.model_name}-endpoint",
    description="LLM inference endpoint",
)

endpoint = ml_client.online_endpoints.begin_create_or_update(endpoint).result()

# Create deployment
deployment = ManagedOnlineDeployment(
    name="blue",
    endpoint_name="{self.model_name}-endpoint",
    model=Model(path="./model"),
    environment=Environment(
        conda_file="conda.yml",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04"
    ),
    code_configuration=CodeConfiguration(
        code="./",
        scoring_script="score.py"
    ),
    instance_type="{self.instance_type}",
    instance_count={self.instance_count},
)

deployment = ml_client.online_deployments.begin_create_or_update(deployment).result()

# Route traffic
endpoint.traffic = {{"blue": 100}}
ml_client.online_endpoints.begin_create_or_update(endpoint).result()

# 6. Test endpoint
response = ml_client.online_endpoints.invoke(
    endpoint_name="{self.model_name}-endpoint",
    request_file="request.json"
)

print(response)


AUTOSCALING:

# Azure ML supports autoscaling based on:
# - Request rate
# - CPU/GPU utilization
# - Custom metrics

from azure.ai.ml.entities import OnlineRequestSettings

deployment.request_settings = OnlineRequestSettings(
    request_timeout_ms=90000,
    max_concurrent_requests_per_instance=1,
)

deployment.scale_settings = {{
    "scale_type": "target_utilization",
    "min_instances": 1,
    "max_instances": 10,
    "polling_interval": 1,
    "target_utilization_percentage": 70,
}}


AZURE VM DEPLOYMENT:

# 1. Create VM with GPU
az vm create \\
    --resource-group {self.resource_group} \\
    --name llm-server \\
    --size {self.instance_type} \\
    --image microsoft-dsvm:ubuntu-2004:2004-gen2:latest \\
    --admin-username azureuser \\
    --generate-ssh-keys

# 2. Install NVIDIA drivers
az vm extension set \\
    --resource-group {self.resource_group} \\
    --vm-name llm-server \\
    --name NvidiaGpuDriverLinux \\
    --publisher Microsoft.HpcCompute

# 3. SSH and install vLLM
ssh azureuser@VM_IP
pip install vllm
python -m vllm.entrypoints.openai.api_server --model {self.model_name}


COST OPTIMIZATION:

1. Use Spot VMs (up to 90% discount):
   az vm create \\
       --priority Spot \\
       --max-price -1 \\
       --eviction-policy Deallocate

2. Use cheaper T4 instances:
   # T4: $0.526/hr (16GB)
   # A100: $3.67/hr (80GB)

3. Auto-shutdown when idle:
   # Configure in Azure Portal
   # Or use automation scripts
        """)


def demo_cost_comparison():
    """Compare costs across cloud providers"""
    print("\n" + "="*80)
    print("CLOUD PROVIDER COST COMPARISON")
    print("="*80)

    scenarios = [
        {
            "name": "Small (7B model, 1x L4/T4)",
            "aws": {"instance": "g5.xlarge", "cost": 1.006},
            "gcp": {"instance": "g2-standard-4", "cost": 0.96},
            "azure": {"instance": "NC4as_T4_v3", "cost": 0.526},
        },
        {
            "name": "Medium (13B model, 1x A10G)",
            "aws": {"instance": "g5.2xlarge", "cost": 1.212},
            "gcp": {"instance": "g2-standard-8", "cost": 1.52},
            "azure": {"instance": "NC6s_v3", "cost": 3.06},
        },
        {
            "name": "Large (70B model, 1x A100 80GB)",
            "aws": {"instance": "p4d.24xlarge/8", "cost": 4.10},
            "gcp": {"instance": "a2-ultragpu-1g", "cost": 5.05},
            "azure": {"instance": "NC24ads_A100_v4", "cost": 3.67},
        },
    ]

    for scenario in scenarios:
        print(f"\n{scenario['name']}:")
        print(f"{'Provider':<15} {'Instance':<25} {'Cost/hour':<15} {'Cost/month (24/7)':<20}")
        print("-"*80)

        for provider in ['aws', 'gcp', 'azure']:
            info = scenario[provider]
            monthly = info['cost'] * 730
            print(f"{provider.upper():<15} {info['instance']:<25} ${info['cost']:<14.3f} ${monthly:,.0f}")

    print("\n" + "="*80)
    print("COST SAVING STRATEGIES")
    print("="*80)
    print("""
1. Use Spot/Preemptible instances:
   ✅ AWS Spot: 70% discount
   ✅ GCP Preemptible: 70% discount
   ✅ Azure Spot: 90% discount
   ⚠️ Can be interrupted! Save checkpoints

2. Use quantized models:
   ✅ 4-bit model = 1/4 memory
   ✅ Can use smaller/cheaper instances
   ✅ Example: 70B 4-bit fits on single A100 (not 8x!)

3. Auto-scaling:
   ✅ Scale to zero during idle
   ✅ Scale up during peak hours
   ✅ Typical saving: 40-60%

4. Reserved instances (1-3 year commit):
   ✅ AWS: 30-70% discount
   ✅ GCP: 37-70% discount
   ✅ Azure: 30-72% discount

5. Use cheaper regions:
   ✅ us-central1 (GCP): Cheapest
   ✅ us-east-1 (AWS): Standard pricing
   ✅ Check each provider's pricing page

6. Batch inference:
   ✅ Process requests in batches
   ✅ Higher throughput = fewer instances
   ✅ Use vLLM continuous batching

Example savings:
  • Llama 2 7B on AWS g5.xlarge
    - On-demand: $1.006/hr × 730 = $735/month
    - Spot: $0.30/hr × 730 = $219/month
    - Saving: $516/month (70%)

  • With auto-scaling (40% utilization):
    - Spot cost: $219 × 0.4 = $88/month
    - Total saving: $647/month (88%)
    """)


if __name__ == "__main__":
    deployer = AzureMLDeployment(
        model_name="meta-llama/Llama-2-7b-hf",
        instance_type="Standard_NC4as_T4_v3"
    )
    deployer.deploy()
    demo_cost_comparison()
```

*[Suite avec Kubernetes dans la partie 3...]*
