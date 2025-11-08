# Chapitre 14 (Partie 3): Kubernetes Deployment

## 4. Kubernetes Deployment

```python
"""
Kubernetes = Container orchestration platform

Why Kubernetes for LLMs:
  ✅ Multi-cloud: Works on AWS, GCP, Azure
  ✅ Auto-scaling: Scale based on load
  ✅ Load balancing: Distribute traffic
  ✅ Rolling updates: Zero-downtime deployments
  ✅ Self-healing: Restart failed pods
  ✅ Resource management: CPU/GPU limits

Managed Kubernetes:
  • EKS (AWS): Elastic Kubernetes Service
  • GKE (GCP): Google Kubernetes Engine
  • AKS (Azure): Azure Kubernetes Service

Components:
  • Pod: Container running vLLM
  • Deployment: Manages replicas
  • Service: Load balancer
  • HPA: Horizontal Pod Autoscaler
  • Ingress: External access

GPU Support:
  • NVIDIA GPU Operator
  • Device plugin for GPU scheduling
  • GPU resource requests/limits
"""

from typing import Dict, List
import yaml


class KubernetesLLMDeployment:
    """
    Deploy LLM to Kubernetes

    Example:
        >>> deployer = KubernetesLLMDeployment(
        ...     model_name="llama-2-7b-awq",
        ...     replicas=3,
        ...     gpu_type="nvidia.com/gpu"
        ... )
        >>> deployer.generate_manifests()
    """

    def __init__(
        self,
        model_name: str,
        namespace: str = "llm-inference",
        replicas: int = 2,
        gpu_count: int = 1,
        gpu_type: str = "nvidia.com/gpu",
        container_image: str = "vllm/vllm-openai:latest",
        service_type: str = "LoadBalancer"
    ):
        self.model_name = model_name
        self.namespace = namespace
        self.replicas = replicas
        self.gpu_count = gpu_count
        self.gpu_type = gpu_type
        self.container_image = container_image
        self.service_type = service_type

    def generate_namespace(self) -> Dict:
        """Generate namespace manifest"""
        return {
            "apiVersion": "v1",
            "kind": "Namespace",
            "metadata": {
                "name": self.namespace
            }
        }

    def generate_deployment(self) -> Dict:
        """Generate deployment manifest"""
        return {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": f"{self.model_name}-deployment",
                "namespace": self.namespace,
                "labels": {
                    "app": self.model_name
                }
            },
            "spec": {
                "replicas": self.replicas,
                "selector": {
                    "matchLabels": {
                        "app": self.model_name
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": self.model_name
                        }
                    },
                    "spec": {
                        "containers": [{
                            "name": "vllm",
                            "image": self.container_image,
                            "command": [
                                "python", "-m",
                                "vllm.entrypoints.openai.api_server"
                            ],
                            "args": [
                                "--model", self.model_name,
                                "--dtype", "float16",
                                "--max-model-len", "4096",
                                "--host", "0.0.0.0",
                                "--port", "8000"
                            ],
                            "ports": [{
                                "containerPort": 8000,
                                "name": "http"
                            }],
                            "resources": {
                                "limits": {
                                    self.gpu_type: self.gpu_count,
                                    "memory": "24Gi",
                                    "cpu": "8"
                                },
                                "requests": {
                                    self.gpu_type: self.gpu_count,
                                    "memory": "16Gi",
                                    "cpu": "4"
                                }
                            },
                            "env": [
                                {
                                    "name": "CUDA_VISIBLE_DEVICES",
                                    "value": "0"
                                },
                                {
                                    "name": "HF_HOME",
                                    "value": "/cache"
                                }
                            ],
                            "volumeMounts": [{
                                "name": "cache",
                                "mountPath": "/cache"
                            }],
                            "livenessProbe": {
                                "httpGet": {
                                    "path": "/health",
                                    "port": 8000
                                },
                                "initialDelaySeconds": 60,
                                "periodSeconds": 10
                            },
                            "readinessProbe": {
                                "httpGet": {
                                    "path": "/health",
                                    "port": 8000
                                },
                                "initialDelaySeconds": 60,
                                "periodSeconds": 5
                            }
                        }],
                        "volumes": [{
                            "name": "cache",
                            "emptyDir": {}
                        }],
                        "nodeSelector": {
                            "cloud.google.com/gke-accelerator": "nvidia-tesla-l4"  # GKE example
                        },
                        "tolerations": [{
                            "key": "nvidia.com/gpu",
                            "operator": "Exists",
                            "effect": "NoSchedule"
                        }]
                    }
                }
            }
        }

    def generate_service(self) -> Dict:
        """Generate service manifest"""
        return {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": f"{self.model_name}-service",
                "namespace": self.namespace
            },
            "spec": {
                "type": self.service_type,
                "selector": {
                    "app": self.model_name
                },
                "ports": [{
                    "protocol": "TCP",
                    "port": 80,
                    "targetPort": 8000
                }]
            }
        }

    def generate_hpa(self) -> Dict:
        """Generate Horizontal Pod Autoscaler manifest"""
        return {
            "apiVersion": "autoscaling/v2",
            "kind": "HorizontalPodAutoscaler",
            "metadata": {
                "name": f"{self.model_name}-hpa",
                "namespace": self.namespace
            },
            "spec": {
                "scaleTargetRef": {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "name": f"{self.model_name}-deployment"
                },
                "minReplicas": 1,
                "maxReplicas": 10,
                "metrics": [
                    {
                        "type": "Resource",
                        "resource": {
                            "name": "cpu",
                            "target": {
                                "type": "Utilization",
                                "averageUtilization": 70
                            }
                        }
                    },
                    {
                        "type": "Resource",
                        "resource": {
                            "name": "memory",
                            "target": {
                                "type": "Utilization",
                                "averageUtilization": 80
                            }
                        }
                    }
                ],
                "behavior": {
                    "scaleDown": {
                        "stabilizationWindowSeconds": 300,
                        "policies": [{
                            "type": "Percent",
                            "value": 50,
                            "periodSeconds": 60
                        }]
                    },
                    "scaleUp": {
                        "stabilizationWindowSeconds": 0,
                        "policies": [{
                            "type": "Percent",
                            "value": 100,
                            "periodSeconds": 15
                        }]
                    }
                }
            }
        }

    def generate_manifests(self, output_dir: str = "./k8s"):
        """Generate all Kubernetes manifests"""
        import os
        os.makedirs(output_dir, exist_ok=True)

        manifests = {
            "namespace.yaml": self.generate_namespace(),
            "deployment.yaml": self.generate_deployment(),
            "service.yaml": self.generate_service(),
            "hpa.yaml": self.generate_hpa(),
        }

        for filename, manifest in manifests.items():
            filepath = os.path.join(output_dir, filename)
            with open(filepath, 'w') as f:
                yaml.dump(manifest, f, default_flow_style=False)

            print(f"✅ Generated: {filepath}")

        print("\n" + "="*80)
        print("DEPLOYMENT INSTRUCTIONS")
        print("="*80)
        print(f"""
# 1. Apply manifests
kubectl apply -f {output_dir}/namespace.yaml
kubectl apply -f {output_dir}/deployment.yaml
kubectl apply -f {output_dir}/service.yaml
kubectl apply -f {output_dir}/hpa.yaml

# 2. Wait for deployment
kubectl wait --for=condition=available --timeout=600s \\
    deployment/{self.model_name}-deployment -n {self.namespace}

# 3. Get service endpoint
kubectl get service {self.model_name}-service -n {self.namespace}

# 4. Test
ENDPOINT=$(kubectl get service {self.model_name}-service -n {self.namespace} \\
    -o jsonpath='{{.status.loadBalancer.ingress[0].ip}}')

curl http://$ENDPOINT/v1/completions \\
  -H "Content-Type: application/json" \\
  -d '{{
    "model": "{self.model_name}",
    "prompt": "Write a poem",
    "max_tokens": 256
  }}'

# 5. Monitor
kubectl get pods -n {self.namespace} -w
kubectl logs -f deployment/{self.model_name}-deployment -n {self.namespace}

# 6. Scale manually (if needed)
kubectl scale deployment/{self.model_name}-deployment -n {self.namespace} --replicas=5
        """)


def demo_gke_deployment():
    """Demo GKE (Google Kubernetes Engine) deployment"""
    print("\n" + "="*80)
    print("GKE DEPLOYMENT WITH GPU")
    print("="*80)

    print("""
STEP-BY-STEP GKE DEPLOYMENT:

# 1. Create GKE cluster with GPU node pool
gcloud container clusters create llm-cluster \\
    --zone=us-central1-a \\
    --machine-type=n1-standard-8 \\
    --num-nodes=1 \\
    --enable-autoscaling \\
    --min-nodes=1 \\
    --max-nodes=10

# 2. Add GPU node pool
gcloud container node-pools create gpu-pool \\
    --cluster=llm-cluster \\
    --zone=us-central1-a \\
    --machine-type=g2-standard-4 \\
    --accelerator=type=nvidia-l4,count=1 \\
    --num-nodes=1 \\
    --enable-autoscaling \\
    --min-nodes=1 \\
    --max-nodes=5 \\
    --spot  # Use spot instances for 70% savings

# 3. Get credentials
gcloud container clusters get-credentials llm-cluster --zone=us-central1-a

# 4. Install NVIDIA GPU Operator
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/gpu-operator/master/deployments/gpu-operator.yaml

# Wait for operator to be ready
kubectl wait --for=condition=ready pod -l app=nvidia-gpu-operator -n gpu-operator-resources

# 5. Deploy LLM (using manifests from above)
kubectl apply -f k8s/

# 6. Monitor GPU usage
kubectl get nodes -o custom-columns=NAME:.metadata.name,GPUs:.status.allocatable.nvidia\\.com/gpu

# 7. Check pod GPU allocation
kubectl describe pod -n llm-inference | grep nvidia.com/gpu


COST OPTIMIZATION WITH SPOT INSTANCES:

# Spot node pool (70% cheaper)
gcloud container node-pools create gpu-pool-spot \\
    --cluster=llm-cluster \\
    --zone=us-central1-a \\
    --machine-type=g2-standard-4 \\
    --accelerator=type=nvidia-l4,count=1 \\
    --spot \\
    --num-nodes=1 \\
    --enable-autoscaling \\
    --min-nodes=0 \\
    --max-nodes=10

# Add node affinity to prefer spot instances
# (Add to deployment.yaml)
affinity:
  nodeAffinity:
    preferredDuringSchedulingIgnoredDuringExecution:
    - weight: 100
      preference:
        matchExpressions:
        - key: cloud.google.com/gke-spot
          operator: In
          values:
          - "true"

# Tolerations for spot preemption
tolerations:
- key: cloud.google.com/gke-spot
  operator: Equal
  value: "true"
  effect: NoSchedule


CLUSTER AUTOSCALER:

# GKE automatically scales nodes based on pod requests
# Configure min/max nodes when creating node pool

# Monitor autoscaling
kubectl get events --sort-by='.lastTimestamp' | grep -i scale


MONITORING WITH PROMETHEUS + GRAFANA:

# 1. Install Prometheus
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install prometheus prometheus-community/kube-prometheus-stack -n monitoring --create-namespace

# 2. Access Grafana
kubectl port-forward -n monitoring svc/prometheus-grafana 3000:80

# Login: admin / prom-operator

# 3. Add dashboards for:
# - GPU utilization
# - Request rate
# - Latency (p50, p95, p99)
# - Error rate
    """)


def demo_eks_deployment():
    """Demo EKS (AWS Elastic Kubernetes Service) deployment"""
    print("\n" + "="*80)
    print("EKS DEPLOYMENT WITH GPU")
    print("="*80)

    print("""
STEP-BY-STEP EKS DEPLOYMENT:

# 1. Install eksctl
brew install eksctl  # macOS
# or download from https://eksctl.io

# 2. Create EKS cluster
eksctl create cluster \\
    --name llm-cluster \\
    --region us-east-1 \\
    --nodegroup-name cpu-nodes \\
    --node-type m5.xlarge \\
    --nodes 1 \\
    --nodes-min 1 \\
    --nodes-max 10

# 3. Add GPU node group
eksctl create nodegroup \\
    --cluster llm-cluster \\
    --region us-east-1 \\
    --name gpu-nodes \\
    --node-type g5.xlarge \\
    --nodes 1 \\
    --nodes-min 1 \\
    --nodes-max 5 \\
    --spot  # Use spot instances

# 4. Install NVIDIA device plugin
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/master/nvidia-device-plugin.yml

# 5. Deploy LLM
kubectl apply -f k8s/

# 6. Get load balancer endpoint
kubectl get svc llm-service -n llm-inference


SPOT INSTANCES WITH KARPENTER:

# Karpenter: Advanced auto-scaler for EKS
# Automatically provisions right-sized instances

# 1. Install Karpenter
helm repo add karpenter https://charts.karpenter.sh
helm install karpenter karpenter/karpenter \\
    --namespace karpenter \\
    --create-namespace

# 2. Create Provisioner
cat <<EOF | kubectl apply -f -
apiVersion: karpenter.sh/v1alpha5
kind: Provisioner
metadata:
  name: gpu-provisioner
spec:
  requirements:
    - key: karpenter.sh/capacity-type
      operator: In
      values: ["spot"]  # Prefer spot
    - key: node.kubernetes.io/instance-type
      operator: In
      values: ["g5.xlarge", "g5.2xlarge"]
  limits:
    resources:
      nvidia.com/gpu: "10"
  providerRef:
    name: default
EOF

# Karpenter will automatically:
# - Provision nodes when pods are unschedulable
# - Choose cheapest spot instance
# - Consolidate nodes when underutilized
# - Save 70-80% on compute costs!


COST SAVINGS WITH EKS:

1. Spot instances: 70% discount
2. Karpenter auto-scaling: Right-size nodes
3. Cluster Autoscaler: Scale to zero

Example monthly cost:
  • 1x g5.xlarge on-demand 24/7: $735
  • 1x g5.xlarge spot (Karpenter): $220
  • With auto-scaling (40% util): $88
  • Total savings: 88%!
    """)


def demo_helm_chart():
    """Demo Helm chart for simplified deployment"""
    print("\n" + "="*80)
    print("HELM CHART FOR LLM DEPLOYMENT")
    print("="*80)

    print("""
HELM = Package manager for Kubernetes

Create Helm chart for reusable deployments:

# Directory structure:
llm-inference/
  Chart.yaml
  values.yaml
  templates/
    deployment.yaml
    service.yaml
    hpa.yaml

# Chart.yaml
apiVersion: v2
name: llm-inference
description: LLM Inference with vLLM
version: 1.0.0

# values.yaml
modelName: meta-llama/Llama-2-7b-hf
replicas: 2
gpu:
  count: 1
  type: nvidia.com/gpu
image:
  repository: vllm/vllm-openai
  tag: latest
resources:
  limits:
    memory: 24Gi
    cpu: 8
  requests:
    memory: 16Gi
    cpu: 4
autoscaling:
  enabled: true
  minReplicas: 1
  maxReplicas: 10
  targetCPUUtilization: 70
service:
  type: LoadBalancer
  port: 80

# templates/deployment.yaml
# (Use template variables like {{ .Values.modelName }})

# Deploy with Helm
helm install llama-7b ./llm-inference

# Override values
helm install llama-13b ./llm-inference \\
    --set modelName=meta-llama/Llama-2-13b-hf \\
    --set gpu.count=1 \\
    --set replicas=3

# Upgrade deployment
helm upgrade llama-7b ./llm-inference \\
    --set replicas=5

# Rollback
helm rollback llama-7b

# List releases
helm list
    """)


if __name__ == "__main__":
    print("="*80)
    print("KUBERNETES LLM DEPLOYMENT")
    print("="*80)

    # Generate manifests
    deployer = KubernetesLLMDeployment(
        model_name="llama-2-7b",
        replicas=2,
        gpu_count=1
    )

    deployer.generate_manifests()

    # Demo different platforms
    demo_gke_deployment()
    demo_eks_deployment()
    demo_helm_chart()

    print("\n" + "="*80)
    print("KUBERNETES BEST PRACTICES")
    print("="*80)
    print("""
1. Resource Limits:
   ✅ Always set resource limits/requests
   ✅ Prevents OOM kills
   ✅ Enables efficient scheduling

2. Health Checks:
   ✅ Liveness probe: Restart if unhealthy
   ✅ Readiness probe: Route traffic only when ready
   ✅ Startup probe: Allow slow starts

3. Auto-scaling:
   ✅ HPA for horizontal scaling
   ✅ Cluster Autoscaler for node scaling
   ✅ Use custom metrics (request rate, queue length)

4. Cost Optimization:
   ✅ Use spot instances (70% cheaper)
   ✅ Karpenter for intelligent provisioning
   ✅ Scale to zero during idle

5. Monitoring:
   ✅ Prometheus + Grafana
   ✅ Track: GPU util, latency, throughput
   ✅ Alert on degradation

6. Security:
   ✅ Network policies (restrict traffic)
   ✅ RBAC (least privilege)
   ✅ Pod security policies
   ✅ Secrets management (not in manifests!)

7. CI/CD:
   ✅ GitOps with ArgoCD/Flux
   ✅ Automated testing
   ✅ Canary deployments
   ✅ Rollback strategy

    """)

    print("\n✅ CHAPITRE 14 TERMINÉ!")
    print("="*80)
    print("""
Vous maîtrisez maintenant:

1. Cloud Providers:
   ✅ AWS (SageMaker, EC2, EKS)
   ✅ GCP (Vertex AI, Compute Engine, GKE)
   ✅ Azure (Azure ML, VMs, AKS)
   ✅ Cost comparison et optimization

2. Deployment Patterns:
   ✅ Managed ML platforms
   ✅ Raw VMs + Load Balancers
   ✅ Kubernetes (production-grade)
   ✅ Serverless (for small models)

3. Kubernetes:
   ✅ Deployments avec GPU
   ✅ Auto-scaling (HPA, Cluster Autoscaler)
   ✅ Load balancing
   ✅ Health checks
   ✅ Helm charts

4. Cost Optimization:
   ✅ Spot instances (70-90% savings)
   ✅ Auto-scaling (40-60% savings)
   ✅ Quantization (4x cheaper instances)
   ✅ Right-sizing avec Karpenter
   ✅ Combined: 80-90% cost reduction!

5. Production Ready:
   ✅ High availability
   ✅ Auto-healing
   ✅ Zero-downtime deployments
   ✅ Monitoring et alerting
   ✅ Security best practices

Real-world example:
  • Llama 2 7B production deployment
  • 3 replicas, auto-scale 1-10
  • GKE with L4 GPUs (spot)
  • Cost: ~$200/month (vs $2000 on-demand!)
  • Throughput: 10k requests/hour
  • Latency: < 100ms p95

Ready for production cloud deployment!

Next: Chapter 15 → APIs and Services (FastAPI, auth, rate limiting)
    """)
```
