# StreamGuard Fraud Detection - Kubernetes Deployment

This guide explains how to deploy the StreamGuard Fraud Detection application on Kubernetes.

## Prerequisites

- Kubernetes cluster (local or cloud)
- kubectl configured to access your cluster
- Docker image built and available (`streamguard-ml:latest`)

## Quick Start

### 1. Build and Push Docker Image

```bash
# Build the image
cd docker/
docker build -t streamguard-ml:latest .

# If using a registry, tag and push
docker tag streamguard-ml:latest your-registry/streamguard-ml:latest
docker push your-registry/streamguard-ml:latest
```

### 2. Deploy to Kubernetes

```bash
# Apply all manifests
kubectl apply -f k8s/

# Or apply individually
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

### 3. Access the Application

Choose one of the service types:

**ClusterIP (Internal access only):**
```bash
kubectl port-forward service/streamguard-ml-service 8080:80
# Access at http://localhost:8080
```

**NodePort (External access via node IP):**
```bash
# Access at http://<NODE_IP>:30080
kubectl get nodes -o wide
```

**LoadBalancer (Cloud environments):**
```bash
# Get external IP
kubectl get service streamguard-ml-loadbalancer
# Access at http://<EXTERNAL_IP>
```

## Training Models

### Option 1: Using Kubernetes Job

```bash
# Create training data PVC and upload data
kubectl apply -f k8s/configmap.yaml

# Copy training data to PVC (example using a helper pod)
kubectl run data-loader --image=busybox --rm -it --restart=Never \
  --overrides='{"spec":{"containers":[{"name":"data-loader","image":"busybox","volumeMounts":[{"name":"training-data","mountPath":"/data"}]}],"volumes":[{"name":"training-data","persistentVolumeClaim":{"claimName":"streamguard-training-data-pvc"}}]}}' \
  -- sh

# Inside the pod, copy your data file
# Then run the training job
kubectl apply -f k8s/training-job.yaml

# Monitor training progress
kubectl logs -f job/streamguard-training-job
```

### Option 2: Using ConfigMap (for smaller datasets)

```bash
# Create ConfigMap from training data
kubectl create configmap streamguard-training-data --from-file=train_transaction.csv

# Deploy the application (it will use the ConfigMap)
kubectl apply -f k8s/deployment.yaml
```

## Kubernetes Resources

### Deployments
- `streamguard-ml-deployment` - Main application deployment with 2 replicas

### Services
- `streamguard-ml-service` - ClusterIP service (internal access)
- `streamguard-ml-nodeport` - NodePort service (external access via node)
- `streamguard-ml-loadbalancer` - LoadBalancer service (cloud environments)

### Storage
- `streamguard-models-pvc` - Persistent storage for trained models (5Gi)
- `streamguard-data-pvc` - Persistent storage for application data (2Gi)
- `streamguard-logs-pvc` - Persistent storage for logs (1Gi)
- `streamguard-training-data-pvc` - Persistent storage for training data (10Gi)

### Jobs
- `streamguard-training-job` - One-time job for training ML models

## Configuration

### Environment Variables
Set via ConfigMap (`streamguard-config`):
- `FLASK_ENV=production`
- `PYTHONUNBUFFERED=1`
- `FLASK_APP=app.py`

### Resource Limits
- **Requests**: 1Gi memory, 500m CPU
- **Limits**: 2Gi memory, 1000m CPU
- **Training Job**: 2-4Gi memory, 1-2 CPU cores

## Monitoring and Health Checks

### Health Probes
- **Liveness Probe**: HTTP GET `/` every 30s
- **Readiness Probe**: HTTP GET `/` every 10s

### Monitoring Commands

```bash
# Check pod status
kubectl get pods -l app=streamguard-ml

# View logs
kubectl logs -f deployment/streamguard-ml-deployment

# Check service endpoints
kubectl get endpoints

# Monitor resource usage
kubectl top pods -l app=streamguard-ml
```

## Scaling

### Horizontal Scaling
```bash
# Scale to 5 replicas
kubectl scale deployment streamguard-ml-deployment --replicas=5

# Auto-scaling (requires metrics server)
kubectl autoscale deployment streamguard-ml-deployment --cpu-percent=70 --min=2 --max=10
```

### Vertical Scaling
Update resource requests/limits in `deployment.yaml` and apply:
```bash
kubectl apply -f k8s/deployment.yaml
```

## Troubleshooting

### Common Issues

**Pods not starting:**
```bash
kubectl describe pod <pod-name>
kubectl logs <pod-name>
```

**Storage issues:**
```bash
kubectl get pvc
kubectl describe pvc <pvc-name>
```

**Service not accessible:**
```bash
kubectl get svc
kubectl describe svc <service-name>
```

**Training job failed:**
```bash
kubectl describe job streamguard-training-job
kubectl logs job/streamguard-training-job
```

### Debug Commands

```bash
# Get all resources
kubectl get all -l app=streamguard-ml

# Exec into running pod
kubectl exec -it <pod-name> -- /bin/bash

# Check events
kubectl get events --sort-by=.metadata.creationTimestamp
```

## Cleanup

```bash
# Delete all resources
kubectl delete -f k8s/

# Or delete individually
kubectl delete deployment streamguard-ml-deployment
kubectl delete service streamguard-ml-service streamguard-ml-nodeport streamguard-ml-loadbalancer
kubectl delete pvc streamguard-models-pvc streamguard-data-pvc streamguard-logs-pvc
kubectl delete job streamguard-training-job
kubectl delete configmap streamguard-config streamguard-training-data
```

## Production Considerations

### Security
- Use secrets for sensitive data
- Enable RBAC
- Use network policies
- Scan images for vulnerabilities

### High Availability
- Deploy across multiple nodes/zones
- Use pod anti-affinity rules
- Configure resource quotas
- Set up monitoring and alerting

### Performance
- Use appropriate storage classes
- Configure resource requests/limits
- Enable horizontal pod autoscaling
- Monitor and optimize based on metrics
