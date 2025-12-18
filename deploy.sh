#!/bin/bash

# Stop the script if any command fails
set -e

echo "Building Docker image... (docker build -t libra-local:v1 .)"
docker build -t libra-local:v1 .
echo "Loading image into cluster..."

minikube start

# Option A: If using Minikube
minikube image load libra-local:v1

# Option B: Docker Desktop (Kubernetes enabled)
# No extra command needed!

echo "Applying Kubernetes configuration... (kubectl apply -f deployment.yaml -f service.yaml)"
kubectl apply -f deployment.yaml -f service.yaml

echo "Waiting for successful deployment (kubectl rollout status deployment/libra-deployment)"
kubectl rollout status deployment/libra-deployment

echo "Deployment complete!"

# Prints the service URL if using Minikube
minikube service libra-service --url