# Kubernetes deployment

| File | Resource | Purpose |
|---|---|---|
| `namespace.yaml` | Namespace `travel-ml` | isolates the app |
| `configmap.yaml` | ConfigMap | `LOG_LEVEL`, `PROJECT_ROOT`, `API_URL` for Streamlit |
| `deployment.yaml` | Deployment `flight-price-api` | 2 replicas, CPU/memory requests and limits, liveness `/health`, readiness `/ready`, rolling update with `maxUnavailable: 0`, Prometheus annotations |
| `service.yaml` | Service (NodePort 30500) | stable endpoint load-balancing across API pods |
| `hpa.yaml` | HorizontalPodAutoscaler | scales the API from 2 to 6 pods at 60% CPU, with a 120 s scale-down window |
| `streamlit.yaml` | Deployment + Service (NodePort 30851) | the web app |

## Why Kubernetes
* **Scalability:** the HPA adds pods when traffic spikes (try `scripts/simulate_traffic.py --n 5000 --workers 32`) and removes them afterwards.
* **Self-healing:** a pod that fails its liveness probe is restarted automatically.
* **Zero-downtime model updates:** a new image rolls out pod by pod, and the readiness probe stops traffic going to a pod until its model is loaded.
* **Rollback:** `revisionHistoryLimit: 5` keeps old ReplicaSets, so `kubectl rollout undo` brings back the previous model in seconds.

## Deploy on minikube
```bash
minikube start && minikube addons enable metrics-server
make docker-build
make k8s-deploy                     # loads images into minikube + kubectl apply -f k8s/
kubectl -n travel-ml get pods,svc,hpa
minikube service flight-price-api -n travel-ml --url
```

Useful commands:
```bash
kubectl -n travel-ml rollout history deployment/flight-price-api
kubectl -n travel-ml rollout undo deployment/flight-price-api
kubectl -n travel-ml scale deployment/flight-price-api --replicas=4
kubectl -n travel-ml get hpa -w
kubectl -n travel-ml logs -l app=flight-price-api --tail=50
```
