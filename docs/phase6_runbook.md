# Phase 6 staging deployment runbook

This is a low-cost local Kubernetes deployment path. It uses a ClusterIP
service and `kubectl port-forward`; it does not create a cloud cluster or expose
the API publicly. Inference runs on CPU inside the image.

## What the deployment creates

- Namespace `vehicle-classifier-staging` isolates staging resources.
- One-replica Deployment uses a GHCR image by immutable SHA-256 digest.
- ClusterIP Service provides stable in-cluster access on port 8000.
- ConfigMap supplies non-secret release/model identity and API upload limits.
- Startup, readiness, and liveness probes use the existing API health routes.
- CPU/memory requests and limits, a non-root user, dropped Linux capabilities,
  read-only root filesystem, and a bounded temporary volume limit runtime use.

The serialized checkpoint remains packaged in the image. No model download,
Secret, PVC, Ingress, cloud account, or external object store is needed to serve.

## Local prerequisite path (kind or Minikube)

Install Docker, `kubectl`, and either kind or Minikube. Docker must be running
and your account must be allowed to access its socket. On a Linux shell, a kind
cluster can be created with:

```bash
kind create cluster --name vehicle-classifier
kubectl cluster-info --context kind-vehicle-classifier
```

For Minikube, use `minikube start` and confirm `kubectl cluster-info` works.
The Kubernetes context is local; do not point this demo at a paid/production
cluster. If the GHCR package is private, configure an image pull Secret named
`ghcr-pull-secret` and add it under `spec.template.spec.imagePullSecrets` in
`k8s/deployment.yaml.tpl`. Keep its credentials outside Git.

## Configure Jenkins once

Install Jenkins Pipeline support and provide a Linux agent with label
`kubernetes-deployer`. That agent needs `kubectl`, `envsubst` (from gettext),
Python 3, and network access to the Kubernetes API and GHCR. Add this repository
as a Pipeline job using `Jenkinsfile`.

In **Manage Jenkins → Credentials**, add the cluster kubeconfig as a **Secret
file** credential with ID `vehicle-classifier-kubeconfig`. For a local demo,
the kubeconfig must be usable from the Jenkins agent itself; a kubeconfig that
points at a laptop-only address will not work from a remote Jenkins agent.
Restrict the identity to the staging namespace and the namespace-scoped
Deployment, Service, ConfigMap, and read permissions it needs. Avoid using a
cluster-admin kubeconfig for a shared Jenkins server.

## Deploy an approved release digest

The GHCR publishing workflow attaches `release_metadata.json` to the GitHub
Release. Download it and paste the complete contents into Jenkins **Build with
Parameters → RELEASE_METADATA_JSON**. The pipeline parses the image and model
identity together and rejects unsupported schemas, non-champion versions,
mismatched image digest/reference, invalid hashes, commits, tags, architecture,
or artifact size.

The resulting immutable reference looks like
`ghcr.io/owner/repository@sha256:<64 lowercase hex digits>`. Do not substitute
the tag or `latest`. Jenkins checks the input formats, confirms the champion
alias, applies the staging objects, waits for rollout, forwards the service to
localhost, and checks both readiness and a valid prediction response. The
prediction smoke also compares the model checksum in readiness and prediction.
Only a successful run is marked successful.

On rollout or smoke failure, Jenkins stops the run and restores the prior
Deployment revision, waits for it to become available, and smoke-checks that
restored service. If the first staging deployment fails, Jenkins removes that
unverified Deployment. If automated recovery itself fails, the Jenkins build
reports that manual operator recovery is needed; inspect the Deployment and
pod events with `kubectl describe` before retrying.

## Observe, rollback, clean up

```bash
kubectl get all -n vehicle-classifier-staging
kubectl rollout history deployment/vehicle-classifier-api -n vehicle-classifier-staging
kubectl describe deployment/vehicle-classifier-api -n vehicle-classifier-staging
kubectl logs deployment/vehicle-classifier-api -n vehicle-classifier-staging
```

To manually restore the immediately previous Deployment revision:

```bash
kubectl rollout undo deployment/vehicle-classifier-api -n vehicle-classifier-staging
kubectl rollout status deployment/vehicle-classifier-api -n vehicle-classifier-staging --timeout=4m
```

Remove the local demo when finished:

```bash
kubectl delete namespace vehicle-classifier-staging
kind delete cluster --name vehicle-classifier
```

Deleting the namespace destroys only these staging Kubernetes objects. It does
not remove the source model, GHCR image, MLflow tracking data, or DVC objects.

## Gate evidence and current limitations

A disposable kind cluster was used to deploy the existing local Phase 4 image;
readiness and a real prediction passed. A follow-up Deployment with a missing
image failed to roll out, revision 1 was restored, and the recovered service
passed both checks again. This demonstrates Kubernetes rollout and revision
recovery locally.

The Jenkins pipeline itself has not run, and the local demonstration used a
tagged local Docker image rather than a GHCR digest. To close the full exit
gate, run Jenkins once with a published GHCR digest and capture: the Jenkins
successful deployment log, the digest on the live Deployment,
readiness/prediction smoke output, a deliberately failing release, Jenkins
rollback output, and the restored Deployment revision. Phase 5's first GHCR
release is still pending, so such a digest may not yet exist.
