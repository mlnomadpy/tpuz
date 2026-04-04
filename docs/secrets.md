---
layout: page
title: Secrets & Security
permalink: /secrets/
---

# Secrets & Security Guide

## The Problem

Training scripts need API keys (WandB, HuggingFace, GitHub). The naive approach leaks them:

```bash
# BAD: visible in ps aux, shell history, gcloud logs
tpuz run my-tpu "WANDB_API_KEY=abc123 python train.py"
```

## Three Approaches (worst → best)

### 1. Environment Dict (fallback)

```python
tpu.run("python train.py", env={"WANDB_API_KEY": "abc123"})
```

**How it works:** Writes a `.env` file to the VM via SCP, sources it before training. Never appears in command line or `ps aux`.

**Risk:** Secret exists on your local machine and transits via SSH to the VM. The `.env` file sits on the VM's disk.

**Use when:** Quick experiments, single-user VMs, you trust the network.

### 2. Google Cloud Secret Manager (recommended)

```python
from tpuz import SecretManager

# One-time: store secrets in GCP
sm = SecretManager(project="my-project")
sm.create("WANDB_API_KEY", "your-key")
sm.create("HF_TOKEN", "hf_...")

# One-time: grant TPU VM access
sm.grant_tpu_access("WANDB_API_KEY")
sm.grant_tpu_access("HF_TOKEN")
# Or grant all at once:
sm.grant_tpu_access_all()

# Training: secrets loaded server-side
tpu.run("python train.py",
    secrets=["WANDB_API_KEY", "HF_TOKEN"])
```

**How it works:** The TPU VM reads secrets directly from Cloud Secret Manager using its service account. Secrets never leave GCP's network, never exist on your local machine, never appear in any log.

**Risk:** Minimal. Requires the VM's service account to have `secretmanager.secretAccessor` role.

**Use when:** Production training, shared VMs, any time secrets matter.

### 3. Comparison

| | env dict | Cloud Secrets |
|---|:-:|:-:|
| Secret on local machine | Yes | **No** |
| Secret in transit | SCP (encrypted) | **Never** |
| Secret on VM disk | .env file | **In memory only** |
| Visible in `ps aux` | No | **No** |
| Visible in shell history | No | **No** |
| Visible in error logs | Redacted | **Never present** |
| Survives VM recreation | Must re-send | **Automatic** (IAM) |
| Setup effort | Zero | One-time IAM setup |

## Setup Guide

### Enable Secret Manager API

```bash
gcloud services enable secretmanager.googleapis.com --project=my-project
```

### Store Secrets

```bash
# From CLI
echo -n "your-wandb-key" | gcloud secrets create WANDB_API_KEY --data-file=-
echo -n "hf_your_token" | gcloud secrets create HF_TOKEN --data-file=-

# Or from Python
from tpuz import SecretManager
sm = SecretManager()
sm.create("WANDB_API_KEY", "your-wandb-key")
sm.create("HF_TOKEN", "hf_your_token")
```

### Grant TPU VM Access

The TPU VM uses the default Compute Engine service account. Grant it access:

```bash
# Find the service account
SA=$(gcloud iam service-accounts list --format='value(email)' \
    --filter='displayName:Compute Engine default')

# Grant access to specific secrets
gcloud secrets add-iam-policy-binding WANDB_API_KEY \
    --member="serviceAccount:$SA" \
    --role="roles/secretmanager.secretAccessor"

gcloud secrets add-iam-policy-binding HF_TOKEN \
    --member="serviceAccount:$SA" \
    --role="roles/secretmanager.secretAccessor"
```

Or from Python:

```python
sm = SecretManager()
sm.grant_tpu_access_all()  # Grants access to ALL secrets
```

### Use in Training

```python
from tpuz import TPU

tpu = TPU("my-tpu", accelerator="v4-8")
tpu.up()
tpu.setup()

# Secrets loaded from Cloud Secret Manager on the VM
tpu.run("python train.py",
    secrets=["WANDB_API_KEY", "HF_TOKEN"],
    sync="./src")
```

### Manage Secrets

```python
sm = SecretManager()
sm.list()                      # ["WANDB_API_KEY", "HF_TOKEN"]
sm.get("WANDB_API_KEY")       # "your-key"
sm.create("NEW_KEY", "value") # Create or update
sm.delete("OLD_KEY")          # Delete
sm.exists("WANDB_API_KEY")    # True
```

## Best Practices

1. **Never commit secrets** to git, `.env` files, or configs
2. **Use Cloud Secrets** for any production training run
3. **Use `env={}` only** for quick local experiments
4. **Rotate secrets** regularly via `sm.create()` (adds new version)
5. **Audit access** via `gcloud secrets get-iam-policy SECRET_NAME`
6. **tpuz auto-redacts** secrets from error messages (WANDB_API_KEY, HF_TOKEN, etc.)

## Training Script Side

Your training script reads secrets from environment variables as usual:

```python
import os
wandb_key = os.environ.get("WANDB_API_KEY")
hf_token = os.environ.get("HF_TOKEN")
```

No code changes needed — tpuz loads the secrets as env vars before your script runs.
