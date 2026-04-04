---
layout: home
title: tpuz
---

# tpuz

Manage GCP TPU & GPU VMs from your terminal. Create, train, debug, recover, teardown — one command.

```bash
pip install tpuz
```

```python
from tpuz import TPU

tpu = TPU("my-tpu", accelerator="v4-8")
tpu.up()
tpu.setup()
tpu.run("python train.py", secrets=["WANDB_API_KEY"], sync="./src")
tpu.logs()
tpu.cost_summary()
tpu.down()
```

---


# Getting Started — From Zero to Training

This guide takes you from nothing to running a training job on a GCP TPU or GPU, step by step.

## Prerequisites

You need three things:

### 1. Install gcloud CLI

```bash
# macOS
brew install google-cloud-sdk

# Linux
curl https://sdk.cloud.google.com | bash

# Verify
gcloud --version
```

### 2. Authenticate

```bash
gcloud auth login
# Opens browser → sign in with your Google account
```

### 3. Set your project

```bash
# If you have one
gcloud config set project my-project

# If you don't, create one at https://console.cloud.google.com
# Then enable Compute Engine API:
gcloud services enable compute.googleapis.com
```

## Install tpuz

```bash
pip install tpuz
```

Zero Python dependencies — it just calls `gcloud`.

## Step 1: Check Your Setup

```python
from tpuz import TPU

tpu = TPU("test", accelerator="v4-8")
tpu.preflight()
# gcloud account: you@gmail.com
# gcloud project: my-project
```

Or from CLI:

```bash
tpuz preflight
```

If this fails, go back to Prerequisites.

## Step 2: Check What's Available

```python
# What TPU types exist?
TPU.list_runtimes(zone="us-central2-b")

# Is v4-8 available right now?
TPU.availability("v4-8", zone="us-central2-b")
# {"available": True, "spot_rate": 2.06}
```

### TPU Types for Beginners

| Type | What it is | Cost (spot) | Best for |
|------|-----------|-------------|----------|
| `v4-8` | 4 TPU chips, 1 VM | $2.06/hr | Starting out, small models |
| `v5litepod-8` | 8 TPU chips, 1 VM | $9.60/hr | Medium models |
| `v4-32` | 16 chips, 4 VMs | $8.24/hr | Large models, multi-host |

If you have TRC access ([apply here](https://sites.research.google/trc/about/)), v5e and v6e are free.

## Step 3: Create a TPU VM

```python
tpu = TPU("my-first-tpu", accelerator="v4-8", zone="us-central2-b")
tpu.up()
# Creating TPU 'my-first-tpu' (v4-8) in us-central2-b...
# TPU 'my-first-tpu' ready! IPs: ['34.x.x.x']
```

This takes 1-3 minutes. The VM is now running and billing.

## Step 4: Install Dependencies

```python
tpu.setup()
# Installing deps...
#   sudo apt-get update...
#   pip install jax[tpu]...
#   pip install flax optax...
# Setup done!
```

To install your own packages:

```python
tpu.setup(extra_pip="flaxchat transformers")
```

### Verify it works

```python
tpu.verify()
#   worker 0: 4 devices
# All workers verified!
```

## Step 5: Upload Your Code & Run

Assume you have a training script `train.py` in `./src/`:

```python
tpu.run("python train.py",
    sync="./src")          # Uploads ./src to the VM
# Uploading ./src → /home/user/workdir...
# Launched: python train.py
```

Training runs in the background — even if you close your laptop.

## Step 6: Watch the Logs

```python
tpu.logs()
# step 0 | loss 9.01 | dt 17.9s
# step 100 | loss 4.82 | dt 0.55s
# step 200 | loss 3.71 | dt 0.56s
# ^C (Ctrl-C to detach — training continues)
```

### Other monitoring options

```python
tpu.is_running()       # Quick check: True/False
tpu.health_pretty()    # Dashboard with worker status
tpu.cost_summary()     # How much you've spent so far
```

## Step 7: Download Results

```python
tpu.collect(["model.pkl", "results.json"],
    local_dir="./outputs")
# Downloaded: ./outputs/model.pkl
# Downloaded: ./outputs/results.json
```

## Step 8: Clean Up

```python
tpu.down()
# Deleting TPU 'my-first-tpu'...
# Deleted.
```

**Important:** Always delete when done — TPU VMs bill continuously while running.

## Step 9: Check What It Cost

```python
tpu.cost_summary()
# $4.12 (2.0h × $2.06/hr v4-8 spot)
```

---

## Next Steps

You've completed the basics! Here's where to go next:

### Save Secrets Properly

Don't pass API keys via `env={}`. Use Cloud Secret Manager:

```python
from tpuz import SecretManager
sm = SecretManager()
sm.create("WANDB_API_KEY", "your-key")
sm.grant_tpu_access_all()

tpu.run("python train.py", secrets=["WANDB_API_KEY"])
```

See [Secrets & Security](secrets.md) for the full guide.

### Add Checkpoint Persistence

Preemption happens. Save checkpoints to GCS:

```python
from tpuz import GCS
gcs = GCS("gs://my-bucket")

tpu.run_with_resume("python train.py", gcs=gcs, run_name="run-01")
# Auto-resumes from latest checkpoint
```

### Handle Preemption

```python
tpu.watch("python train.py", max_retries=5)
# Automatically recreates VM and restarts on preemption
```

### Use GPUs Instead

```python
from tpuz import GCE
vm = GCE.gpu("my-vm", gpu="a100")
vm.up()
# Same API: setup(), run(), logs(), down()
```

See [GPU VMs](gpu.md) for the full guide.

### Debug Interactively

```python
tpu.repl()                             # Python REPL on the VM
tpu.debug("python train.py")           # VS Code debugger
tpu.tunnel(6006)                       # TensorBoard
```

### All-in-One Training

```python
tpu.run_once("python train.py",
    sync="./src",
    collect_files=["model.pkl"],
    gcs=gcs,
    notify_url="https://hooks.slack.com/...")
# up → setup → resume → run → wait → collect → notify → down
```

See [Best Practices](best-practices.md) for production workflows.

---

## Troubleshooting

### "No running kernels" / connection error

Your gcloud auth may have expired:
```bash
gcloud auth login
```

### "Quota exceeded"

You've hit your TPU quota. Options:
- Try a different zone: `tpu = TPU("x", zone="europe-west4-a")`
- Use Queued Resources: `tpu.up_queued()` (waits for capacity)
- Apply for more quota: [console.cloud.google.com/iam-admin/quotas](https://console.cloud.google.com/iam-admin/quotas)

### "VM already exists"

`up()` is idempotent. If the VM exists, it skips. To recreate:
```python
tpu.down()
tpu.up()
```

### Training process died

```python
tpu.health_pretty()   # Check worker status
tpu.logs()            # Check error in logs
tpu.run("python train.py")  # Restart
```

### Preempted

Spot VMs can be reclaimed. Use `watch()` for auto-recovery:
```python
tpu.watch("python train.py")
```

---


# Complete Usage Guide

## Installation

```bash
pip install tpuz
```

Requirements:
- Python 3.10+
- `gcloud` CLI installed: [cloud.google.com/sdk/docs/install](https://cloud.google.com/sdk/docs/install)
- Authenticated: `gcloud auth login`
- Project set: `gcloud config set project YOUR_PROJECT`

## Core Concepts

### TPU Class

The `TPU` class manages a single TPU VM. It wraps `gcloud` CLI commands:

```python
from tpuz import TPU

tpu = TPU(
    name="my-tpu",              # VM name (must be unique in zone)
    accelerator="v4-8",         # TPU type
    zone="us-central2-b",       # GCP zone
    project="my-project",       # GCP project (or GCLOUD_PROJECT env)
    preemptible=True,           # Use spot pricing (3x cheaper)
)
```

### Lifecycle

```
preflight() → up() → setup() → run() → logs() → wait() → collect() → down()
```

Each step is independent — you can skip any or run them in any order.

## Step-by-Step

### 1. Preflight Check

```python
tpu.preflight()
# gcloud account: user@gmail.com
# gcloud project: my-project
```

Verifies your `gcloud` is configured. Catches common issues before they waste time.

### 2. Create the VM

```python
# Standard creation
tpu.up()

# Queued Resources (waits for capacity — better for spot)
tpu.up_queued(timeout_hours=2)

# Multi-zone failover (tries each zone)
tpu = TPU.create_multi_zone("my-tpu", "v4-8",
    zones=["us-central2-b", "us-central1-a", "europe-west4-a"])
```

`up()` is idempotent — if the VM already exists, it skips creation.

### 3. Install Dependencies

```python
tpu.setup()                          # JAX[TPU] + common deps
tpu.setup(extra_pip="flaxchat")      # + custom packages
```

Runs on all workers in parallel. Detects Python version, installs via apt or pixi.

### 4. Verify

```python
tpu.verify()
#   worker 0: 4 devices
#   worker 1: 4 devices
# All workers verified!
```

### 5. Upload Code & Run

```python
# Upload local directory + launch training
tpu.run("python train.py --steps=50000",
    sync="./src",                          # Upload ./src to VM
    secrets=["WANDB_API_KEY", "HF_TOKEN"], # From Cloud Secret Manager
)
```

Training runs in a detached `nohup` session — it continues even if you disconnect.

### 6. Monitor

```python
tpu.logs()                  # Stream logs (Ctrl-C to detach)
tpu.logs_all(lines=20)      # All workers color-coded
tpu.health_pretty()         # Worker status dashboard
tpu.is_running()            # Quick check
tpu.cost_summary()          # Running cost
```

### 7. Wait for Completion

```python
success = tpu.wait(timeout_hours=24)
# Polls log every 60s for "COMPLETE" or "FAILED"
```

### 8. Collect Results

```python
tpu.collect(["model.pkl", "results.json", "report.md"],
    local_dir="./outputs")
```

### 9. Teardown

```python
tpu.down()
```

## GCS Checkpoints

Persist checkpoints to Google Cloud Storage so preemption doesn't lose progress:

```python
from tpuz import GCS

gcs = GCS("gs://my-bucket")

# Upload a checkpoint
gcs.upload_checkpoint("./checkpoints", "run-01", step=1000)
# Stored as: gs://my-bucket/checkpoints/run-01/step-001000/

# Find latest
step = gcs.latest_step("run-01")  # 5000

# Download
gcs.download_checkpoint("run-01", step=5000, local_dir="./ckpt")

# Auto-resume training from latest checkpoint
tpu.run_with_resume("python train.py", gcs=gcs, run_name="run-01")
# Finds step 5000 → appends --resume-from-step=5000

# List all runs
gcs.list_runs()  # ["run-01", "run-02", "run-03"]
```

## Secrets

See [Secrets & Security](secrets.md) for the full guide.

**Quick version:**

```python
from tpuz import SecretManager

# One-time: store secrets in GCP
sm = SecretManager()
sm.create("WANDB_API_KEY", "your-key")
sm.grant_tpu_access_all()

# Training: secrets loaded server-side
tpu.run("python train.py", secrets=["WANDB_API_KEY"])
```

## Preemption Recovery

Spot TPUs can be preempted at any time. tpuz handles this automatically:

```python
# Basic recovery
tpu.watch("python train.py", max_retries=5)

# With Slack notifications
tpu.watch_notify("python train.py",
    notify_url="https://hooks.slack.com/services/...",
    max_retries=5)
```

Flow: Poll every 60s → PREEMPTED → delete → recreate → setup → restart from checkpoint.

## Debugging

### Interactive REPL

```python
tpu.repl()  # Opens Python on worker 0
# Multi-host: other workers wait at barrier
```

### VS Code Debugger

```python
tpu.debug("python train.py", port=5678)
# Prints VS Code launch.json config to attach
```

### SSH Tunnel

```python
tpu.tunnel(6006)         # TensorBoard
tpu.tunnel(8888, 9999)   # Jupyter on local:9999
```

### Health Dashboard

```python
tpu.health_pretty()
#   Worker     Status          Last Log
#   -------------------------------------------
#   worker 0   running         step 1234 | loss 2.31
#   worker 1   running         step 1234 | loss 2.31
#   worker 2   stopped         (no log)
```

## Scaling

Change accelerator type without changing code:

```python
tpu.scale("v4-32")  # Deletes v4-8, creates v4-32, re-runs setup
# JAX automatically handles the larger mesh
```

## Cost Tracking

```python
tpu.cost_summary()
# $12.36 (6.0h × $2.06/hr v4-8 spot)

# Check pricing before creating
TPU.availability("v4-8", zone="us-central2-b")
# {"available": True, "spot_rate": 2.06, "on_demand_rate": 6.18}
```

### Budget Limits

```python
tpu.schedule("python train.py",
    start_after="22:00",  # Cheaper spot at night
    max_cost=50.0)        # Auto-kill at $50
```

## Run-Once (Docker-like)

Complete lifecycle in one call:

```python
tpu.run_once("python train.py",
    sync="./src",
    collect_files=["model.pkl"],
    gcs=gcs,
    notify_url="https://hooks.slack.com/...",
)
# up → setup → resume → run → wait → collect → notify → down
```

## Environment Snapshot

Save/restore pip packages across preemptions:

```python
tpu.snapshot_env(gcs=gcs)  # pip freeze → GCS
# After preemption + recreation:
tpu.restore_env(gcs=gcs)   # pip install from frozen list
```

## CLI Reference

```bash
# Lifecycle
tpuz up NAME -a v4-8            # Create VM
tpuz down NAME                  # Delete VM
tpuz status NAME                # Show state + IPs
tpuz list                       # List all VMs
tpuz preflight                  # Check gcloud config
tpuz avail v4-8                 # Check availability + price
tpuz runtimes                   # List runtime versions

# Training
tpuz setup NAME --pip=PKG       # Install deps
tpuz verify NAME                # Check JAX on all workers
tpuz run NAME "cmd" --sync=DIR  # Upload + launch
tpuz logs NAME                  # Stream logs
tpuz logs-all NAME              # All workers
tpuz kill NAME                  # Stop training
tpuz wait NAME                  # Wait for completion
tpuz collect NAME file1 file2   # Download artifacts

# Debugging
tpuz repl NAME                  # Interactive Python
tpuz debug NAME "cmd"           # debugpy attach
tpuz health NAME                # Worker dashboard
tpuz tunnel NAME PORT           # SSH tunnel
tpuz scale NAME ACCELERATOR     # Scale up/down
tpuz cost NAME                  # Show cost

# Recovery
tpuz watch NAME "cmd"           # Auto-recover preemption

# All-in-one
tpuz train NAME "cmd" -a v4-8 --sync=. --recover --teardown
tpuz run-once NAME "cmd" --sync=. --collect model.pkl
```

## TPU Types

| Accelerator | Chips | Workers | Spot $/hr | Zones |
|-------------|-------|---------|-----------|-------|
| `v4-8` | 4 | 1 | $2.06 | us-central2-b |
| `v4-32` | 16 | 4 | $8.24 | us-central2-b |
| `v5litepod-8` | 8 | 1 | $9.60 | us-central1-a |
| `v5litepod-64` | 64 | 8 | $76.80 | us-central1-a |
| `v6e-8` | 8 | 1 | $9.60 | europe-west4-a |

---


# GPU VM Management

tpuz manages both TPU VMs and GCE GPU VMs with the same API.

## Quick Start

```python
from tpuz import GCE

vm = GCE.gpu("my-vm", gpu="a100", zone="us-central1-a")
vm.up()
vm.setup()
vm.run("python train.py", sync="./src")
vm.logs()
vm.down()
```

## GPU Shorthands

Instead of remembering machine types, use GPU shorthands:

```python
vm = GCE.gpu("my-vm", gpu="a100")      # 1x A100 40GB
vm = GCE.gpu("my-vm", gpu="a100x4")    # 4x A100 40GB
vm = GCE.gpu("my-vm", gpu="a100x8")    # 8x A100 40GB
vm = GCE.gpu("my-vm", gpu="a100-80")   # 1x A100 80GB
vm = GCE.gpu("my-vm", gpu="h100x8")    # 8x H100 80GB
vm = GCE.gpu("my-vm", gpu="t4")        # 1x T4 16GB (cheapest)
vm = GCE.gpu("my-vm", gpu="l4")        # 1x L4 24GB
```

### Full GPU Table

| Shorthand | Machine Type | GPU | Count | VRAM |
|-----------|-------------|-----|-------|------|
| `t4` | n1-standard-8 | Tesla T4 | 1 | 16 GB |
| `t4x2` | n1-standard-16 | Tesla T4 | 2 | 32 GB |
| `t4x4` | n1-standard-32 | Tesla T4 | 4 | 64 GB |
| `l4` | g2-standard-8 | L4 | 1 | 24 GB |
| `l4x2` | g2-standard-16 | L4 | 2 | 48 GB |
| `a100` | a2-highgpu-1g | A100 40GB | 1 | 40 GB |
| `a100x2` | a2-highgpu-2g | A100 40GB | 2 | 80 GB |
| `a100x4` | a2-highgpu-4g | A100 40GB | 4 | 160 GB |
| `a100x8` | a2-megagpu-16g | A100 40GB | 8 | 320 GB |
| `a100-80` | a2-ultragpu-1g | A100 80GB | 1 | 80 GB |
| `h100x8` | a3-highgpu-8g | H100 80GB | 8 | 640 GB |

## Lifecycle

```python
vm.up()         # Create (idempotent — skips if exists, starts if stopped)
vm.stop()       # Stop (keeps disk, stops compute billing)
vm.start()      # Restart a stopped VM
vm.down()       # Delete entirely (disk + compute gone)
vm.info()       # VMInfo(state, machine_type, external_ip)
```

### Stop vs Down

- **`stop()`** — VM stops running, disk is kept. You pay for disk only (~$0.04/GB/month). Use for overnight pauses.
- **`down()`** — VM is deleted. Everything gone. Use when fully done.

## Training

```python
vm.setup()                           # Install JAX[CUDA] + deps
vm.setup(extra_pip="flaxchat")       # + custom packages

vm.run("python train.py",
    sync="./src",                    # Upload code
    secrets=["WANDB_API_KEY"],       # Cloud Secret Manager
    env={"BATCH_SIZE": "32"},        # Extra env vars
)

vm.logs()                            # Stream logs (Ctrl-C to detach)
vm.is_running()                      # Check if alive
vm.kill()                            # Stop training
```

## SSH & Files

```python
vm.ssh("nvidia-smi")                 # Run any command
vm.scp_to("./data", "/home/user/data")
vm.scp_from("/home/user/model.pkl", "./model.pkl")
vm.collect(["model.pkl", "results.json"], local_dir="./outputs")
```

## SSH Tunnel

```python
vm.tunnel(6006)           # TensorBoard: localhost:6006
vm.tunnel(8888, 9999)     # Jupyter: localhost:9999
```

## Custom Machine Types

For advanced use, pass machine type directly:

```python
vm = GCE("my-vm",
    machine_type="n1-highmem-16",    # Custom machine type
    zone="us-west1-b",
    gpu="nvidia-tesla-v100",         # Raw GPU type
    gpu_count=2,
    boot_disk_size="500GB",
    image_family="pytorch-latest-gpu",
)
```

## Spot/Preemptible

Spot VMs are ~60-70% cheaper but can be reclaimed:

```python
vm = GCE.gpu("my-vm", gpu="a100", preemptible=True)   # Default: spot
vm = GCE.gpu("my-vm", gpu="a100", preemptible=False)  # On-demand (stable)
```

Spot VMs are **stopped** (not deleted) on preemption. Your disk and data are preserved. Just `vm.start()` to resume.

## TPU vs GPU: When to Use What

| | TPU (`TPU` class) | GPU (`GCE` class) |
|---|---|---|
| **Best for** | Large JAX models, TPU pods | PyTorch, small JAX, debugging |
| **Multi-device** | Built-in (v4-32 = 4 workers) | Manual (NCCL, torchrun) |
| **Preemption** | VM deleted | VM stopped (disk kept) |
| **JAX install** | `jax[tpu]` | `jax[cuda12]` |
| **Pricing** | v4-8: $2.06/hr spot | A100: ~$3.67/hr spot |
| **Free tier** | TRC program | None |

## Listing VMs

```python
GCE.list(zone="us-central1-a")  # All GCE VMs in zone
```

## Example: Full GPU Training

```python
from tpuz import GCE, GCS, SecretManager

# Setup secrets (one-time)
sm = SecretManager()
sm.create("WANDB_API_KEY", "your-key")
sm.grant_tpu_access_all()

# Create and train
gcs = GCS("gs://my-bucket")
vm = GCE.gpu("train-a100", gpu="a100x4", zone="us-central1-a")
vm.up()
vm.setup(extra_pip="flaxchat")

vm.run("torchrun --nproc_per_node=4 train.py",
    sync="./src",
    secrets=["WANDB_API_KEY", "HF_TOKEN"])

vm.logs()
vm.collect(["model.pkl", "results.json"])
vm.down()
```

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

---


# Best Practices

## Training Workflow

### 1. Start Small, Scale Up

```python
from tpuz import TPU

# Develop on v4-8 (single host, cheap)
dev = TPU("dev", accelerator="v4-8", preemptible=True)
dev.up()
dev.setup(extra_pip="your-package")
dev.repl()  # Interactive development

# Test a short run
dev.run("python train.py --steps=10", sync="./src")
dev.logs()

# When ready, scale up
dev.scale("v4-32")  # Now 4 workers
dev.run("python train.py --steps=50000", sync="./src")
```

### 2. Always Use GCS for Checkpoints

Without GCS, a preemption = lost training. With GCS:

```python
from tpuz import TPU, GCS

gcs = GCS("gs://my-bucket")
tpu = TPU("train", accelerator="v4-8")
tpu.up()
tpu.setup()

# Auto-resumes from latest checkpoint
tpu.run_with_resume("python train.py", gcs=gcs, run_name="run-01", sync="./src")
tpu.watch("python train.py")  # Auto-recover on preemption
```

### 3. Use Cloud Secrets, Not env={}

```python
# GOOD: secrets never leave GCP
tpu.run("python train.py", secrets=["WANDB_API_KEY", "HF_TOKEN"])

# OK for quick tests only
tpu.run("python train.py", env={"WANDB_API_KEY": os.environ["WANDB_API_KEY"]})
```

See [secrets.md](secrets.md) for full setup guide.

### 4. Verify Before Training

```python
tpu.setup(extra_pip="flaxchat")
tpu.verify()  # Confirms JAX works on all workers
# worker 0: 4 devices
# worker 1: 4 devices
# ...
```

### 5. Use Notifications for Long Runs

```python
tpu.watch_notify("python train.py",
    notify_url="https://hooks.slack.com/services/...",
    max_retries=5)
# You'll get Slack messages on: preemption, recovery, completion, failure
```

### 6. Budget with Scheduled Training

```python
tpu.schedule("python train.py",
    start_after="22:00",    # Spot prices are lower at night
    max_cost=50.0,          # Auto-kill at $50
    sync="./src")
```

### 7. Collect Artifacts Before Teardown

```python
tpu.collect(["model.pkl", "results.json", "report.md"], local_dir="./outputs")
tpu.cost_summary()  # Know what it cost
tpu.down()
```

## Multi-Host Best Practices

### Same Code, Different Scale

JAX SPMD means your code is identical on all workers. The only difference is the mesh size:

```python
# This code works on v4-8 (1 worker) AND v4-32 (4 workers)
mesh = jax.sharding.Mesh(jax.devices(), ('data',))
```

### Debug on Single Host First

Multi-host debugging is painful. Always verify on single host:

```bash
tpuz repl my-tpu              # Interactive Python
tpuz debug my-tpu "python train.py"  # VS Code debugger
```

### Monitor All Workers

```bash
tpuz health my-tpu   # Dashboard view
tpuz logs-all my-tpu  # Color-coded per-worker logs
```

### If a Worker Dies

```bash
tpuz health my-tpu
#   worker 0   running     step 1234
#   worker 1   stopped     (no log)    <-- problem!
#   worker 2   running     step 1234

# Option 1: Restart training (relies on checkpoint)
tpuz kill my-tpu
tpuz run my-tpu "python train.py --resume"

# Option 2: Recreate the whole pod
tpuz scale my-tpu v4-32  # Deletes and recreates
```

## Cost Optimization

### Use Preemptible/Spot VMs

Always `preemptible=True` (default). Spot prices are ~3x cheaper:

| Accelerator | On-demand | Spot |
|-------------|-----------|------|
| v4-8 | ~$6.18/hr | **$2.06/hr** |
| v5litepod-8 | ~$28.80/hr | **$9.60/hr** |

### Use Queued Resources for Reliability

```python
tpu.up_queued()  # Waits for capacity instead of failing
```

### Check Before Creating

```python
TPU.availability("v4-8", zone="us-central2-b")
# {"available": True, "spot_rate": 2.06}
```

### Multi-Zone Failover

```python
tpu = TPU.create_multi_zone("my-tpu", "v4-8",
    zones=["us-central2-b", "us-central1-a", "europe-west4-a"])
```

### Track Costs

```python
tpu.cost_summary()
# $12.36 (6.0h × $2.06/hr v4-8 spot)
```

## File Organization

Recommended project structure for TPU training:

```
my-project/
├── src/
│   ├── model.py
│   ├── train.py        # Entry point
│   └── data.py
├── configs/
│   └── v4-8.yaml
├── outputs/            # tpu.collect() downloads here
│   ├── model.pkl
│   └── results.json
└── launch.py           # tpuz orchestration script
```

```python
# launch.py
from tpuz import TPU, GCS

tpu = TPU("train", "v4-8")
gcs = GCS("gs://my-bucket")

tpu.run_once(
    "python src/train.py --config configs/v4-8.yaml",
    sync=".",
    secrets=["WANDB_API_KEY"],
    collect_files=["outputs/model.pkl", "outputs/results.json"],
    gcs=gcs,
    notify_url=os.environ.get("SLACK_WEBHOOK"),
)
```

---

## Acknowledgments

Cloud TPU resources provided by Google's [TPU Research Cloud (TRC)](https://sites.research.google/trc/about/) program.

[GitHub](https://github.com/mlnomadpy/tpuz) · [PyPI](https://pypi.org/p/tpuz) · [kgz](https://github.com/mlnomadpy/kgz)
