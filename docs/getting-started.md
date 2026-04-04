---
layout: page
title: Getting Started
permalink: /getting-started/
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
