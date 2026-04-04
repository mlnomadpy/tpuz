---
layout: page
title: Usage Guide
permalink: /guide/
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
