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

Or one command:

```bash
tpuz train my-tpu "python train.py" -a v4-8 --sync=. --recover --teardown
```

---

# Getting Started

## Prerequisites

1. **gcloud CLI**: `brew install google-cloud-sdk` (macOS) or [cloud.google.com/sdk](https://cloud.google.com/sdk/docs/install)
2. **Authenticate**: `gcloud auth login`
3. **Set project**: `gcloud config set project YOUR_PROJECT`

## Step 1: Check Setup

```python
from tpuz import TPU

tpu = TPU("test", accelerator="v4-8")
tpu.preflight()
# gcloud account: you@gmail.com
# gcloud project: my-project
```

## Step 2: Create VM

```python
tpu.up()           # Standard creation
tpu.up_queued()    # Queued Resources (waits for capacity — better for spot)
```

## Step 3: Install Dependencies

```python
tpu.setup()                          # JAX[TPU] + common deps
tpu.setup(extra_pip="flaxchat")      # + custom packages
tpu.verify()                         # Confirm JAX works on all workers
```

## Step 4: Upload Code & Run

```python
tpu.run("python train.py",
    sync="./src",                          # Upload local directory
    secrets=["WANDB_API_KEY", "HF_TOKEN"], # From Cloud Secret Manager
)
```

## Step 5: Monitor

```python
tpu.logs()                  # Stream logs (Ctrl-C to detach)
tpu.logs_all(lines=20)      # All workers color-coded
tpu.health_check()          # Full dashboard
tpu.training_progress()     # Parsed step/loss/lr from logs
tpu.cost_summary()          # Running cost
```

## Step 6: Collect & Cleanup

```python
tpu.collect(["model.pkl", "results.json"], local_dir="./outputs")
tpu.cost_summary()   # $4.12 (2.0h × $2.06/hr v4-8 spot)
tpu.down()
```

---

# Complete API Reference

## Lifecycle

```python
tpu.preflight()                # Verify gcloud config
tpu.up()                       # Create VM (idempotent)
tpu.up_queued(timeout_hours=2) # Queued Resources API
tpu.down()                     # Delete VM
tpu.down_queued()              # Delete QR + VM
tpu.info()                     # TPUInfo(state, accelerator, ips)
tpu.describe()                 # Alias for info()
tpu.setup(extra_pip="pkg")     # Install JAX[TPU] + deps
tpu.verify()                   # Check JAX on all workers
```

## SSH & Files

```python
# SSH — returns stdout string by default
output = tpu.ssh("echo hello")

# Structured result with stderr and returncode
result = tpu.ssh("cat /tmp/log", structured=True)
result.stdout       # "log content..."
result.stderr       # ""
result.returncode   # 0
result.ok           # True

# All workers in parallel (with per-worker retries)
outputs = tpu.ssh_all("hostname", retries=3)

# File transfer
tpu.scp_to("./src", "/home/user/src")
tpu.scp_from("/home/user/model.pkl", "./model.pkl")

# Aliases
tpu.push("./config.json", "/home/user/config.json")
tpu.pull("/home/user/metrics.db", "./metrics.db")
```

## Training

```python
tpu.run("python train.py",
    sync="./src",                    # Upload code
    secrets=["WANDB_API_KEY"],       # Cloud Secret Manager
    env={"BATCH_SIZE": "32"},        # Extra env vars (.env file)
)

tpu.logs()                    # Stream (Ctrl-C to detach)
tpu.logs_all()                # All workers color-coded
tpu.is_running()              # Check if alive
tpu.kill()                    # Stop training
tpu.wait()                    # Poll for COMPLETE/FAILED sentinel
tpu.collect(["model.pkl"])    # Download artifacts
```

## Git Repo Setup

```python
tpu.clone_repo(
    url="https://github.com/org/repo.git",
    branch="main",
    install=True,                      # pip install -e . after clone
    github_token="ghp_...",            # For private repos
)
# Clones/pulls on ALL workers in parallel
```

## GCS Checkpoints

```python
from tpuz import GCS

gcs = GCS("gs://my-bucket")
gcs.upload_checkpoint("./ckpt", "run-01", step=1000)
gcs.download_checkpoint("run-01", step=1000)
gcs.latest_step("run-01")        # 5000
gcs.checkpoint_path("run-01")    # "gs://my-bucket/checkpoints/run-01"
gcs.list_runs()                  # ["run-01", "run-02"]

# Auto-resume from latest checkpoint
tpu.run_with_resume("python train.py", gcs=gcs, run_name="run-01")
# Finds step 5000 → appends --resume-from-step=5000
```

## Secrets (Cloud Secret Manager)

**Always prefer `secrets=` over `env=`.** Secrets never leave GCP.

```python
from tpuz import SecretManager

# One-time setup
sm = SecretManager()
sm.create("WANDB_API_KEY", "your-key")  # Idempotent: creates or updates
sm.ensure("HF_TOKEN", "hf_...")          # Create-or-skip (no update)
sm.grant_tpu_access_all()                # Grant VM access via IAM

# Use in training
tpu.run("python train.py", secrets=["WANDB_API_KEY", "HF_TOKEN"])

# Manage
sm.list()                    # ["WANDB_API_KEY", "HF_TOKEN"]
sm.get("WANDB_API_KEY")     # "your-key"
sm.exists("WANDB_API_KEY")  # True
sm.delete("OLD_KEY")
```

### Security Comparison

| | `env={}` | `secrets=[]` |
|---|:-:|:-:|
| Secret on your machine | Yes | **No** |
| In transit | SCP (encrypted) | **Never** |
| On VM disk | .env file | **Memory only** |
| Survives preemption | Must re-send | **Automatic** |

Full guide: [Secrets & Security](#secrets--security)

## Preemption Recovery

```python
# Basic — auto recreate + restart
tpu.watch("python train.py", max_retries=5)

# With Slack notifications
tpu.watch_notify("python train.py",
    notify_url="https://hooks.slack.com/...",
    max_retries=5)
```

Flow: Poll every 60s → PREEMPTED → delete → recreate → setup → restart from checkpoint.

## Debugging

```python
tpu.repl()                              # Interactive Python on worker 0
tpu.debug("python train.py", port=5678) # VS Code debugpy attach
tpu.tunnel(6006)                        # TensorBoard: localhost:6006
tpu.tunnel(8888, 9999)                  # Jupyter: localhost:9999
```

## Health Dashboard

```python
tpu.health_check()
#   Health Check for 'my-tpu'
#   ==================================================
#   Process:   running
#   Heartbeat: fresh (12s ago)
#   Disk:      45% (90/200 GB)
#   GPU:       85% utilization
#   Training:  step 1234/5000 | loss 2.31 | 56,000 tok/s
#   ETA:       ~35m

# Just training metrics
tpu.training_progress()
# {"step": 1234, "total_steps": 5000, "loss": 2.31, "tok_per_sec": 56000}

# Worker-level health
tpu.health_pretty()
#   worker 0   running    step 1234 | loss 2.31
#   worker 1   running    step 1234 | loss 2.31
#   worker 2   stopped    (no log)
```

## Cost Tracking

```python
tpu.cost_summary()   # "$4.12 (2.0h × $2.06/hr v4-8 spot)"

# Budget enforcement
tpu.set_budget(max_usd=50,
    notify_url="https://hooks.slack.com/...")
# Alerts at $40, kills training at $50

# Check before creating
TPU.availability("v4-8", zone="us-central2-b")
# {"available": True, "spot_rate": 2.06, "on_demand_rate": 6.18}
```

## Scaling

```python
tpu.scale("v4-32")   # Delete → recreate with v4-32 → re-setup
# Same code works — JAX handles the mesh
```

## Multi-Zone Failover

```python
tpu = TPU.create_multi_zone("my-tpu", "v4-8",
    zones=["us-central2-b", "us-central1-a", "europe-west4-a"])
```

## Scheduled Training

```python
tpu.schedule("python train.py",
    start_after="22:00",     # Cheaper spot at night
    max_cost=10.0,           # Kill if exceeds $10
    sync="./src")
```

## Run-Once (Docker-like)

```python
tpu.run_once("python train.py",
    sync="./src",
    collect_files=["model.pkl", "results.json"],
    gcs=gcs,
    notify_url="https://hooks.slack.com/...")
# up → setup → resume → run → wait → collect → notify → down
```

## One-Liner

```python
from tpuz import Launcher

Launcher("my-tpu", "v4-8").train(
    command="python train.py",
    sync="./src",
    setup_pip="flaxchat",
    auto_recover=True,
    teardown_after=True,
)
```

## Environment Snapshot

```python
tpu.snapshot_env(gcs=gcs)   # pip freeze → GCS
tpu.restore_env(gcs=gcs)    # Restore after preemption
```

## Profiles

```python
tpu.save_profile("big-run")                      # Save config
tpu = TPU.from_profile("big-run", "new-tpu")    # Reuse
```

## Dry Run

```python
tpu.dry_run("python train.py", sync="./src", secrets=["KEY"])
# Prints all gcloud commands without executing
```

## Config Upload (Multi-Host)

```python
tpu.upload_config({"model": {...}, "training": {...}})
# JSON → worker 0 → internal network copy to all workers
```

## Worker Count

```python
TPU.num_workers_for("v4-8")    # 1
TPU.num_workers_for("v4-32")   # 4
TPU.num_workers_for("v6e-64")  # 8
```

---

# GPU VMs

Same API for GCE GPU instances:

```python
from tpuz import GCE

vm = GCE.gpu("my-vm", gpu="a100")      # 1x A100 40GB
vm = GCE.gpu("my-vm", gpu="a100x4")    # 4x A100
vm = GCE.gpu("my-vm", gpu="h100x8")    # 8x H100
vm = GCE.gpu("my-vm", gpu="t4")        # 1x T4 (cheapest)

vm.up()
vm.setup()                               # JAX[CUDA] + deps
vm.run("python train.py", sync="./src")
vm.logs()
vm.stop()                                # Pause (keeps disk)
vm.start()                               # Resume
vm.down()                                # Delete
```

### GPU Shorthands

| Shorthand | GPU | Count | VRAM |
|-----------|-----|-------|------|
| `t4` | Tesla T4 | 1 | 16 GB |
| `l4` | L4 | 1 | 24 GB |
| `a100` | A100 40GB | 1 | 40 GB |
| `a100x4` | A100 40GB | 4 | 160 GB |
| `a100x8` | A100 40GB | 8 | 320 GB |
| `h100x8` | H100 80GB | 8 | 640 GB |

---

# Secrets & Security

## Three Approaches

### 1. Cloud Secret Manager (recommended)

```python
from tpuz import SecretManager
sm = SecretManager()
sm.create("WANDB_API_KEY", "key")
sm.grant_tpu_access_all()
tpu.run("cmd", secrets=["WANDB_API_KEY"])
```

Secrets never leave GCP. VM reads them via its service account.

### 2. env dict (fallback)

```python
tpu.run("cmd", env={"KEY": "val"})
```

Written to `.env` file via SCP. Not in command line or `ps aux`, but transits your machine.

### Setup

```bash
# Enable API
gcloud services enable secretmanager.googleapis.com

# Store secret
echo -n "your-key" | gcloud secrets create WANDB_API_KEY --data-file=-

# Grant TPU access
gcloud secrets add-iam-policy-binding WANDB_API_KEY \
    --member="serviceAccount:$(gcloud iam service-accounts list --format='value(email)' --filter='Compute Engine')" \
    --role="roles/secretmanager.secretAccessor"
```

---

# Multi-Host (TPU Pods)

Worker count auto-detected. All SSH parallel with retries:

| Accelerator | Chips | Workers | Spot $/hr |
|-------------|-------|---------|-----------|
| `v4-8` | 4 | 1 | $2.06 |
| `v4-32` | 16 | 4 | $8.24 |
| `v5litepod-8` | 8 | 1 | $9.60 |
| `v5litepod-64` | 64 | 8 | $76.80 |
| `v6e-8` | 8 | 1 | $9.60 |

---

# CLI Reference

```bash
# Lifecycle
tpuz up NAME -a v4-8          tpuz preflight
tpuz down NAME                tpuz avail v4-8
tpuz status NAME              tpuz runtimes
tpuz list

# Training
tpuz setup NAME --pip=PKG     tpuz logs NAME
tpuz verify NAME              tpuz logs-all NAME
tpuz run NAME "cmd" --sync=.  tpuz kill NAME
tpuz wait NAME                tpuz collect NAME files...

# Debugging
tpuz repl NAME                tpuz tunnel NAME 6006
tpuz debug NAME "cmd"         tpuz scale NAME v4-32
tpuz health NAME              tpuz cost NAME

# Recovery & All-in-one
tpuz watch NAME "cmd"
tpuz train NAME "cmd" -a v4-8 --sync=. --recover --teardown
tpuz run-once NAME "cmd" --sync=. --collect model.pkl
```

---

# Best Practices

1. **Start small, scale up**: Develop on v4-8, scale to v4-32 when ready
2. **Always use GCS**: Preemption = lost training without checkpoint persistence
3. **Use Cloud Secrets**: Never pass API keys via `env={}`
4. **Verify before training**: `tpu.verify()` catches silent multi-host failures
5. **Use notifications**: `watch_notify()` sends Slack on preemption/completion
6. **Budget your runs**: `set_budget(50)` prevents surprise bills
7. **Collect before teardown**: `tpu.collect()` then `tpu.down()`
8. **Use Queued Resources**: `up_queued()` is more reliable than `up()` for spot

---

## Acknowledgments

Cloud TPU resources provided by Google's [TPU Research Cloud (TRC)](https://sites.research.google/trc/about/) program.

[GitHub](https://github.com/mlnomadpy/tpuz) · [PyPI](https://pypi.org/p/tpuz) · [kgz](https://github.com/mlnomadpy/kgz)
