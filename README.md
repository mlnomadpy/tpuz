# tpuz

Manage GCP TPU VMs from your terminal. Create, train, debug, recover, teardown — one command.

```bash
pip install tpuz
```

## Why?

Training on TPU pods requires 10+ gcloud commands, manual SSH to each worker, no preemption handling, no cost visibility, and painful debugging. tpuz wraps it all:

```python
from tpuz import TPU

tpu = TPU("my-tpu", accelerator="v4-8")
tpu.up()
tpu.setup()
tpu.run("python train.py", sync="./src")
tpu.logs()
tpu.down()
```

Or in one command:

```bash
tpuz train my-tpu "python train.py" -a v4-8 --sync=. --recover --teardown
```

## Features

### Lifecycle

```python
tpu.preflight()            # Verify gcloud config
tpu.up()                   # Create VM (idempotent)
tpu.up_queued()            # Queued Resources API (reliable spot)
tpu.down()                 # Delete VM
tpu.info()                 # State, IPs, accelerator
tpu.setup(extra_pip="jax") # Install JAX[TPU] + deps
tpu.verify()               # Verify JAX on all workers
```

### Training

```python
tpu.run("python train.py", sync="./src", env={"KEY": "val"})
tpu.logs()                 # Stream logs (Ctrl-C to detach)
tpu.logs_all()             # Color-coded logs from ALL workers
tpu.is_running()           # Check if alive
tpu.kill()                 # Stop training
tpu.wait()                 # Poll for COMPLETE/FAILED
tpu.collect(["model.pkl"]) # Download artifacts
```

### Cost Tracking

```python
tpu.cost_summary()  # "$4.12 (2.0h x $2.06/hr v4-8 spot)"
```

### GCS Checkpoint Sync

```python
from tpuz import GCS

gcs = GCS("gs://my-bucket")
gcs.upload_checkpoint("./ckpt", "run-01", step=1000)
gcs.latest_step("run-01")   # 5000
gcs.list_runs()              # ["run-01", "run-02"]

# Auto-resume from latest checkpoint
tpu.run_with_resume("python train.py", gcs=gcs, run_name="run-01")
# Finds step 5000 -> appends --resume-from-step=5000
```

### Preemption Recovery

```python
tpu.watch("python train.py", max_retries=5)
# Polls every 60s -> on PREEMPTED: delete -> recreate -> setup -> restart

# With Slack notifications
tpu.watch_notify("python train.py",
    notify_url="https://hooks.slack.com/services/...",
    max_retries=5)
```

### Debugging

```python
tpu.repl()                                # Interactive Python on worker 0
tpu.debug("python train.py", port=5678)   # VS Code debugger attach
tpu.logs_all(lines=20)                    # All workers side by side
tpu.health_pretty()                       # Worker dashboard:
#   Worker     Status          Last Log
#   -------------------------------------------
#   worker 0   running         step 1234 | loss 2.31
#   worker 1   running         step 1234 | loss 2.31
#   worker 2   stopped         (no log)
```

### SSH Tunnel

```python
tpu.tunnel(6006)           # TensorBoard: localhost:6006
tpu.tunnel(8888, 9999)     # Jupyter: localhost:9999 -> TPU:8888
```

### Scaling

```python
tpu.scale("v4-32")  # Delete -> recreate with v4-32 -> re-setup
```

### Multi-Zone Failover

```python
tpu = TPU.create_multi_zone("my-tpu", "v4-8",
    zones=["us-central2-b", "us-central1-a", "europe-west4-a"])
```

### Availability Check

```python
TPU.availability("v4-8", zone="us-central2-b")
# {"available": True, "spot_rate": 2.06, "on_demand_rate": 6.18}
```

### Run-Once (Docker-like)

```python
tpu.run_once("python train.py",
    sync="./src",
    collect_files=["model.pkl", "results.json"],
    gcs=gcs,
    notify_url="https://hooks.slack.com/...")
# up -> setup -> resume -> run -> wait -> collect -> notify -> down
```

### Scheduled Training

```python
tpu.schedule("python train.py",
    start_after="22:00",   # Wait until 10 PM
    max_cost=10.0)         # Kill if exceeds $10
```

### Environment Snapshot/Restore

```python
tpu.snapshot_env(gcs=gcs)   # pip freeze -> GCS
tpu.restore_env(gcs=gcs)    # Restore after preemption
```

### Secrets (Cloud Secret Manager)

**Recommended:** Use Google Cloud Secret Manager. Secrets never leave GCP:

```python
from tpuz import SecretManager

# One-time setup
sm = SecretManager()
sm.create("WANDB_API_KEY", "your-key")
sm.grant_tpu_access_all()

# Training: VM reads secrets directly from GCP
tpu.run("python train.py", secrets=["WANDB_API_KEY", "HF_TOKEN"])
```

**Fallback:** `env={}` writes a `.env` file via SCP (encrypted, but secrets transit your machine).

See [docs/secrets.md](docs/secrets.md) for full setup guide and security comparison.

## Multi-Host (TPU Pods)

Worker count auto-detected. All SSH commands run in parallel with per-worker retries:

| Accelerator | Chips | Workers | Spot $/hr |
|-------------|-------|---------|-----------|
| `v4-8` | 4 | 1 | $2.06 |
| `v4-32` | 16 | 4 | $8.24 |
| `v5litepod-8` | 8 | 1 | $9.60 |
| `v5litepod-64` | 64 | 8 | $76.80 |
| `v6e-8` | 8 | 1 | $9.60 |

## CLI

```bash
# Lifecycle
tpuz up my-tpu -a v4-8
tpuz down my-tpu
tpuz status my-tpu
tpuz list
tpuz preflight
tpuz avail v4-8

# Training
tpuz setup my-tpu --pip="flaxchat"
tpuz verify my-tpu
tpuz run my-tpu "python train.py" --sync=./src
tpuz logs my-tpu
tpuz logs-all my-tpu
tpuz kill my-tpu
tpuz wait my-tpu
tpuz collect my-tpu model.pkl results.json

# Debugging
tpuz repl my-tpu
tpuz debug my-tpu "python train.py"
tpuz health my-tpu
tpuz tunnel my-tpu 6006
tpuz scale my-tpu v4-32
tpuz cost my-tpu

# Recovery
tpuz watch my-tpu "python train.py"

# All-in-one
tpuz train my-tpu "python train.py" -a v4-8 --sync=. --recover --teardown
tpuz run-once my-tpu "python train.py" --sync=. --collect model.pkl
```

## Development Workflow

```python
from tpuz import TPU

# 1. Develop on single host
dev = TPU("dev", accelerator="v4-8")
dev.up()
dev.setup()
dev.repl()  # Interactive development

# 2. Test training
dev.run("python train.py --steps=10", sync="./src")
dev.logs()

# 3. Scale up
dev.scale("v4-32")  # 4 workers now
dev.run("python train.py --steps=50000", sync="./src")
dev.watch("python train.py --steps=50000")

# 4. Collect and cleanup
dev.collect(["model.pkl", "results.json"])
dev.cost_summary()  # $12.36
dev.down()
```

## Documentation

- **[docs/secrets.md](docs/secrets.md)** — Secrets & security guide (Cloud Secret Manager setup)
- **[docs/best-practices.md](docs/best-practices.md)** — Training workflow, cost optimization, multi-host tips
- **[SKILL.md](SKILL.md)** — Claude Code skill reference
- **[CLAUDE.md](CLAUDE.md)** — Quick reference for AI agents

## Requirements

- `gcloud` CLI installed and authenticated
- GCP project with TPU quota
- Python 3.10+
- Zero Python dependencies

## Pair with kgz

```bash
pip install kgz     # Kaggle free GPUs
pip install tpuz    # GCP TPU pods
```

## Claude Code Integration

```bash
mkdir -p ~/.claude/skills/tpuz-guide
cp SKILL.md ~/.claude/skills/tpuz-guide/skill.md
```

## License

MIT

## Acknowledgments

Cloud TPU resources for developing and testing tpuz were provided by Google's [TPU Research Cloud (TRC)](https://sites.research.google/trc/about/) program. We gratefully acknowledge their support in making TPU access available for open-source research.
