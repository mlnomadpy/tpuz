<div align="center">

# tpuz

**Manage GCP TPU & GPU VMs from your terminal.**

Create, train, debug, recover, teardown — one command.

[![PyPI](https://img.shields.io/pypi/v/tpuz)](https://pypi.org/project/tpuz/)
[![Tests](https://img.shields.io/github/actions/workflow/status/mlnomadpy/tpuz/test.yaml?label=tests)](https://github.com/mlnomadpy/tpuz/actions)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)

[Getting Started](https://www.tahabouhsine.com/tpuz/getting-started/) · [Docs](https://www.tahabouhsine.com/tpuz/guide/) · [GPU Guide](https://www.tahabouhsine.com/tpuz/gpu/) · [Security](https://www.tahabouhsine.com/tpuz/secrets/)

</div>

---

## Why?

Training on GCP TPUs/GPUs means juggling `gcloud` commands, SSH sessions, preemption recovery, cost tracking, and secrets. tpuz handles all of it:

```python
from tpuz import TPU

tpu = TPU("my-tpu", accelerator="v4-8")
tpu.up()
tpu.setup()
tpu.run("python train.py", secrets=["WANDB_API_KEY"], sync="./src")
tpu.logs()
tpu.cost_summary()  # $4.12 (2.0h × $2.06/hr)
tpu.down()
```

Or one command:

```bash
tpuz train my-tpu "python train.py" -a v4-8 --sync=. --recover --teardown
```

## Install

```bash
pip install tpuz
```

Zero Python dependencies. Requires `gcloud` CLI ([install](https://cloud.google.com/sdk/docs/install)).

## Features

### Core

```python
tpu.up()                    # Create TPU VM (idempotent)
tpu.up_queued()             # Queued Resources (reliable spot)
tpu.setup()                 # Install JAX[TPU] + deps
tpu.verify()                # Check JAX on all workers
tpu.run("cmd", sync=".")    # Upload code + launch
tpu.logs()                  # Stream training logs
tpu.wait()                  # Poll for completion
tpu.collect(["model.pkl"])  # Download artifacts
tpu.down()                  # Delete VM
```

### GPU VMs

```python
from tpuz import GCE

vm = GCE.gpu("my-vm", gpu="a100")   # A100 40GB
vm = GCE.gpu("my-vm", gpu="h100x8") # 8x H100
vm = GCE.gpu("my-vm", gpu="t4")     # T4 (cheapest)
vm.up()                              # Same API as TPU
```

### Secrets (Cloud Secret Manager)

```python
from tpuz import SecretManager

sm = SecretManager()
sm.create("WANDB_API_KEY", "your-key")
sm.grant_tpu_access_all()

tpu.run("python train.py", secrets=["WANDB_API_KEY"])
# Secrets never leave GCP — loaded server-side via IAM
```

### Checkpoints (GCS)

```python
from tpuz import GCS

gcs = GCS("gs://my-bucket")
tpu.run_with_resume("python train.py", gcs=gcs, run_name="run-01")
# Auto-detects latest checkpoint → appends --resume-from-step=5000
```

### Preemption Recovery

```python
tpu.watch_notify("python train.py",
    notify_url="https://hooks.slack.com/...",
    max_retries=5)
# Auto: delete → recreate → setup → restart from checkpoint → Slack notify
```

### Debugging

```python
tpu.repl()                             # Interactive Python REPL
tpu.debug("python train.py")           # VS Code debugger attach
tpu.tunnel(6006)                       # TensorBoard
tpu.health_check()                     # Full health dashboard:
#   Process:   running
#   Heartbeat: fresh (12s ago)
#   Disk:      45% (90/200 GB)
#   GPU:       85% utilization
#   Training:  step 1234/5000 | loss 2.31 | 56,000 tok/s
#   ETA:       ~35m
```

### Cost Control

```python
tpu.cost_summary()                     # $4.12 (2.0h × $2.06/hr)
tpu.set_budget(50, notify_url=slack)   # Alert at $40, kill at $50
tpu.schedule("python train.py",
    start_after="22:00", max_cost=10)  # Train overnight, budget $10
```

### Scaling & Failover

```python
tpu.scale("v4-32")                     # Upgrade: v4-8 → v4-32
TPU.create_multi_zone("tpu", "v4-8",
    zones=["us-central2-b", "europe-west4-a"])  # Try each zone
```

### Run-Once (Docker-like)

```python
tpu.run_once("python train.py",
    sync="./src", collect_files=["model.pkl"],
    gcs=gcs, notify_url=slack)
# up → setup → resume → run → wait → collect → notify → down
```

### Profiles & Audit

```python
tpu.save_profile("big-run")            # Save config for reuse
tpu = TPU.from_profile("big-run", "new-tpu")  # Reuse later
tpu.dry_run("python train.py")         # Preview commands without executing
```

## Multi-Host (TPU Pods)

Auto-detected. All SSH commands parallel with per-worker retries:

| Accelerator | Chips | Workers | Spot $/hr |
|-------------|-------|---------|-----------|
| `v4-8` | 4 | 1 | $2.06 |
| `v4-32` | 16 | 4 | $8.24 |
| `v5litepod-8` | 8 | 1 | $9.60 |
| `v5litepod-64` | 64 | 8 | $76.80 |
| `v6e-8` | 8 | 1 | $9.60 |

## CLI

```bash
tpuz up NAME -a v4-8          tpuz logs NAME
tpuz down NAME                tpuz logs-all NAME
tpuz status NAME              tpuz health NAME
tpuz setup NAME               tpuz tunnel NAME 6006
tpuz verify NAME              tpuz repl NAME
tpuz run NAME "cmd" --sync=.  tpuz debug NAME "cmd"
tpuz wait NAME                tpuz scale NAME v4-32
tpuz kill NAME                tpuz cost NAME
tpuz collect NAME files...    tpuz avail v4-8
tpuz watch NAME "cmd"         tpuz preflight
tpuz train NAME "cmd" -a v4-8 --recover --teardown
tpuz run-once NAME "cmd" --sync=. --collect model.pkl
```

## Documentation

- **[Getting Started](https://www.tahabouhsine.com/tpuz/getting-started/)** — Zero to training in 9 steps
- **[Usage Guide](https://www.tahabouhsine.com/tpuz/guide/)** — Every feature explained
- **[GPU VMs](https://www.tahabouhsine.com/tpuz/gpu/)** — A100/H100/T4 management
- **[Secrets & Security](https://www.tahabouhsine.com/tpuz/secrets/)** — Cloud Secret Manager setup
- **[Best Practices](https://www.tahabouhsine.com/tpuz/best-practices/)** — Production workflows

## Pair with kgz

```bash
pip install kgz     # Kaggle free GPUs — execute code remotely
pip install tpuz    # GCP TPU/GPU pods — manage VM lifecycle
```

## Claude Code

```bash
mkdir -p ~/.claude/skills/tpuz-guide
cp SKILL.md ~/.claude/skills/tpuz-guide/skill.md
```

## Acknowledgments

Cloud TPU resources provided by Google's [TPU Research Cloud (TRC)](https://sites.research.google/trc/about/) program.

## License

MIT
