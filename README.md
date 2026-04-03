# tpuz

Manage GCP TPU VMs from your terminal. Create, train, recover, teardown — zero config.

```bash
pip install tpuz
```

## Why?

GCP TPU VMs need 10+ gcloud commands to set up, run training, handle preemption, and clean up. tpuz wraps it all into simple Python/CLI commands.

## Quick Start

### Python

```python
from tpuz import TPU

tpu = TPU("my-tpu", accelerator="v4-8", zone="us-central2-b")

tpu.up()                                    # Create TPU VM
tpu.setup()                                 # Install JAX[TPU] + deps
tpu.run("python train.py", sync="./src")    # Upload code + run
tpu.logs()                                  # Stream training logs (Ctrl-C to detach)
tpu.down()                                  # Delete VM
```

### CLI

```bash
# Step by step
tpuz up my-tpu -a v4-8
tpuz setup my-tpu
tpuz run my-tpu "python train.py" --sync=./src
tpuz logs my-tpu
tpuz down my-tpu

# Or all at once
tpuz train my-tpu "python train.py" -a v4-8 --sync=. --recover --teardown
```

### One-Liner

```python
from tpuz import Launcher

Launcher("my-tpu", "v4-8").train(
    command="python train.py",
    sync="./src",
    setup_pip="flaxchat",
    env={"WANDB_API_KEY": "..."},
    auto_recover=True,      # Auto-restart on preemption
    teardown_after=True,    # Delete VM when done
)
```

## Features

| Feature | Command | Description |
|---------|---------|-------------|
| Create | `tpu.up()` | Idempotent — skips if exists |
| Delete | `tpu.down()` | Clean teardown |
| SSH | `tpu.ssh("cmd")` | Run any command |
| Multi-host SSH | `tpu.ssh_all("cmd")` | Parallel on all workers |
| Setup | `tpu.setup()` | Install JAX[TPU] + deps |
| Run | `tpu.run("cmd")` | Detached nohup with PID tracking |
| Logs | `tpu.logs()` | Real-time log streaming |
| Upload | `tpu.scp_to(local, remote)` | Copy files to VM |
| Download | `tpu.scp_from(remote, local)` | Copy from VM |
| Status | `tpu.info()` | State, IPs, accelerator |
| Kill | `tpu.kill()` | Stop training process |
| Recovery | `tpu.watch("cmd")` | Auto-recreate on preemption |
| List | `TPU.list()` | List all TPU VMs |

## Multi-Host (TPU Pods)

Automatically detected from accelerator type:

| Accelerator | Chips | Workers | Auto-handled |
|-------------|-------|---------|-------------|
| `v4-8` | 4 | 1 | Single-host |
| `v4-32` | 16 | 4 | `ssh_all()` parallel |
| `v5litepod-64` | 64 | 8 | `ssh_all()` parallel |
| `v6e-128` | 128 | 16 | `ssh_all()` parallel |

```python
tpu = TPU("big-pod", accelerator="v5litepod-64")
tpu.up()
tpu.ssh_all("hostname")  # Runs on all 8 workers in parallel
```

## Preemption Recovery

```python
# Auto-recover: recreate VM + restart training on preemption
tpu.watch("python train.py", max_retries=5)
```

Or via CLI:
```bash
tpuz watch my-tpu "python train.py" --retries=5
```

How it works:
1. Polls VM state every 60s
2. On PREEMPTED/TERMINATED: delete → recreate → setup → restart
3. Training resumes from checkpoint (Orbax/safetensors)
4. Max 5 retries

## TPU Types & Zones

| Type | Chips | Free Tier | Zones |
|------|-------|-----------|-------|
| `v4-8` | 4 | On-demand | us-central2-b |
| `v5litepod-8` | 8 | TRC | us-central1-a |
| `v5litepod-64` | 64 | TRC | us-central1-a |
| `v6e-8` | 8 | TRC | europe-west4-a |

## Requirements

- `gcloud` CLI installed and authenticated (`gcloud auth login`)
- GCP project with TPU quota (`gcloud config set project my-project`)
- Python 3.10+

## Pair with kgz

```bash
pip install kgz     # Kaggle free GPUs — execute code remotely
pip install tpuz    # GCP TPU pods — manage VM lifecycle
```

## License

MIT

## Claude Code Integration

tpuz includes a `SKILL.md` for [Claude Code](https://claude.ai/claude-code). To enable it:

```bash
mkdir -p ~/.claude/skills/tpuz-guide
cp SKILL.md ~/.claude/skills/tpuz-guide/skill.md
```

This gives Claude Code full knowledge of the tpuz API so it can manage TPU VMs on your behalf.
