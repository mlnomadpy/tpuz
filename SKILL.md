# tpuz — Manage GCP TPU VMs

Use this knowledge when the user wants to create/manage TPU VMs, run training on GCP TPUs, or when you see `import tpuz` or `from tpuz import`.

## Install

```bash
pip install tpuz
```

Requires: `gcloud` CLI installed and authenticated.

## Core API

```python
from tpuz import TPU

tpu = TPU("my-tpu", accelerator="v4-8", zone="us-central2-b", preemptible=True)

# Lifecycle
tpu.up()             # Create VM (idempotent)
tpu.down()           # Delete VM
info = tpu.info()    # TPUInfo(state, accelerator, ips)

# SSH
output = tpu.ssh("echo hello")           # Single worker
outputs = tpu.ssh_all("echo hello")      # All workers parallel

# File transfer
tpu.scp_to("./src", "/home/user/src")
tpu.scp_from("/home/user/model.pkl", "./model.pkl")

# Setup
tpu.setup()                          # Install JAX[TPU] + deps
tpu.setup(extra_pip="flaxchat")      # + custom packages

# Training
tpu.run("python train.py", sync="./src")   # Upload code + launch detached
tpu.logs()                                  # Stream logs (Ctrl-C to detach)
tpu.is_running()                            # Check if training alive
tpu.kill()                                  # Stop training

# Preemption recovery
tpu.watch("python train.py", max_retries=5)  # Auto-recreate on preemption
```

## One-Command Training

```python
from tpuz import Launcher

Launcher("my-tpu", "v4-8").train(
    command="python train.py",
    sync="./src",                # Upload local dir
    setup_pip="flaxchat",        # Extra packages
    env={"WANDB_API_KEY": "..."}, # Env vars
    auto_recover=True,           # Handle preemption
    teardown_after=True,         # Delete VM when done
)
```

## TPU Types & Workers

| Accelerator | Chips | Workers | Runtime |
|-------------|-------|---------|---------|
| `v4-8` | 4 | 1 | v2-alpha-tpuv4 |
| `v4-32` | 16 | 4 | v2-alpha-tpuv4 |
| `v5litepod-8` | 8 | 1 | v2-alpha-tpuv5-lite |
| `v5litepod-64` | 64 | 8 | v2-alpha-tpuv5-lite |
| `v6e-8` | 8 | 1 | v2-alpha-tpuv6e |

Worker count auto-detected: `<=8 chips → 1, <=16 → 2, else chips//8`
Runtime version auto-detected from accelerator prefix.

## Multi-Host

For pods with >8 chips, `ssh_all()` runs commands on all workers in parallel:

```python
tpu = TPU("pod", accelerator="v5litepod-64")  # 8 workers
tpu.ssh_all("hostname")  # Returns list of 8 outputs
tpu.run("python train.py")  # Launches on all workers
```

## CLI

```bash
# Step by step
tpuz up my-tpu -a v4-8
tpuz setup my-tpu
tpuz run my-tpu "python train.py" --sync=./src
tpuz logs my-tpu
tpuz status my-tpu
tpuz kill my-tpu
tpuz down my-tpu

# All at once
tpuz train my-tpu "python train.py" -a v4-8 --sync=. --recover --teardown

# Other
tpuz ssh my-tpu "nvidia-smi"
tpuz upload my-tpu local.py /remote/path
tpuz download my-tpu /remote/file ./local
tpuz list
tpuz watch my-tpu "python train.py" --retries=5
```

## Preemption Recovery Flow

1. `tpu.watch(cmd)` polls VM state every 60s
2. On PREEMPTED/TERMINATED → delete → recreate → setup → restart
3. Training resumes from checkpoint (Orbax handles this)
4. Max retries configurable (default 5)

## Common Zones

| Zone | TPU Types | Notes |
|------|-----------|-------|
| us-central2-b | v4 | On-demand |
| us-central1-a | v5e | TRC free tier |
| europe-west4-a | v6e | TRC free tier |

## Environment

```python
# Set GCP project
tpu = TPU("x", project="my-gcp-project")
# Or via env
os.environ["GCLOUD_PROJECT"] = "my-gcp-project"
```

## Source

- Repo: `/Users/tahabsn/Documents/GitHub/tpuz`
- 820 lines, 12 tests
- Zero Python deps (calls gcloud CLI)
