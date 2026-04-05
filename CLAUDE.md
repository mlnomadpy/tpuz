# tpuz — Claude Code Integration

## What This Is

tpuz manages GCP TPU & GPU VMs. Create, train, debug, monitor, recover, teardown — from Python or CLI.

## Key Modules

| Module | What |
|--------|------|
| `tpu.py` | TPU VM lifecycle, SSH (configurable timeout), training, debugging, scaling, secrets, costs, budget, profiles, audit |
| `gce.py` | GCE GPU VMs (A100/H100/T4/L4) with same API |
| `gcs.py` | GCS checkpoint sync, upload/download/list runs |
| `secrets.py` | Cloud Secret Manager (create/get/grant/load) |
| `costs.py` | Cost tracking with TPU/GPU hourly rates (spot + on-demand) |
| `health.py` | Heartbeat, disk, GPU idle, training progress parser, ETA |
| `profiles.py` | Save/load/list/delete VM configs |
| `audit.py` | JSONL action history logging |
| `notify.py` | Slack/webhook notifications |
| `launcher.py` | One-command training orchestrator |
| `cli.py` | 27 CLI subcommands |

## Quick Usage

```python
from tpuz import TPU, GCS, SecretManager

tpu = TPU("my-tpu", accelerator="v4-8", ssh_timeout=120)
tpu.up()
tpu.setup()                 # waits for SSH + apt lock automatically
tpu.run("python train.py", secrets=["WANDB_API_KEY"], sync="./src")
tpu.logs()
tpu.health_check()          # process, heartbeat, disk, GPU, training metrics
tpu.cost_summary()          # $4.12
tpu.down()
```

## IMPORTANT: Secrets

Always use Cloud Secret Manager:
```python
tpu.run(cmd, secrets=["WANDB_API_KEY"])  # GOOD: never leaves GCP
tpu.run(cmd, env={"KEY": "val"})         # OK for quick tests only
```

## SSH Timeout

```python
# Default: 120s
tpu = TPU("name", ssh_timeout=120)

# Custom default
tpu = TPU("name", ssh_timeout=300)

# Disable timeout
tpu = TPU("name", ssh_timeout=None)

# Per-call override
tpu.ssh("cmd", timeout=600)
tpu.ssh("cmd", timeout=None)  # no timeout
```

## Setup (uses uv — 10-50x faster than pip)

```python
tpu.setup()                              # waits for SSH, installs uv + JAX
tpu.setup(python_version="3.11")         # installs Python 3.11 via uv
tpu.setup(extra_pip="flaxchat wandb")    # + custom packages
```

`setup()` automatically:
1. Waits for SSH readiness (up to 180s)
2. Installs uv (astral.sh/uv) — Rust-native, 10-50x faster
3. Installs target Python version via `uv python install`
4. Installs all packages in one shot with `uv pip install` (no build isolation conflicts)

## GPU VMs

```python
from tpuz import GCE
vm = GCE.gpu("my-vm", gpu="a100")  # Same API: up/setup/run/logs/down
vm.stop()                           # Pause (keeps disk)
vm.start()                          # Resume
```

## GCS Checkpoints

```python
gcs = GCS("gs://bucket")
gcs.upload_checkpoint("./ckpt", "run", step=1000)
gcs.latest_step("run")              # 5000
tpu.run_with_resume("cmd", gcs=gcs) # auto-resume from latest
```

## Health & Monitoring

```python
tpu.health_check()        # full dashboard
tpu.training_progress()   # parse step/loss/lr/tok_s/mfu
tpu.monitor()             # HealthMonitor instance
```

## Cost & Budget

```python
tpu.cost_summary()                              # $4.12
tpu.set_budget(50, notify_url=slack)            # kill at $50
tpu.schedule("cmd", start_after="22:00", max_cost=10)
```

## Recovery

```python
tpu.watch("cmd", max_retries=5)                 # auto-recover preemption
tpu.watch_notify("cmd", notify_url=slack)       # + Slack alerts
```

## Dry Run

```python
tpu.dry_run("python train.py", sync="./src", secrets=["WANDB_API_KEY"])
# Prints all gcloud commands without executing
```

## Profiles

```python
tpu.save_profile("big-run")
tpu2 = TPU.from_profile("big-run", "new-name")
```

## Run-Once (Docker-like)

```python
tpu.run_once("cmd", sync=".", collect_files=["model.pkl"], gcs=gcs, notify_url=slack)
# up → setup → resume → run → wait → collect → notify → down
```

## Debugging

```python
tpu.repl()                    # interactive Python on worker 0
tpu.debug("cmd", port=5678)  # VS Code debugpy
tpu.tunnel(6006)              # TensorBoard
```

## Tests

```bash
pytest tests/ -v  # 54 tests, no GCP needed
```

## CLI

```bash
tpuz up/down/status/list/preflight/avail/runtimes
tpuz setup/verify/run/logs/logs-all/kill/wait/collect
tpuz repl/debug/health/tunnel/scale/cost
tpuz watch/train/run-once
tpuz ssh/upload/download
```
