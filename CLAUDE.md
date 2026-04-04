# tpuz — Claude Code Integration

## What This Is

tpuz manages GCP TPU & GPU VMs. Create, train, debug, monitor, recover, teardown — from Python or CLI.

## Key Modules

| Module | What |
|--------|------|
| `tpu.py` | TPU VM lifecycle, SSH, training, debugging, scaling, secrets, costs, budget |
| `gce.py` | GCE GPU VMs (A100/H100/T4) with same API |
| `gcs.py` | GCS checkpoint sync |
| `secrets.py` | Cloud Secret Manager |
| `costs.py` | Cost tracking with TPU/GPU hourly rates |
| `health.py` | Heartbeat, disk, GPU idle, training progress parser |
| `profiles.py` | Save/load VM configs |
| `audit.py` | Action history logging |
| `notify.py` | Slack/webhook notifications |
| `launcher.py` | One-command training orchestrator |

## Quick Usage

```python
from tpuz import TPU, GCS, SecretManager

tpu = TPU("my-tpu", accelerator="v4-8")
tpu.up()
tpu.setup()
tpu.run("python train.py", secrets=["WANDB_API_KEY"], sync="./src")
tpu.logs()
tpu.health_check()     # Process, heartbeat, disk, GPU, training metrics
tpu.cost_summary()     # $4.12
tpu.down()
```

## IMPORTANT: Secrets

Always use Cloud Secret Manager:
```python
tpu.run(cmd, secrets=["WANDB_API_KEY"])  # GOOD: never leaves GCP
tpu.run(cmd, env={"KEY": "val"})         # OK for quick tests only
```

## GPU VMs

```python
from tpuz import GCE
vm = GCE.gpu("my-vm", gpu="a100")  # Same API: up/setup/run/logs/down
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

## Tests

```bash
pytest tests/ -v  # 48 tests, no GCP needed
```
