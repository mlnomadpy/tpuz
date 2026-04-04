# tpuz — Claude Code Integration

## What This Is

tpuz manages GCP TPU VMs via gcloud CLI. Create, run, debug, monitor, recover, teardown.

## Key Files

- `tpuz/tpu.py` — Core `TPU` class (lifecycle, SSH, multi-host, debugging, secrets, costs)
- `tpuz/gcs.py` — GCS checkpoint sync
- `tpuz/secrets.py` — Google Cloud Secret Manager integration
- `tpuz/costs.py` — Cost tracking with TPU hourly rates
- `tpuz/notify.py` — Slack/webhook notifications
- `tpuz/launcher.py` — One-command training orchestrator
- `tpuz/cli.py` — CLI with 25+ commands

## Quick Usage

```python
from tpuz import TPU, GCS, SecretManager

tpu = TPU("my-tpu", accelerator="v4-8")
tpu.up()
tpu.setup()
tpu.run("python train.py", secrets=["WANDB_API_KEY"], sync="./src")
tpu.logs()
tpu.cost_summary()
tpu.down()
```

## Secrets (IMPORTANT)

Always use Cloud Secret Manager, not env vars:
```python
# GOOD: secrets never leave GCP
tpu.run("python train.py", secrets=["WANDB_API_KEY", "HF_TOKEN"])

# OK for quick tests only
tpu.run("python train.py", env={"KEY": "val"})
```

## Docs

- `docs/secrets.md` — Full secrets & security guide
- `docs/best-practices.md` — Training workflow, costs, multi-host tips

## Running Tests

```bash
pytest tests/ -v  # 35 tests, no GCP needed
```
