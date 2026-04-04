# tpuz — Manage GCP TPU VMs

Use this knowledge when the user wants to create/manage TPU VMs, run training on GCP TPUs, debug multi-host issues, or when you see `import tpuz` or `from tpuz import`.

## Install

```bash
pip install tpuz
```

Requires: `gcloud` CLI installed and authenticated.

## Core API

```python
from tpuz import TPU

tpu = TPU("my-tpu", accelerator="v4-8", zone="us-central2-b", preemptible=True)

tpu.preflight()          # Verify gcloud config
tpu.up()                 # Create VM (idempotent)
tpu.up_queued()          # Create via Queued Resources (reliable spot)
tpu.down()               # Delete VM
info = tpu.info()        # TPUInfo(state, accelerator, ips)

tpu.setup(extra_pip="flaxchat")  # Install JAX[TPU] + deps
tpu.verify()                      # Verify JAX on all workers

tpu.ssh("echo hello")                 # Single worker
tpu.ssh_all("echo hello", retries=3)  # All workers parallel with retries
tpu.scp_to("./src", "/remote/src")

tpu.run("python train.py", sync="./src", env={"KEY": "val"})
tpu.logs()                    # Stream logs
tpu.logs_all()                # All workers color-coded
tpu.wait()                    # Poll for COMPLETE/FAILED
tpu.health_pretty()           # Worker dashboard
tpu.watch("python train.py")  # Auto-recover preemption
```

## Cost Tracking

```python
tpu.cost_summary()  # "$4.12 (2.0h x $2.06/hr v4-8 spot)"
```

## GCS Checkpoint Sync

```python
from tpuz import GCS
gcs = GCS("gs://bucket")
gcs.upload_checkpoint("./ckpt", "run-01", step=1000)
gcs.latest_step("run-01")  # 5000
tpu.run_with_resume("python train.py", gcs=gcs)  # Auto-resume
```

## SSH Tunnel

```python
tpu.tunnel(6006)  # TensorBoard: localhost:6006
```

## Debugging

```python
tpu.repl()                    # Interactive Python on worker 0
tpu.debug("python train.py")  # VS Code debugpy attach
tpu.scale("v4-32")            # Scale up accelerator
```

## Multi-Zone Failover

```python
tpu = TPU.create_multi_zone("my-tpu", "v4-8",
    zones=["us-central2-b", "us-central1-a"])
```

## Run-Once

```python
tpu.run_once("python train.py", sync=".", collect_files=["model.pkl"],
    gcs=gcs, notify_url="https://hooks.slack.com/...")
# up -> setup -> resume -> run -> wait -> collect -> notify -> down
```

## Scheduled Training

```python
tpu.schedule("python train.py", start_after="22:00", max_cost=10.0)
```

## Environment Snapshot

```python
tpu.snapshot_env(gcs=gcs)  # pip freeze -> GCS
tpu.restore_env(gcs=gcs)   # Restore after preemption
```

## CLI

```bash
tpuz up NAME -a v4-8
tpuz run NAME "python train.py" --sync=.
tpuz logs NAME
tpuz health NAME
tpuz cost NAME
tpuz tunnel NAME 6006
tpuz repl NAME
tpuz wait NAME
tpuz run-once NAME "python train.py" --sync=. --notify=URL
tpuz train NAME "python train.py" -a v4-8 --recover --teardown
```

## Source

- Repo: `/Users/tahabsn/Documents/GitHub/tpuz`
- 1,896 lines, 33 tests, zero Python deps

## Google Cloud Secret Manager

```python
from tpuz import SecretManager

# Store secrets in GCP (one-time)
sm = SecretManager(project="my-project")
sm.create("WANDB_API_KEY", "your-key")
sm.create("HF_TOKEN", "hf_...")
sm.grant_tpu_access_all()  # Grant VM access via IAM

# List/read/delete
sm.list()                    # ["WANDB_API_KEY", "HF_TOKEN"]
sm.get("WANDB_API_KEY")     # "your-key"
sm.exists("WANDB_API_KEY")  # True
sm.delete("OLD_KEY")

# Training: secrets loaded server-side, never leave GCP
tpu.run("python train.py", secrets=["WANDB_API_KEY", "HF_TOKEN"])
# Or load manually:
tpu.load_secrets(["WANDB_API_KEY", "HF_TOKEN"])
```

IMPORTANT: Always prefer `secrets=["KEY"]` over `env={"KEY": "val"}`.
The `secrets` param uses Cloud Secret Manager (server-side, never leaves GCP).
The `env` param writes a .env file via SCP (encrypted but secrets transit your machine).
