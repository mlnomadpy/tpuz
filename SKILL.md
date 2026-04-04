# tpuz — Manage GCP TPU & GPU VMs

Use when user wants to manage TPU/GPU VMs, run training on GCP, or when you see `import tpuz`.

## Install

```bash
pip install tpuz
```

## TPU

```python
from tpuz import TPU
tpu = TPU("name", accelerator="v4-8", zone="us-central2-b", preemptible=True)
tpu.preflight()              # Check gcloud config
tpu.up()                     # Create (idempotent)
tpu.up_queued()              # Queued Resources (reliable spot)
tpu.setup(extra_pip="pkg")   # Install JAX[TPU] + deps
tpu.verify()                 # Check JAX on all workers
tpu.run("cmd", sync=".", secrets=["KEY"], env={"K": "V"})
tpu.logs()                   # Stream (Ctrl-C detach)
tpu.logs_all()               # All workers color-coded
tpu.wait()                   # Poll for COMPLETE/FAILED
tpu.collect(["model.pkl"])   # Download artifacts
tpu.down()                   # Delete
```

## GPU

```python
from tpuz import GCE
vm = GCE.gpu("name", gpu="a100")  # a100, a100x4, h100x8, t4, l4
vm.up(); vm.setup(); vm.run("cmd"); vm.logs(); vm.down()
vm.stop()    # Pause (keeps disk)
vm.start()   # Resume
```

## Secrets (Cloud Secret Manager — ALWAYS prefer this)

```python
from tpuz import SecretManager
sm = SecretManager()
sm.create("WANDB_API_KEY", "key"); sm.grant_tpu_access_all()
tpu.run("cmd", secrets=["WANDB_API_KEY"])  # Never leaves GCP
```

## GCS Checkpoints

```python
from tpuz import GCS
gcs = GCS("gs://bucket")
gcs.upload_checkpoint("./ckpt", "run", step=1000)
gcs.latest_step("run")  # 5000
tpu.run_with_resume("cmd", gcs=gcs)  # Auto-resume
```

## Debugging

```python
tpu.repl()                    # Interactive Python
tpu.debug("cmd", port=5678)  # VS Code debugpy
tpu.tunnel(6006)              # TensorBoard
tpu.health_check()            # Full dashboard
tpu.training_progress()       # Parse step/loss/lr from log
```

## Cost & Budget

```python
tpu.cost_summary()                              # $4.12
tpu.set_budget(50, notify_url=slack)            # Kill at $50
tpu.schedule("cmd", start_after="22:00", max_cost=10)
TPU.availability("v4-8")                        # Check price/capacity
```

## Recovery

```python
tpu.watch("cmd", max_retries=5)                 # Auto-recover preemption
tpu.watch_notify("cmd", notify_url=slack)       # + Slack alerts
```

## Scaling

```python
tpu.scale("v4-32")                              # Upgrade accelerator
TPU.create_multi_zone("name", "v4-8", zones=[...])  # Multi-zone failover
```

## Profiles & Dry Run

```python
tpu.save_profile("big-run")                     # Save config
tpu = TPU.from_profile("big-run", "new-tpu")   # Reuse
tpu.dry_run("cmd", sync=".", secrets=["KEY"])   # Preview without executing
```

## Run-Once (Docker-like)

```python
tpu.run_once("cmd", sync=".", collect_files=["model.pkl"], gcs=gcs, notify_url=slack)
# up → setup → resume → run → wait → collect → notify → down
```

## Environment

```python
tpu.snapshot_env(gcs=gcs)   # pip freeze → GCS
tpu.restore_env(gcs=gcs)    # Restore after preemption
```

## CLI

```bash
tpuz up/down/status/list/preflight/avail
tpuz setup/verify/run/logs/logs-all/kill/wait/collect
tpuz repl/debug/health/tunnel/scale/cost
tpuz watch/train/run-once
```

## TPU Types

v4-8 ($2.06/hr), v4-32 ($8.24), v5litepod-8 ($9.60), v5litepod-64 ($76.80), v6e-8 ($9.60)
