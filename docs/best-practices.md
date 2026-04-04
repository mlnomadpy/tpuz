---
layout: page
title: Best Practices
permalink: /best-practices/
---

# Best Practices

## Training Workflow

### 1. Start Small, Scale Up

```python
from tpuz import TPU

# Develop on v4-8 (single host, cheap)
dev = TPU("dev", accelerator="v4-8", preemptible=True)
dev.up()
dev.setup(extra_pip="your-package")
dev.repl()  # Interactive development

# Test a short run
dev.run("python train.py --steps=10", sync="./src")
dev.logs()

# When ready, scale up
dev.scale("v4-32")  # Now 4 workers
dev.run("python train.py --steps=50000", sync="./src")
```

### 2. Always Use GCS for Checkpoints

Without GCS, a preemption = lost training. With GCS:

```python
from tpuz import TPU, GCS

gcs = GCS("gs://my-bucket")
tpu = TPU("train", accelerator="v4-8")
tpu.up()
tpu.setup()

# Auto-resumes from latest checkpoint
tpu.run_with_resume("python train.py", gcs=gcs, run_name="run-01", sync="./src")
tpu.watch("python train.py")  # Auto-recover on preemption
```

### 3. Use Cloud Secrets, Not env={}

```python
# GOOD: secrets never leave GCP
tpu.run("python train.py", secrets=["WANDB_API_KEY", "HF_TOKEN"])

# OK for quick tests only
tpu.run("python train.py", env={"WANDB_API_KEY": os.environ["WANDB_API_KEY"]})
```

See [secrets.md](secrets.md) for full setup guide.

### 4. Verify Before Training

```python
tpu.setup(extra_pip="flaxchat")
tpu.verify()  # Confirms JAX works on all workers
# worker 0: 4 devices
# worker 1: 4 devices
# ...
```

### 5. Use Notifications for Long Runs

```python
tpu.watch_notify("python train.py",
    notify_url="https://hooks.slack.com/services/...",
    max_retries=5)
# You'll get Slack messages on: preemption, recovery, completion, failure
```

### 6. Budget with Scheduled Training

```python
tpu.schedule("python train.py",
    start_after="22:00",    # Spot prices are lower at night
    max_cost=50.0,          # Auto-kill at $50
    sync="./src")
```

### 7. Collect Artifacts Before Teardown

```python
tpu.collect(["model.pkl", "results.json", "report.md"], local_dir="./outputs")
tpu.cost_summary()  # Know what it cost
tpu.down()
```

## Multi-Host Best Practices

### Same Code, Different Scale

JAX SPMD means your code is identical on all workers. The only difference is the mesh size:

```python
# This code works on v4-8 (1 worker) AND v4-32 (4 workers)
mesh = jax.sharding.Mesh(jax.devices(), ('data',))
```

### Debug on Single Host First

Multi-host debugging is painful. Always verify on single host:

```bash
tpuz repl my-tpu              # Interactive Python
tpuz debug my-tpu "python train.py"  # VS Code debugger
```

### Monitor All Workers

```bash
tpuz health my-tpu   # Dashboard view
tpuz logs-all my-tpu  # Color-coded per-worker logs
```

### If a Worker Dies

```bash
tpuz health my-tpu
#   worker 0   running     step 1234
#   worker 1   stopped     (no log)    <-- problem!
#   worker 2   running     step 1234

# Option 1: Restart training (relies on checkpoint)
tpuz kill my-tpu
tpuz run my-tpu "python train.py --resume"

# Option 2: Recreate the whole pod
tpuz scale my-tpu v4-32  # Deletes and recreates
```

## Cost Optimization

### Use Preemptible/Spot VMs

Always `preemptible=True` (default). Spot prices are ~3x cheaper:

| Accelerator | On-demand | Spot |
|-------------|-----------|------|
| v4-8 | ~$6.18/hr | **$2.06/hr** |
| v5litepod-8 | ~$28.80/hr | **$9.60/hr** |

### Use Queued Resources for Reliability

```python
tpu.up_queued()  # Waits for capacity instead of failing
```

### Check Before Creating

```python
TPU.availability("v4-8", zone="us-central2-b")
# {"available": True, "spot_rate": 2.06}
```

### Multi-Zone Failover

```python
tpu = TPU.create_multi_zone("my-tpu", "v4-8",
    zones=["us-central2-b", "us-central1-a", "europe-west4-a"])
```

### Track Costs

```python
tpu.cost_summary()
# $12.36 (6.0h × $2.06/hr v4-8 spot)
```

## File Organization

Recommended project structure for TPU training:

```
my-project/
├── src/
│   ├── model.py
│   ├── train.py        # Entry point
│   └── data.py
├── configs/
│   └── v4-8.yaml
├── outputs/            # tpu.collect() downloads here
│   ├── model.pkl
│   └── results.json
└── launch.py           # tpuz orchestration script
```

```python
# launch.py
from tpuz import TPU, GCS

tpu = TPU("train", "v4-8")
gcs = GCS("gs://my-bucket")

tpu.run_once(
    "python src/train.py --config configs/v4-8.yaml",
    sync=".",
    secrets=["WANDB_API_KEY"],
    collect_files=["outputs/model.pkl", "outputs/results.json"],
    gcs=gcs,
    notify_url=os.environ.get("SLACK_WEBHOOK"),
)
```
