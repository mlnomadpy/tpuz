---
layout: page
title: GPU VMs
permalink: /gpu/
---

# GPU VM Management

tpuz manages both TPU VMs and GCE GPU VMs with the same API.

## Quick Start

```python
from tpuz import GCE

vm = GCE.gpu("my-vm", gpu="a100", zone="us-central1-a")
vm.up()
vm.setup()
vm.run("python train.py", sync="./src")
vm.logs()
vm.down()
```

## GPU Shorthands

Instead of remembering machine types, use GPU shorthands:

```python
vm = GCE.gpu("my-vm", gpu="a100")      # 1x A100 40GB
vm = GCE.gpu("my-vm", gpu="a100x4")    # 4x A100 40GB
vm = GCE.gpu("my-vm", gpu="a100x8")    # 8x A100 40GB
vm = GCE.gpu("my-vm", gpu="a100-80")   # 1x A100 80GB
vm = GCE.gpu("my-vm", gpu="h100x8")    # 8x H100 80GB
vm = GCE.gpu("my-vm", gpu="t4")        # 1x T4 16GB (cheapest)
vm = GCE.gpu("my-vm", gpu="l4")        # 1x L4 24GB
```

### Full GPU Table

| Shorthand | Machine Type | GPU | Count | VRAM |
|-----------|-------------|-----|-------|------|
| `t4` | n1-standard-8 | Tesla T4 | 1 | 16 GB |
| `t4x2` | n1-standard-16 | Tesla T4 | 2 | 32 GB |
| `t4x4` | n1-standard-32 | Tesla T4 | 4 | 64 GB |
| `l4` | g2-standard-8 | L4 | 1 | 24 GB |
| `l4x2` | g2-standard-16 | L4 | 2 | 48 GB |
| `a100` | a2-highgpu-1g | A100 40GB | 1 | 40 GB |
| `a100x2` | a2-highgpu-2g | A100 40GB | 2 | 80 GB |
| `a100x4` | a2-highgpu-4g | A100 40GB | 4 | 160 GB |
| `a100x8` | a2-megagpu-16g | A100 40GB | 8 | 320 GB |
| `a100-80` | a2-ultragpu-1g | A100 80GB | 1 | 80 GB |
| `h100x8` | a3-highgpu-8g | H100 80GB | 8 | 640 GB |

## Lifecycle

```python
vm.up()         # Create (idempotent — skips if exists, starts if stopped)
vm.stop()       # Stop (keeps disk, stops compute billing)
vm.start()      # Restart a stopped VM
vm.down()       # Delete entirely (disk + compute gone)
vm.info()       # VMInfo(state, machine_type, external_ip)
```

### Stop vs Down

- **`stop()`** — VM stops running, disk is kept. You pay for disk only (~$0.04/GB/month). Use for overnight pauses.
- **`down()`** — VM is deleted. Everything gone. Use when fully done.

## Training

```python
vm.setup()                           # Install JAX[CUDA] + deps
vm.setup(extra_pip="flaxchat")       # + custom packages

vm.run("python train.py",
    sync="./src",                    # Upload code
    secrets=["WANDB_API_KEY"],       # Cloud Secret Manager
    env={"BATCH_SIZE": "32"},        # Extra env vars
)

vm.logs()                            # Stream logs (Ctrl-C to detach)
vm.is_running()                      # Check if alive
vm.kill()                            # Stop training
```

## SSH & Files

```python
vm.ssh("nvidia-smi")                 # Run any command
vm.scp_to("./data", "/home/user/data")
vm.scp_from("/home/user/model.pkl", "./model.pkl")
vm.collect(["model.pkl", "results.json"], local_dir="./outputs")
```

## SSH Tunnel

```python
vm.tunnel(6006)           # TensorBoard: localhost:6006
vm.tunnel(8888, 9999)     # Jupyter: localhost:9999
```

## Custom Machine Types

For advanced use, pass machine type directly:

```python
vm = GCE("my-vm",
    machine_type="n1-highmem-16",    # Custom machine type
    zone="us-west1-b",
    gpu="nvidia-tesla-v100",         # Raw GPU type
    gpu_count=2,
    boot_disk_size="500GB",
    image_family="pytorch-latest-gpu",
)
```

## Spot/Preemptible

Spot VMs are ~60-70% cheaper but can be reclaimed:

```python
vm = GCE.gpu("my-vm", gpu="a100", preemptible=True)   # Default: spot
vm = GCE.gpu("my-vm", gpu="a100", preemptible=False)  # On-demand (stable)
```

Spot VMs are **stopped** (not deleted) on preemption. Your disk and data are preserved. Just `vm.start()` to resume.

## TPU vs GPU: When to Use What

| | TPU (`TPU` class) | GPU (`GCE` class) |
|---|---|---|
| **Best for** | Large JAX models, TPU pods | PyTorch, small JAX, debugging |
| **Multi-device** | Built-in (v4-32 = 4 workers) | Manual (NCCL, torchrun) |
| **Preemption** | VM deleted | VM stopped (disk kept) |
| **JAX install** | `jax[tpu]` | `jax[cuda12]` |
| **Pricing** | v4-8: $2.06/hr spot | A100: ~$3.67/hr spot |
| **Free tier** | TRC program | None |

## Listing VMs

```python
GCE.list(zone="us-central1-a")  # All GCE VMs in zone
```

## Example: Full GPU Training

```python
from tpuz import GCE, GCS, SecretManager

# Setup secrets (one-time)
sm = SecretManager()
sm.create("WANDB_API_KEY", "your-key")
sm.grant_tpu_access_all()

# Create and train
gcs = GCS("gs://my-bucket")
vm = GCE.gpu("train-a100", gpu="a100x4", zone="us-central1-a")
vm.up()
vm.setup(extra_pip="flaxchat")

vm.run("torchrun --nproc_per_node=4 train.py",
    sync="./src",
    secrets=["WANDB_API_KEY", "HF_TOKEN"])

vm.logs()
vm.collect(["model.pkl", "results.json"])
vm.down()
```
