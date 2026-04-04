---
layout: home
title: tpuz
---

# tpuz

Manage GCP TPU & GPU VMs from your terminal. Create, train, debug, recover, teardown — one command.

```bash
pip install tpuz
```

```python
from tpuz import TPU

tpu = TPU("my-tpu", accelerator="v4-8")
tpu.up()
tpu.setup()
tpu.run("python train.py", secrets=["WANDB_API_KEY"], sync="./src")
tpu.logs()
tpu.cost_summary()
tpu.down()
```

## Documentation

1. **[Getting Started](getting-started.md)** — From zero to training, step by step
2. **[Complete Usage Guide](guide.md)** — Every feature explained with examples
3. **[GPU VMs](gpu.md)** — Managing A100/H100/T4 GPU instances
4. **[Secrets & Security](secrets.md)** — Cloud Secret Manager setup and best practices
5. **[Best Practices](best-practices.md)** — Production workflows, cost optimization, multi-host tips

## At a Glance

| Category | Features |
|----------|----------|
| **TPU VMs** | `up`, `up_queued`, `down`, `setup`, `verify`, `scale` |
| **GPU VMs** | `GCE.gpu("a100")`, `stop`/`start`, same SSH/run/logs API |
| **Training** | `run`, `logs`, `logs_all`, `wait`, `kill`, `collect` |
| **Debugging** | `repl`, `debug` (VS Code), `health`, `tunnel` |
| **Secrets** | Cloud Secret Manager, `.env` fallback, auto-redaction |
| **Checkpoints** | GCS upload/download, `latest_step`, `run_with_resume` |
| **Costs** | `cost_summary`, hourly rates, `schedule` with budget |
| **Recovery** | `watch`, `watch_notify`, Queued Resources, multi-zone |

## Acknowledgments

Cloud TPU resources provided by Google's [TPU Research Cloud (TRC)](https://sites.research.google/trc/about/) program.

[GitHub](https://github.com/tahabsn/tpuz) | [PyPI](https://pypi.org/p/tpuz)
