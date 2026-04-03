# tpuz — Claude Code Integration

## What This Is

tpuz manages GCP TPU VMs via gcloud CLI. Create, run, monitor, recover, teardown — all from Python or terminal.

## Key Files

- `tpuz/tpu.py` — Core `TPU` class (lifecycle, SSH, multi-host, preemption recovery)
- `tpuz/launcher.py` — `Launcher` for one-command training
- `tpuz/cli.py` — CLI with 12 commands

## Quick Usage

```python
from tpuz import TPU

tpu = TPU("my-tpu", accelerator="v4-8")
tpu.up()                           # Create VM
tpu.setup()                        # Install JAX
tpu.run("python train.py")         # Launch training
tpu.logs()                         # Stream logs
tpu.down()                         # Delete VM
```

## One-liner

```python
from tpuz import Launcher
Launcher("my-tpu", "v4-8").train("python train.py", sync="./src", auto_recover=True)
```

## CLI

```bash
tpuz up my-tpu -a v4-8
tpuz run my-tpu "python train.py" --sync=./src
tpuz logs my-tpu
tpuz down my-tpu
tpuz train my-tpu "python train.py" -a v4-8 --sync=. --recover
```

## Requires

- `gcloud` CLI installed and authenticated
- GCP project with TPU quota
