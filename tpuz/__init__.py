"""
tpuz — Manage GCP TPU VMs from your terminal.

Usage:
    from tpuz import TPU

    tpu = TPU("my-tpu", accelerator="v4-8")
    tpu.up()
    tpu.run("python train.py")
    tpu.logs()
    tpu.down()

CLI:
    tpuz up my-tpu --accelerator=v4-8
    tpuz run my-tpu "python train.py"
    tpuz logs my-tpu
    tpuz down my-tpu
"""

from tpuz.tpu import TPU, TPUInfo
from tpuz.launcher import Launcher

__version__ = "0.1.0"
__all__ = ["TPU", "TPUInfo", "Launcher"]
