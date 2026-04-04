"""
tpuz — Manage GCP TPU VMs from your terminal.

Usage:
    from tpuz import TPU, GCS

    tpu = TPU("my-tpu", accelerator="v4-8")
    tpu.up()
    tpu.run("python train.py")
    tpu.logs()
    tpu.cost_summary()  # $2.14
    tpu.down()

    gcs = GCS("gs://my-bucket")
    gcs.upload_checkpoint("./ckpt", "run-01", step=1000)
"""

from tpuz.tpu import TPU, TPUInfo
from tpuz.launcher import Launcher
from tpuz.gcs import GCS
from tpuz.costs import CostTracker, hourly_rate

__version__ = "0.1.0"
__all__ = ["TPU", "TPUInfo", "Launcher", "GCS", "CostTracker", "hourly_rate"]
