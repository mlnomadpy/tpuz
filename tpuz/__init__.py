"""
tpuz — Manage GCP TPU VMs and GPU instances from your terminal.
"""
from tpuz.tpu import TPU, TPUInfo
from tpuz.gce import GCE
from tpuz.launcher import Launcher
from tpuz.gcs import GCS
from tpuz.costs import CostTracker, hourly_rate
from tpuz.secrets import SecretManager

__version__ = "0.1.0"
__all__ = ["TPU", "TPUInfo", "GCE", "Launcher", "GCS", "CostTracker", "hourly_rate", "SecretManager"]
