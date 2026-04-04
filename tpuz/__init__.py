"""
tpuz — Manage GCP TPU & GPU VMs from your terminal.
"""
from tpuz.tpu import TPU, TPUInfo
from tpuz.gce import GCE
from tpuz.launcher import Launcher
from tpuz.gcs import GCS
from tpuz.costs import CostTracker, hourly_rate
from tpuz.secrets import SecretManager
from tpuz.health import HealthMonitor, parse_training_progress
from tpuz.profiles import save_profile, load_profile, list_profiles
from tpuz.audit import log_action, get_history

__version__ = "0.1.0"
__all__ = [
    "TPU", "TPUInfo", "GCE", "Launcher", "GCS",
    "CostTracker", "hourly_rate", "SecretManager",
    "HealthMonitor", "parse_training_progress",
    "save_profile", "load_profile", "list_profiles",
    "log_action", "get_history",
]
