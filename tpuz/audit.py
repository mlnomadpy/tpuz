"""
Audit log — records every tpuz action with timestamps.
"""

import os
import json
import time
from datetime import datetime


AUDIT_PATH = os.path.expanduser("~/.tpuz/audit.jsonl")


def log_action(action, tpu_name="", details=None):
    """Append an action to the audit log."""
    os.makedirs(os.path.dirname(AUDIT_PATH), exist_ok=True)
    entry = {
        "timestamp": datetime.now().isoformat(),
        "epoch": time.time(),
        "action": action,
        "tpu": tpu_name,
    }
    if details:
        entry["details"] = details
    with open(AUDIT_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def get_history(tpu_name=None, last_n=50):
    """Read audit log, optionally filtered by TPU name."""
    if not os.path.exists(AUDIT_PATH):
        return []
    entries = []
    with open(AUDIT_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if tpu_name is None or entry.get("tpu") == tpu_name:
                    entries.append(entry)
            except json.JSONDecodeError:
                pass
    return entries[-last_n:]


def print_history(tpu_name=None, last_n=20):
    """Print formatted audit log."""
    entries = get_history(tpu_name, last_n)
    if not entries:
        print("No actions recorded.")
        return
    for e in entries:
        ts = e.get("timestamp", "?")[:19]
        action = e.get("action", "?")
        tpu = e.get("tpu", "")
        details = e.get("details", "")
        det_str = f" — {details}" if details else ""
        print(f"  {ts}  {tpu:15s}  {action}{det_str}")


def clear_history():
    """Clear the audit log."""
    if os.path.exists(AUDIT_PATH):
        os.unlink(AUDIT_PATH)
