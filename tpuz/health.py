"""
Health monitoring — heartbeat checks, disk usage, idle detection, progress parsing.
"""

import re
import time


def parse_training_progress(log_line):
    """
    Parse training metrics from a log line.
    Handles common formats:
        step 100 | loss 3.71 | dt 0.55s | tok/s 56000
        Step 100/5000 (2.0%) | loss: 3.71 | lr: 0.0003
        epoch 1 step 100 loss=3.71 lr=3e-4

    Returns dict with parsed fields, or empty dict if no match.
    """
    metrics = {}

    # Step
    m = re.search(r'step\s*[:=]?\s*(\d+)', log_line, re.IGNORECASE)
    if m:
        metrics['step'] = int(m.group(1))

    # Total steps
    m = re.search(r'step\s*\d+\s*/\s*(\d+)', log_line, re.IGNORECASE)
    if m:
        metrics['total_steps'] = int(m.group(1))

    # Loss
    m = re.search(r'loss\s*[:=]?\s*([\d.]+)', log_line, re.IGNORECASE)
    if m:
        metrics['loss'] = float(m.group(1))

    # Learning rate
    m = re.search(r'lr\s*[:=]?\s*([\d.e-]+)', log_line, re.IGNORECASE)
    if m:
        try:
            metrics['lr'] = float(m.group(1))
        except ValueError:
            pass

    # Tokens per second
    m = re.search(r'tok/?s\s*[:=]?\s*([\d,]+)', log_line, re.IGNORECASE)
    if m:
        metrics['tok_per_sec'] = int(m.group(1).replace(',', ''))

    # dt (time per step)
    m = re.search(r'dt\s*[:=]?\s*([\d.]+)\s*(?:ms|s)?', log_line, re.IGNORECASE)
    if m:
        metrics['dt'] = float(m.group(1))

    # Percentage
    m = re.search(r'(\d+\.?\d*)%', log_line)
    if m:
        metrics['percent'] = float(m.group(1))

    # MFU
    m = re.search(r'mfu\s*[:=]?\s*([\d.]+)', log_line, re.IGNORECASE)
    if m:
        metrics['mfu'] = float(m.group(1))

    # Epoch
    m = re.search(r'epoch\s*[:=]?\s*(\d+)', log_line, re.IGNORECASE)
    if m:
        metrics['epoch'] = int(m.group(1))

    return metrics


def estimate_eta(metrics, start_time=None):
    """Estimate time remaining from parsed metrics."""
    if 'step' in metrics and 'total_steps' in metrics:
        remaining = metrics['total_steps'] - metrics['step']
        if 'dt' in metrics:
            eta_s = remaining * metrics['dt']
            if metrics.get('dt') < 1:  # dt in seconds
                pass
            else:  # dt in milliseconds
                eta_s = remaining * metrics['dt'] / 1000
            return eta_s
    return None


class HealthMonitor:
    """
    Monitor training health via heartbeat files and log parsing.

    Usage:
        monitor = HealthMonitor(tpu)
        monitor.check()  # Returns health status dict
    """

    def __init__(self, tpu, heartbeat_interval=60):
        self.tpu = tpu
        self.heartbeat_interval = heartbeat_interval
        self._heartbeat_path = f"{tpu.workdir}/.heartbeat"

    def inject_heartbeat(self):
        """
        Add a heartbeat writer to the training process.
        Call this before tpu.run() — it creates a background script on the VM.
        """
        script = (
            f"while true; do date +%s > {self._heartbeat_path}; sleep {self.heartbeat_interval}; done"
        )
        self.tpu.ssh(f"nohup bash -c '{script}' > /dev/null 2>&1 &", timeout=10)

    def check_heartbeat(self):
        """Check if heartbeat is fresh. Returns (alive, age_seconds)."""
        try:
            ts = self.tpu.ssh(f"cat {self._heartbeat_path} 2>/dev/null", timeout=10)
            if not ts.strip():
                return False, -1
            age = time.time() - float(ts.strip())
            alive = age < self.heartbeat_interval * 3  # 3x tolerance
            return alive, age
        except Exception:
            return False, -1

    def check_disk(self):
        """Check disk usage. Returns (used_percent, used_gb, total_gb)."""
        try:
            out = self.tpu.ssh("df -BG / | tail -1 | awk '{print $3,$4,$5}'", timeout=10)
            parts = out.strip().split()
            if len(parts) >= 3:
                used = float(parts[0].rstrip('G'))
                avail = float(parts[1].rstrip('G'))
                pct = int(parts[2].rstrip('%'))
                return pct, used, used + avail
        except Exception:
            pass
        return -1, -1, -1

    def check_gpu_idle(self):
        """Check if GPUs are idle (TPU or NVIDIA). Returns (idle, utilization)."""
        try:
            out = self.tpu.ssh(
                "nvidia-smi --query-gpu=utilization.gpu --format=csv,nounits,noheader 2>/dev/null || echo -1",
                timeout=10,
            )
            utils = [int(x.strip()) for x in out.strip().split('\n') if x.strip().lstrip('-').isdigit()]
            if utils and utils[0] >= 0:
                avg = sum(utils) / len(utils)
                return avg < 5, avg
        except Exception:
            pass
        return False, -1

    def parse_latest_log(self, lines=5):
        """Parse metrics from the latest log lines."""
        try:
            log_path = f"{self.tpu.workdir}/{self.tpu.log_file}"
            out = self.tpu.ssh(f"tail -n {lines} {log_path} 2>/dev/null", timeout=10)
            all_metrics = {}
            for line in out.strip().split('\n'):
                m = parse_training_progress(line)
                if m:
                    all_metrics.update(m)
            return all_metrics
        except Exception:
            return {}

    def check(self):
        """Full health check. Returns status dict."""
        status = {"timestamp": time.time()}

        # Process alive
        status["process_alive"] = self.tpu.is_running()

        # Heartbeat
        alive, age = self.check_heartbeat()
        status["heartbeat_alive"] = alive
        status["heartbeat_age_s"] = round(age, 1) if age >= 0 else None

        # Disk
        pct, used, total = self.check_disk()
        status["disk_percent"] = pct
        status["disk_used_gb"] = used
        status["disk_total_gb"] = total
        status["disk_warning"] = pct > 80 if pct >= 0 else False

        # GPU idle
        idle, util = self.check_gpu_idle()
        status["gpu_idle"] = idle
        status["gpu_utilization"] = util

        # Training progress
        status["metrics"] = self.parse_latest_log()
        eta = estimate_eta(status["metrics"])
        if eta:
            status["eta_seconds"] = round(eta)
            status["eta_human"] = f"{eta/3600:.1f}h" if eta > 3600 else f"{eta/60:.0f}m"

        return status

    def check_pretty(self):
        """Print a formatted health report."""
        s = self.check()
        GREEN, RED, YELLOW, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[0m"

        def _c(ok):
            return GREEN if ok else RED

        print(f"\n  Health Check for '{self.tpu.name}'")
        print(f"  {'='*50}")
        print(f"  Process:   {_c(s['process_alive'])}{'running' if s['process_alive'] else 'stopped'}{RESET}")

        if s['heartbeat_age_s'] is not None:
            hb_ok = s['heartbeat_alive']
            print(f"  Heartbeat: {_c(hb_ok)}{'fresh' if hb_ok else 'stale'} ({s['heartbeat_age_s']:.0f}s ago){RESET}")

        if s['disk_percent'] >= 0:
            disk_ok = not s['disk_warning']
            print(f"  Disk:      {_c(disk_ok)}{s['disk_percent']}% ({s['disk_used_gb']:.0f}/{s['disk_total_gb']:.0f} GB){RESET}")

        if s['gpu_utilization'] >= 0:
            gpu_ok = not s['gpu_idle']
            print(f"  GPU:       {_c(gpu_ok)}{s['gpu_utilization']:.0f}% utilization{RESET}")
            if s['gpu_idle']:
                print(f"             {YELLOW}WARNING: GPU idle — wasting money{RESET}")

        m = s.get('metrics', {})
        if m:
            parts = []
            if 'step' in m:
                step_str = f"step {m['step']}"
                if 'total_steps' in m:
                    step_str += f"/{m['total_steps']}"
                parts.append(step_str)
            if 'loss' in m:
                parts.append(f"loss {m['loss']:.4f}")
            if 'tok_per_sec' in m:
                parts.append(f"{m['tok_per_sec']:,} tok/s")
            if parts:
                print(f"  Training:  {' | '.join(parts)}")

        if 'eta_human' in s:
            print(f"  ETA:       ~{s['eta_human']}")
        print()
        return s
