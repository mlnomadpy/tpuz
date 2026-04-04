"""
TPU VM lifecycle management via gcloud CLI.

Handles creation, deletion, SSH, SCP, and multi-host coordination.
Zero dependencies — just calls gcloud.
"""

import os
import json
import time
import subprocess
import threading
from dataclasses import dataclass


# Runtime version defaults per TPU generation
_RUNTIMES = {
    "v5p":       "v2-alpha-tpuv5",
    "v5litepod": "v2-alpha-tpuv5-lite",
    "v6e":       "v2-alpha-tpuv6e",
    "v4":        "v2-alpha-tpuv4",
}


@dataclass
class TPUInfo:
    """Information about a TPU VM."""
    name: str
    state: str              # CREATING, READY, PREEMPTED, TERMINATED
    accelerator: str        # e.g. v4-8
    zone: str
    external_ips: list
    internal_ips: list
    preemptible: bool = False


class TPU:
    """
    Manage a GCP TPU VM.

    Usage:
        tpu = TPU("my-tpu", accelerator="v4-8", zone="us-central2-b")
        tpu.up()                          # Create
        tpu.ssh("echo hello")             # Remote command
        tpu.run("python train.py")        # Detached training
        tpu.logs()                        # Stream logs
        tpu.down()                        # Delete

    Multi-host (v4-32, v5litepod-64, etc.):
        tpu.ssh_all("echo hello")         # Runs on all workers in parallel
    """

    def __init__(
        self,
        name: str,
        accelerator: str = "v4-8",
        zone: str = "us-central2-b",
        project: str = None,
        preemptible: bool = True,
        runtime: str = None,
    ):
        self.name = name
        self.accelerator = accelerator
        self.zone = zone
        self.project = project or os.environ.get("GCLOUD_PROJECT", "")
        self.preemptible = preemptible
        self.runtime = runtime or self._detect_runtime(accelerator)
        self.num_workers = self._worker_count(accelerator)
        self.workdir = f"/home/{os.environ.get('USER', 'user')}/workdir"
        self.log_file = "train.log"

    @staticmethod
    def _detect_runtime(accelerator):
        for prefix, rt in _RUNTIMES.items():
            if accelerator.startswith(prefix):
                return rt
        return "tpu-vm-base"

    @staticmethod
    def _worker_count(accelerator):
        parts = accelerator.rsplit("-", 1)
        if len(parts) == 2 and parts[1].isdigit():
            chips = int(parts[1])
            if chips <= 8: return 1
            if chips <= 16: return 2
            return chips // 8
        return 1

    # Secrets to redact from logs
    _REDACT_PATTERNS = ["WANDB_API_KEY", "HF_TOKEN", "HUGGING_FACE", "GITHUB_TOKEN", "API_KEY"]

    # ----------------------------------------------------------------
    # gcloud helper
    # ----------------------------------------------------------------
    def _gcloud(self, args, timeout=300, check=True):
        cmd = ["gcloud"] + args
        if self.project:
            cmd += [f"--project={self.project}"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if check and result.returncode != 0:
            stderr = self._redact(result.stderr.strip())
            raise RuntimeError(f"gcloud error: {stderr}")
        return result

    @classmethod
    def _redact(cls, text):
        """Redact secrets from log output."""
        import re
        for pat in cls._REDACT_PATTERNS:
            text = re.sub(rf'{pat}=\S+', f'{pat}=***', text)
        return text

    def preflight(self):
        """Verify gcloud is configured correctly. Raises on failure."""
        result = self._gcloud(["config", "get-value", "account"], check=False)
        account = result.stdout.strip()
        if not account or account == "(unset)":
            raise RuntimeError("No gcloud account. Run: gcloud auth login")
        print(f"gcloud account: {account}")

        if self.project:
            print(f"gcloud project: {self.project}")
        else:
            result = self._gcloud(["config", "get-value", "project"], check=False)
            proj = result.stdout.strip()
            if not proj or proj == "(unset)":
                raise RuntimeError("No gcloud project. Run: gcloud config set project PROJECT")
            self.project = proj
            print(f"gcloud project: {proj}")

    @staticmethod
    def list_runtimes(zone="us-central2-b", project=None):
        """List available TPU runtime versions for a zone."""
        args = ["compute", "tpus", "versions", "list", f"--zone={zone}", "--format=json"]
        if project:
            args += [f"--project={project}"]
        result = subprocess.run(["gcloud"] + args, capture_output=True, text=True)
        if result.returncode != 0:
            return []
        return [v.get("name", "") for v in json.loads(result.stdout)]

    # ----------------------------------------------------------------
    # Lifecycle
    # ----------------------------------------------------------------
    def up(self, wait=True):
        """Create TPU VM. Idempotent — skips if exists."""
        info = self.info()
        if info is not None:
            print(f"TPU '{self.name}' exists (state: {info.state})")
            return info

        print(f"Creating TPU '{self.name}' ({self.accelerator}) in {self.zone}...")
        args = [
            "compute", "tpus", "tpu-vm", "create", self.name,
            f"--zone={self.zone}",
            f"--accelerator-type={self.accelerator}",
            f"--version={self.runtime}",
        ]
        if self.preemptible:
            args.append("--preemptible")
        self._gcloud(args, timeout=600)

        if wait:
            self._wait_ready()

        info = self.info()
        print(f"TPU '{self.name}' ready! IPs: {info.external_ips}")
        return info

    def up_queued(self, timeout_hours=2):
        """
        Create TPU VM via Queued Resources API.
        Automatically waits for capacity and provisions when available.
        Much more reliable than up() for spot/preemptible TPUs.
        """
        print(f"Queuing TPU '{self.name}' ({self.accelerator})...")
        qr_name = f"{self.name}-qr"

        args = [
            "compute", "tpus", "queued-resources", "create", qr_name,
            f"--zone={self.zone}",
            f"--accelerator-type={self.accelerator}",
            f"--runtime-version={self.runtime}",
            f"--node-id={self.name}",
        ]
        if self.preemptible:
            args.append("--spot")

        self._gcloud(args, timeout=60)
        print(f"Queued resource '{qr_name}' created. Waiting for capacity...")

        # Poll until ACTIVE
        timeout_s = timeout_hours * 3600
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            result = self._gcloud(
                ["compute", "tpus", "queued-resources", "describe", qr_name,
                 f"--zone={self.zone}", "--format=json"],
                check=False,
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                state = data.get("state", {}).get("state", "UNKNOWN")
                if state == "ACTIVE":
                    print(f"TPU '{self.name}' provisioned!")
                    self._wait_ready()
                    return self.info()
                elif state == "FAILED":
                    raise RuntimeError(f"Queued resource failed: {data}")
                print(f"  State: {state}")
            time.sleep(30)

        raise TimeoutError(f"Queued resource not ready after {timeout_hours}h")

    def down_queued(self):
        """Delete both queued resource and TPU VM."""
        qr_name = f"{self.name}-qr"
        self._gcloud(
            ["compute", "tpus", "queued-resources", "delete", qr_name,
             f"--zone={self.zone}", "--quiet", "--force"],
            check=False,
        )
        self.down()

    def down(self):
        """Delete TPU VM."""
        print(f"Deleting TPU '{self.name}'...")
        self._gcloud(
            ["compute", "tpus", "tpu-vm", "delete", self.name,
             f"--zone={self.zone}", "--quiet"],
            check=False,
        )
        print("Deleted.")

    def info(self):
        """Get VM info. Returns None if not found."""
        result = self._gcloud(
            ["compute", "tpus", "tpu-vm", "describe", self.name,
             f"--zone={self.zone}", "--format=json"],
            check=False,
        )
        if result.returncode != 0:
            return None

        data = json.loads(result.stdout)
        ext_ips, int_ips = [], []
        for ep in data.get("networkEndpoints", []):
            for ap in ep.get("accessConfig", []):
                if "externalIp" in ap:
                    ext_ips.append(ap["externalIp"])
            if "ipAddress" in ep:
                int_ips.append(ep["ipAddress"])

        return TPUInfo(
            name=self.name,
            state=data.get("state", "UNKNOWN"),
            accelerator=data.get("acceleratorType", ""),
            zone=self.zone,
            external_ips=ext_ips,
            internal_ips=int_ips,
            preemptible="PREEMPTIBLE" in data.get("schedulingConfig", {}).get("preemptible", ""),
        )

    def _wait_ready(self, timeout=600, poll=15):
        t0 = time.time()
        while time.time() - t0 < timeout:
            info = self.info()
            if info and info.state == "READY":
                return
            state = info.state if info else "CREATING"
            print(f"  Waiting... ({state})")
            time.sleep(poll)
        raise TimeoutError(f"TPU not ready after {timeout}s")

    # ----------------------------------------------------------------
    # SSH / SCP
    # ----------------------------------------------------------------
    def ssh(self, cmd, worker=0, timeout=120):
        """Run a command on the TPU VM."""
        args = [
            "compute", "tpus", "tpu-vm", "ssh", self.name,
            f"--zone={self.zone}", f"--worker={worker}",
            "--command", cmd,
        ]
        result = self._gcloud(args, timeout=timeout)
        return result.stdout.strip()

    def ssh_all(self, cmd, timeout=120, retries=3):
        """Run on all workers in parallel with per-worker retries."""
        if self.num_workers == 1:
            return [self.ssh(cmd, 0, timeout)]

        results = [None] * self.num_workers
        errors = []

        def _run(w):
            for attempt in range(retries):
                try:
                    results[w] = self.ssh(cmd, w, timeout)
                    return
                except Exception as e:
                    if attempt < retries - 1:
                        time.sleep(2)
                    else:
                        errors.append((w, str(e)))

        threads = [threading.Thread(target=_run, args=(w,)) for w in range(self.num_workers)]
        for t in threads: t.start()
        for t in threads: t.join(timeout + 30)

        if errors:
            print(f"Warning: errors on workers {[e[0] for e in errors]}")
        return results

    def scp_to(self, local, remote, worker=0):
        """Copy file/dir to VM."""
        args = [
            "compute", "tpus", "tpu-vm", "scp",
            local, f"{self.name}:{remote}",
            f"--zone={self.zone}", f"--worker={worker}",
        ]
        if os.path.isdir(local):
            args.append("--recurse")
        self._gcloud(args, timeout=300)

    def scp_from(self, remote, local, worker=0):
        """Copy file/dir from VM."""
        args = [
            "compute", "tpus", "tpu-vm", "scp",
            f"{self.name}:{remote}", local,
            f"--zone={self.zone}", f"--worker={worker}",
        ]
        self._gcloud(args, timeout=300)

    # ----------------------------------------------------------------
    # Verification & Config
    # ----------------------------------------------------------------
    def verify(self):
        """
        Verify all workers have JAX working with expected device count.
        Returns True if all workers report correct device count.
        """
        expected_per_worker = 4 if self.num_workers == 1 else 4  # each worker has 4 chips typically
        cmd = "python3 -c \"import jax; print(jax.device_count())\""
        outputs = self.ssh_all(cmd, timeout=30)

        all_ok = True
        for w, out in enumerate(outputs):
            if out is None:
                print(f"  worker {w}: UNREACHABLE")
                all_ok = False
            else:
                count = out.strip()
                print(f"  worker {w}: {count} devices")
                if not count.isdigit():
                    all_ok = False

        if all_ok:
            print("All workers verified!")
        else:
            print("WARNING: Some workers failed verification")
        return all_ok

    def upload_config(self, config_dict, remote_path=None):
        """
        Serialize config dict to JSON, upload to all workers.
        For multi-host: uploads to worker 0, then distributes via internal network.
        """
        import tempfile
        if remote_path is None:
            remote_path = f"{self.workdir}/config.json"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_dict, f, indent=2, default=str)
            local_path = f.name

        # Upload to worker 0
        self.scp_to(local_path, remote_path, worker=0)

        # Multi-host: copy from worker 0 to others via internal network
        if self.num_workers > 1:
            info = self.info()
            if info and info.internal_ips:
                w0_ip = info.internal_ips[0]
                for w in range(1, self.num_workers):
                    self.ssh(
                        f"scp -o StrictHostKeyChecking=no {w0_ip}:{remote_path} {remote_path}",
                        worker=w, timeout=30,
                    )
        os.unlink(local_path)
        print(f"Config uploaded to {self.num_workers} workers: {remote_path}")
        return remote_path

    def wait(self, poll=60, timeout_hours=24, complete_sentinel="COMPLETE", fail_sentinel="FAILED"):
        """
        Poll training log until sentinel found.
        Returns True on success, False on failure.
        Raises TimeoutError if neither found.
        """
        log_path = f"{self.workdir}/{self.log_file}"
        timeout_s = timeout_hours * 3600
        t0 = time.time()

        print(f"Waiting for training (timeout: {timeout_hours}h)...")
        while time.time() - t0 < timeout_s:
            try:
                tail = self.ssh(f"tail -5 {log_path} 2>/dev/null", timeout=15)
                if complete_sentinel in tail:
                    print("Training completed!")
                    return True
                if fail_sentinel in tail:
                    print("Training FAILED!")
                    return False
                # Show last line as progress
                last = tail.strip().split("\n")[-1] if tail.strip() else "..."
                elapsed = time.time() - t0
                print(f"  [{elapsed/60:.0f}m] {last[:80]}")
            except Exception as e:
                print(f"  Connection error: {e}")
            time.sleep(poll)

        raise TimeoutError(f"Training not done after {timeout_hours}h")

    # ----------------------------------------------------------------
    # Training
    # ----------------------------------------------------------------
    def run(self, cmd, env=None, sync=None):
        """
        Launch a command in detached mode (nohup).

        Args:
            cmd: Command to run (e.g. "python train.py")
            env: Dict of env vars to export
            sync: Local directory to upload before running
        """
        if sync:
            print(f"Uploading {sync} → {self.workdir}...")
            self.scp_to(sync, self.workdir, worker=0)
            if self.num_workers > 1:
                for w in range(1, self.num_workers):
                    self.scp_to(sync, self.workdir, worker=w)

        # Write secrets to a .env file instead of command line (avoids ps/history leak)
        env_str = ""
        if env:
            import tempfile
            env_content = "\n".join(f"export {k}={v}" for k, v in env.items())
            with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
                f.write(env_content)
                env_file = f.name
            env_remote = f"{self.workdir}/.env"
            self.scp_to(env_file, env_remote, worker=0)
            os.unlink(env_file)
            if self.num_workers > 1:
                for w in range(1, self.num_workers):
                    self.ssh(f"cp {env_remote} {env_remote}", worker=w, timeout=10)
            env_str = f"source {env_remote} && "

        log_path = f"{self.workdir}/{self.log_file}"
        launch = (
            f"cd {self.workdir} && {env_str} "
            f"nohup {cmd} > {log_path} 2>&1 & "
            f"echo $! > {self.workdir}/train.pid && "
            f"echo 'PID:' $(cat {self.workdir}/train.pid)"
        )

        print(f"Launching: {cmd}")
        if self.num_workers == 1:
            output = self.ssh(launch, timeout=30)
        else:
            output = self.ssh_all(launch, timeout=30)
        print(f"  {output}")

    def logs(self, lines=50, follow=True):
        """Stream training logs."""
        log_path = f"{self.workdir}/{self.log_file}"
        cmd = f"tail -n {lines}"
        if follow:
            cmd += " -f"
        cmd += f" {log_path}"

        if follow:
            args = [
                "gcloud", "compute", "tpus", "tpu-vm", "ssh", self.name,
                f"--zone={self.zone}", f"--worker=0", "--command", cmd,
            ]
            if self.project:
                args += [f"--project={self.project}"]
            try:
                proc = subprocess.Popen(args, stdout=subprocess.PIPE, text=True)
                for line in proc.stdout:
                    print(line, end="", flush=True)
            except KeyboardInterrupt:
                proc.terminate()
                print("\nDetached.")
        else:
            return self.ssh(cmd)

    def is_running(self):
        """Check if training process is alive."""
        try:
            pid = self.ssh(f"cat {self.workdir}/train.pid 2>/dev/null", timeout=10)
            alive = self.ssh(f"kill -0 {pid.strip()} 2>/dev/null && echo y || echo n", timeout=10)
            return alive.strip() == "y"
        except Exception:
            return False

    def kill(self):
        """Kill the training process."""
        try:
            self.ssh(f"kill $(cat {self.workdir}/train.pid) 2>/dev/null", timeout=10)
        except Exception:
            pass

    # ----------------------------------------------------------------
    # Setup
    # ----------------------------------------------------------------
    def setup(self, extra_pip="", python_version="3.11"):
        """
        Install JAX[TPU] and common deps.
        Detects Python version and installs if needed.
        """
        print("Setting up environment...")

        # Check Python version, install if needed
        print(f"  Checking Python {python_version}...")
        py_check = self.ssh(f"python{python_version} --version 2>/dev/null || echo MISSING", timeout=15)
        if "MISSING" in py_check:
            print(f"  Installing Python {python_version}...")
            self.ssh_all(
                f"sudo apt-get update -qq && sudo apt-get install -y -qq python{python_version} python{python_version}-venv python{python_version}-dev > /dev/null 2>&1 || "
                f"echo 'apt failed, trying pixi...' && curl -fsSL https://pixi.sh/install.sh | bash && export PATH=$HOME/.pixi/bin:$PATH && pixi global install python",
                timeout=300,
            )

        cmds = [
            "sudo apt-get update -qq && sudo apt-get install -y -qq python3-pip > /dev/null 2>&1",
            "pip install -q 'jax[tpu]' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html",
            "pip install -q flax optax orbax-checkpoint datasets pyarrow pyyaml wandb",
        ]
        if extra_pip:
            cmds.append(f"pip install -q {extra_pip}")
        for cmd in cmds:
            print(f"  {cmd[:60]}...")
            self.ssh_all(cmd, timeout=300)
        print("Setup done!")

    # ----------------------------------------------------------------
    # Preemption recovery
    # ----------------------------------------------------------------
    def watch(self, cmd, max_retries=5, poll=60):
        """
        Watch for preemption and auto-recover.
        Re-creates VM and restarts training (relies on checkpoint for resume).
        """
        for attempt in range(max_retries):
            print(f"Watching (attempt {attempt + 1}/{max_retries})...")
            while True:
                time.sleep(poll)
                info = self.info()

                if info is None or info.state in ("PREEMPTED", "TERMINATED"):
                    print(f"Preempted! Recovering...")
                    try: self.down()
                    except: pass
                    time.sleep(10)
                    self.up()
                    self.setup()
                    self.run(cmd)
                    break

                if not self.is_running():
                    # Check if finished or crashed
                    log = self.ssh(f"tail -3 {self.workdir}/{self.log_file} 2>/dev/null", timeout=10)
                    if "COMPLETE" in log:
                        print("Training completed!")
                        return True
                    print("Process died. Restarting...")
                    self.run(cmd)
                    break

        print(f"Max retries ({max_retries}) exceeded.")
        return False

    # ----------------------------------------------------------------
    # Debugging & Development
    # ----------------------------------------------------------------
    def repl(self, setup_cmd=None):
        """
        Open an interactive Python REPL on worker 0.
        For multi-host: other workers wait at a barrier.

        Args:
            setup_cmd: Optional setup command to run before REPL
                       (e.g. "cd /home/user/project && source venv/bin/activate")
        """
        if self.num_workers > 1:
            # Launch a barrier wait on workers 1..N
            # They'll block until worker 0 sends a signal file
            barrier_cmd = (
                f"while [ ! -f {self.workdir}/.repl_done ]; do sleep 2; done; "
                f"rm -f {self.workdir}/.repl_done"
            )
            print(f"Starting barrier on {self.num_workers - 1} workers...")
            for w in range(1, self.num_workers):
                # Launch barrier in background on each worker
                self.ssh(f"nohup bash -c '{barrier_cmd}' &", worker=w, timeout=10)

        prefix = f"{setup_cmd} && " if setup_cmd else ""

        # Interactive SSH to worker 0 (not captured — goes directly to terminal)
        args = [
            "gcloud", "compute", "tpus", "tpu-vm", "ssh", self.name,
            f"--zone={self.zone}", "--worker=0",
            "--command", f"{prefix}python3",
        ]
        if self.project:
            args += [f"--project={self.project}"]

        print(f"Opening REPL on {self.name} worker 0 (Ctrl-D to exit)...")
        try:
            subprocess.run(args)
        finally:
            if self.num_workers > 1:
                # Signal other workers to resume
                self.ssh(f"touch {self.workdir}/.repl_done", worker=0, timeout=10)
                print("Released barrier on other workers.")

    def debug(self, cmd, port=5678):
        """
        Launch training with debugpy attached, print VS Code connect URL.

        Args:
            cmd: Training command (e.g. "python train.py")
            port: Debug port (default 5678)
        """
        # Install debugpy if needed
        self.ssh(f"pip install -q debugpy", timeout=60)

        # Wrap command with debugpy
        debug_cmd = f"python3 -m debugpy --listen 0.0.0.0:{port} --wait-for-client -m {cmd.replace('python ', '').replace('python3 ', '')}"

        # Get external IP
        info = self.info()
        ip = info.external_ips[0] if info and info.external_ips else "UNKNOWN"

        log_path = f"{self.workdir}/{self.log_file}"
        launch = (
            f"cd {self.workdir} && "
            f"nohup {debug_cmd} > {log_path} 2>&1 & "
            f"echo $! > {self.workdir}/train.pid"
        )
        self.ssh(launch, timeout=30)

        print(f"Debugger listening on {ip}:{port}")
        print(f"VS Code: Add to launch.json:")
        print(f'  {{"type": "debugpy", "request": "attach", "connect": {{"host": "{ip}", "port": {port}}}}}')
        print(f"Or: python -m debugpy --connect {ip}:{port}")

    def logs_all(self, lines=20):
        """
        Show logs from ALL workers side by side with color-coded prefixes.
        Useful for debugging multi-host issues.
        """
        COLORS = [
            "\033[32m",  # green
            "\033[33m",  # yellow
            "\033[34m",  # blue
            "\033[35m",  # magenta
            "\033[36m",  # cyan
            "\033[91m",  # bright red
            "\033[92m",  # bright green
            "\033[93m",  # bright yellow
        ]
        RESET = "\033[0m"

        log_path = f"{self.workdir}/{self.log_file}"
        outputs = self.ssh_all(f"tail -n {lines} {log_path} 2>/dev/null || echo '(no log)'", timeout=30)

        for w, output in enumerate(outputs):
            if output is None:
                output = "(no response)"
            color = COLORS[w % len(COLORS)]
            print(f"\n{color}{'=' * 60}")
            print(f"  Worker {w}")
            print(f"{'=' * 60}{RESET}")
            for line in output.strip().split("\n"):
                print(f"{color}[w{w}]{RESET} {line}")

    def health(self):
        """
        Health check across all workers.
        Returns list of worker status dicts.
        """
        log_path = f"{self.workdir}/{self.log_file}"

        # Check PID + last log line on all workers in parallel
        check_cmd = (
            f"PID=$(cat {self.workdir}/train.pid 2>/dev/null); "
            f"ALIVE=$(kill -0 $PID 2>/dev/null && echo y || echo n); "
            f"LAST=$(tail -1 {log_path} 2>/dev/null || echo ''); "
            f'echo "${{ALIVE}}|${{LAST}}"'
        )
        outputs = self.ssh_all(check_cmd, timeout=15)

        workers = []
        for w, out in enumerate(outputs):
            if out is None:
                workers.append({"worker": w, "status": "unreachable"})
                continue
            parts = out.strip().split("|", 1)
            alive = parts[0] == "y" if parts else False
            last_log = parts[1] if len(parts) > 1 else ""
            workers.append({
                "worker": w,
                "status": "running" if alive else "stopped",
                "last_log": last_log[:100],
            })
        return workers

    def health_pretty(self):
        """Print a pretty health dashboard for all workers."""
        COLORS = {"running": "\033[32m", "stopped": "\033[31m", "unreachable": "\033[33m"}
        RESET = "\033[0m"

        workers = self.health()
        print(f"\n  {'Worker':<10} {'Status':<15} {'Last Log'}")
        print(f"  {'-'*60}")
        for w in workers:
            color = COLORS.get(w["status"], "")
            log = w.get("last_log", "")[:50]
            wname = f"worker {w['worker']}"
            print(f"  {wname:<10} {color}{w['status']:<15}{RESET} {log}")
        print()

    def scale(self, new_accelerator):
        """
        Scale to a different accelerator type.
        Deletes current VM and creates a new one.
        Code on VM is lost — use with sync= in run() to re-upload.

        Args:
            new_accelerator: e.g. "v4-32" to scale up from "v4-8"
        """
        old = self.accelerator
        print(f"Scaling {self.name}: {old} → {new_accelerator}")
        self.down()
        time.sleep(5)
        self.accelerator = new_accelerator
        self.runtime = self._detect_runtime(new_accelerator)
        self.num_workers = self._worker_count(new_accelerator)
        self.up()
        self.setup()
        print(f"Scaled to {new_accelerator} ({self.num_workers} workers)")

    # ----------------------------------------------------------------
    # Static helpers
    # ----------------------------------------------------------------
    @staticmethod
    def list(zone="us-central2-b", project=None):
        """List all TPU VMs in a zone."""
        args = ["compute", "tpus", "tpu-vm", "list", f"--zone={zone}", "--format=json"]
        if project:
            args += [f"--project={project}"]
        result = subprocess.run(["gcloud"] + args, capture_output=True, text=True)
        if result.returncode != 0:
            return []
        return json.loads(result.stdout)

    # ----------------------------------------------------------------
    # Cost tracking
    # ----------------------------------------------------------------
    def cost(self):
        """Get cost tracker for this TPU. Starts counting from first up()."""
        from tpuz.costs import CostTracker
        if not hasattr(self, '_cost_tracker'):
            self._cost_tracker = CostTracker(self.accelerator, self.preemptible)
        return self._cost_tracker

    def cost_summary(self):
        """Print cost summary."""
        c = self.cost()
        print(c.summary())
        return c.cost

    # ----------------------------------------------------------------
    # SSH tunnel / port forwarding
    # ----------------------------------------------------------------
    def tunnel(self, remote_port, local_port=None, worker=0):
        """
        Open an SSH tunnel for TensorBoard, Jupyter, debugpy, etc.
        Blocks until Ctrl-C.

        Args:
            remote_port: Port on the TPU VM
            local_port: Local port (default: same as remote)
            worker: Which worker to tunnel to
        """
        if local_port is None:
            local_port = remote_port

        print(f"Tunneling localhost:{local_port} → {self.name}:{remote_port} (worker {worker})")
        print("Press Ctrl-C to close")

        args = [
            "gcloud", "compute", "tpus", "tpu-vm", "ssh", self.name,
            f"--zone={self.zone}", f"--worker={worker}",
            "--", f"-NL", f"{local_port}:localhost:{remote_port}",
        ]
        if self.project:
            args += [f"--project={self.project}"]

        try:
            subprocess.run(args)
        except KeyboardInterrupt:
            print("\nTunnel closed.")

    # ----------------------------------------------------------------
    # Availability check
    # ----------------------------------------------------------------
    @staticmethod
    def availability(accelerator, zone="us-central2-b", project=None):
        """
        Check if a TPU type is available in a zone.
        Returns dict with availability info.
        """
        from tpuz.costs import hourly_rate
        args = [
            "gcloud", "compute", "tpus", "accelerator-types", "describe",
            accelerator, f"--zone={zone}", "--format=json",
        ]
        if project:
            args += [f"--project={project}"]
        result = subprocess.run(args, capture_output=True, text=True, timeout=30)

        rate = hourly_rate(accelerator, preemptible=True)
        if result.returncode != 0:
            return {"available": False, "accelerator": accelerator, "zone": zone,
                    "spot_rate": rate, "error": result.stderr.strip()[:100]}

        return {"available": True, "accelerator": accelerator, "zone": zone,
                "spot_rate": rate, "on_demand_rate": rate * 3}

    # ----------------------------------------------------------------
    # Multi-zone failover
    # ----------------------------------------------------------------
    @classmethod
    def create_multi_zone(cls, name, accelerator, zones, project=None, preemptible=True):
        """
        Try creating in each zone until one succeeds.
        Returns (TPU, zone_used) or raises if all fail.
        """
        for zone in zones:
            print(f"Trying {zone}...")
            tpu = cls(name, accelerator, zone, project, preemptible)
            try:
                tpu.up()
                print(f"Created in {zone}!")
                return tpu
            except RuntimeError as e:
                print(f"  {zone} failed: {e}")
                continue
        raise RuntimeError(f"No capacity in any zone: {zones}")

    # ----------------------------------------------------------------
    # Checkpoint-aware restart
    # ----------------------------------------------------------------
    def run_with_resume(self, cmd, gcs=None, run_name=None, resume_flag="--resume-from-step", **kwargs):
        """
        Launch training with auto-detected checkpoint resume.
        Finds latest step in GCS and appends resume flag.

        Args:
            cmd: Base training command
            gcs: GCS instance for checkpoint lookup
            run_name: GCS run name (default: self.name)
            resume_flag: CLI flag for resume step
        """
        run_name = run_name or self.name

        if gcs:
            step = gcs.latest_step(run_name)
            if step is not None:
                cmd = f"{cmd} {resume_flag}={step}"
                print(f"Resuming from step {step}")
            else:
                print("No checkpoint found, starting fresh")

        self.run(cmd, **kwargs)

    # ----------------------------------------------------------------
    # Artifact collection
    # ----------------------------------------------------------------
    def collect(self, remote_files, local_dir="./outputs"):
        """
        Download specific files from the VM after training.

        Args:
            remote_files: List of remote paths (relative to workdir)
            local_dir: Local directory to save to
        """
        import os
        os.makedirs(local_dir, exist_ok=True)
        collected = []
        for f in remote_files:
            remote = f if f.startswith("/") else f"{self.workdir}/{f}"
            local = os.path.join(local_dir, os.path.basename(f))
            try:
                self.scp_from(remote, local)
                collected.append(local)
                print(f"  Downloaded: {local}")
            except Exception as e:
                print(f"  Failed: {f} ({e})")
        return collected

    # ----------------------------------------------------------------
    # Environment snapshot / restore
    # ----------------------------------------------------------------
    def snapshot_env(self, gcs=None):
        """Save pip freeze to workdir (and optionally GCS)."""
        self.ssh(f"pip freeze > {self.workdir}/requirements.txt", timeout=30)
        print("Environment snapshot saved to requirements.txt")
        if gcs:
            self.scp_from(f"{self.workdir}/requirements.txt", "/tmp/tpuz_reqs.txt")
            gcs.upload("/tmp/tpuz_reqs.txt", f"envs/{self.name}-requirements.txt")
            print(f"  Uploaded to GCS: envs/{self.name}-requirements.txt")

    def restore_env(self, gcs=None):
        """Restore pip packages from snapshot."""
        if gcs and gcs.exists(f"envs/{self.name}-requirements.txt"):
            gcs.download(f"envs/{self.name}-requirements.txt", "/tmp/")
            self.scp_to("/tmp/tpuz_reqs.txt", f"{self.workdir}/requirements.txt")

        self.ssh_all(
            f"pip install -q -r {self.workdir}/requirements.txt",
            timeout=300,
        )
        print("Environment restored from requirements.txt")

    # ----------------------------------------------------------------
    # Notifications
    # ----------------------------------------------------------------
    def watch_notify(self, cmd, notify_url=None, max_retries=5, poll=60):
        """
        Like watch() but sends notifications on preemption/completion.

        Args:
            cmd: Training command
            notify_url: Slack webhook or generic webhook URL
            max_retries: Max preemption recoveries
            poll: Poll interval in seconds
        """
        from tpuz.notify import notify

        for attempt in range(max_retries):
            print(f"Watching (attempt {attempt + 1}/{max_retries})...")
            while True:
                time.sleep(poll)
                info = self.info()

                if info is None or info.state in ("PREEMPTED", "TERMINATED"):
                    msg = f"TPU '{self.name}' preempted! Recovering (attempt {attempt + 1})..."
                    print(msg)
                    notify(notify_url, msg)
                    try: self.down()
                    except: pass
                    time.sleep(10)
                    self.up()
                    self.setup()
                    self.run(cmd)
                    break

                if not self.is_running():
                    log = self.ssh(f"tail -3 {self.workdir}/{self.log_file} 2>/dev/null", timeout=10)
                    if "COMPLETE" in log:
                        msg = f"Training on '{self.name}' completed!"
                        print(msg)
                        notify(notify_url, msg)
                        return True
                    msg = f"Training on '{self.name}' died. Restarting..."
                    print(msg)
                    notify(notify_url, msg)
                    self.run(cmd)
                    break

        msg = f"TPU '{self.name}' max retries ({max_retries}) exceeded."
        print(msg)
        notify(notify_url, msg)
        return False

    # ----------------------------------------------------------------
    # Run-once (Docker-like)
    # ----------------------------------------------------------------
    def run_once(self, cmd, sync=None, collect_files=None, gcs=None, notify_url=None, **kwargs):
        """
        Complete lifecycle: up → setup → run → wait → collect → down.
        Like `docker run` but for TPUs.

        Args:
            cmd: Training command
            sync: Local dir to upload
            collect_files: List of files to download after training
            gcs: GCS for checkpoint persistence
            notify_url: Notification webhook
        """
        from tpuz.notify import notify
        cost = self.cost()

        try:
            self.up()
            cost.start()
            self.setup()

            if gcs:
                self.run_with_resume(cmd, gcs=gcs, sync=sync, **kwargs)
            else:
                self.run(cmd, sync=sync, **kwargs)

            success = self.wait()

            if collect_files:
                self.collect(collect_files)

            cost.stop()
            msg = f"Training {'completed' if success else 'failed'} on '{self.name}'. {cost.summary()}"
            print(msg)
            notify(notify_url, msg)
            return success

        finally:
            cost.stop()
            self.down()

    # ----------------------------------------------------------------
    # Scheduled training
    # ----------------------------------------------------------------
    def schedule(self, cmd, start_after=None, max_cost=None, **kwargs):
        """
        Schedule training to start later or within a cost budget.

        Args:
            cmd: Training command
            start_after: Time string "HH:MM" to wait until (e.g. "22:00")
            max_cost: Max USD to spend (kills training when exceeded)
        """
        import datetime

        if start_after:
            now = datetime.datetime.now()
            h, m = map(int, start_after.split(":"))
            target = now.replace(hour=h, minute=m, second=0)
            if target <= now:
                target += datetime.timedelta(days=1)
            wait_secs = (target - now).total_seconds()
            print(f"Scheduled: waiting until {start_after} ({wait_secs/3600:.1f}h)...")
            time.sleep(wait_secs)

        self.up()
        cost = self.cost()
        cost.start()
        self.setup()
        self.run(cmd, **kwargs)

        # Monitor cost
        if max_cost:
            print(f"Cost limit: ${max_cost:.2f}")
            while self.is_running():
                time.sleep(60)
                if cost.cost >= max_cost:
                    print(f"Cost limit reached: {cost.summary()}")
                    self.kill()
                    break
        else:
            self.wait()

        cost.stop()
        print(f"Final: {cost.summary()}")
        return cost.cost

    def __repr__(self):
        return f"TPU(name={self.name!r}, accelerator={self.accelerator!r}, zone={self.zone!r})"
