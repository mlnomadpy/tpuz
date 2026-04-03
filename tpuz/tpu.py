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

    # ----------------------------------------------------------------
    # gcloud helper
    # ----------------------------------------------------------------
    def _gcloud(self, args, timeout=300, check=True):
        cmd = ["gcloud"] + args
        if self.project:
            cmd += [f"--project={self.project}"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if check and result.returncode != 0:
            raise RuntimeError(f"gcloud error: {result.stderr.strip()}")
        return result

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

    def ssh_all(self, cmd, timeout=120):
        """Run on all workers in parallel. Returns list of outputs."""
        if self.num_workers == 1:
            return [self.ssh(cmd, 0, timeout)]

        results = [None] * self.num_workers
        errors = []

        def _run(w):
            try:
                results[w] = self.ssh(cmd, w, timeout)
            except Exception as e:
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

        env_str = ""
        if env:
            env_str = " ".join(f"export {k}={v};" for k, v in env.items())

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
    def setup(self, extra_pip=""):
        """Install JAX[TPU] and common deps."""
        print("Installing deps...")
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

    def __repr__(self):
        return f"TPU(name={self.name!r}, accelerator={self.accelerator!r}, zone={self.zone!r})"
