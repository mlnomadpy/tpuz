"""
GCE (Google Compute Engine) VM management.

For managing GPU VMs (A100, H100, L4, T4) alongside TPU VMs.
Wraps `gcloud compute instances` commands.

Usage:
    from tpuz.gce import GCE

    vm = GCE("my-gpu-vm", machine_type="a2-highgpu-1g", zone="us-central1-a")
    vm.up()
    vm.ssh("nvidia-smi")
    vm.run("python train.py", sync="./src")
    vm.logs()
    vm.down()
"""

import os
import json
import time
import subprocess
import threading
from dataclasses import dataclass


@dataclass
class VMInfo:
    name: str
    state: str          # RUNNING, TERMINATED, STAGING, etc.
    machine_type: str
    zone: str
    external_ip: str
    internal_ip: str


# Common GPU machine types
GPU_MACHINES = {
    "t4":       "n1-standard-8",     # + --accelerator=type=nvidia-tesla-t4,count=1
    "t4x2":     "n1-standard-16",    # + count=2
    "t4x4":     "n1-standard-32",    # + count=4
    "l4":       "g2-standard-8",     # L4 attached
    "l4x2":     "g2-standard-16",
    "a100":     "a2-highgpu-1g",     # 1x A100 40GB
    "a100x2":   "a2-highgpu-2g",
    "a100x4":   "a2-highgpu-4g",
    "a100x8":   "a2-megagpu-16g",    # 8x A100
    "a100-80":  "a2-ultragpu-1g",    # 1x A100 80GB
    "h100x8":   "a3-highgpu-8g",     # 8x H100
}


class GCE:
    """
    Manage a GCE GPU VM.

    Usage:
        vm = GCE("my-vm", machine_type="a2-highgpu-1g", zone="us-central1-a")
        vm.up()
        vm.ssh("nvidia-smi")
        vm.down()

    Shorthand for GPU types:
        vm = GCE.gpu("my-vm", gpu="a100", zone="us-central1-a")
    """

    def __init__(self, name, machine_type="n1-standard-8", zone="us-central1-a",
                 project=None, preemptible=True, gpu=None, gpu_count=1,
                 boot_disk_size="200GB", image_family="pytorch-latest-gpu",
                 image_project="deeplearning-platform-release"):
        self.name = name
        self.machine_type = machine_type
        self.zone = zone
        self.project = project or os.environ.get("GCLOUD_PROJECT", "")
        self.preemptible = preemptible
        self.gpu = gpu
        self.gpu_count = gpu_count
        self.boot_disk_size = boot_disk_size
        self.image_family = image_family
        self.image_project = image_project
        self.workdir = f"/home/{os.environ.get('USER', 'user')}/workdir"
        self.log_file = "train.log"

    @classmethod
    def gpu(cls, name, gpu="a100", zone="us-central1-a", **kwargs):
        """Create from a GPU shorthand: t4, l4, a100, a100x8, h100x8."""
        if gpu in GPU_MACHINES:
            machine_type = GPU_MACHINES[gpu]
            # Parse GPU type and count from shorthand
            gpu_map = {
                "t4": ("nvidia-tesla-t4", 1), "t4x2": ("nvidia-tesla-t4", 2), "t4x4": ("nvidia-tesla-t4", 4),
                "l4": ("nvidia-l4", 1), "l4x2": ("nvidia-l4", 2),
                "a100": ("nvidia-tesla-a100", 1), "a100x2": ("nvidia-tesla-a100", 2),
                "a100x4": ("nvidia-tesla-a100", 4), "a100x8": ("nvidia-tesla-a100", 8),
                "a100-80": ("nvidia-a100-80gb", 1),
                "h100x8": ("nvidia-h100-80gb", 8),
            }
            actual_gpu, count = gpu_map.get(gpu, (f"nvidia-{gpu}", 1))
            return cls(name, machine_type, zone, gpu=actual_gpu, gpu_count=count, **kwargs)
        return cls(name, gpu, zone, **kwargs)

    def _gcloud(self, args, timeout=300, check=True):
        cmd = ["gcloud"] + args
        if self.project:
            cmd += [f"--project={self.project}"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if check and result.returncode != 0:
            raise RuntimeError(f"gcloud error: {result.stderr.strip()[:200]}")
        return result

    # ----------------------------------------------------------------
    # Lifecycle
    # ----------------------------------------------------------------
    def up(self, wait=True):
        """Create GCE VM. Idempotent."""
        info = self.info()
        if info is not None:
            if info.state == "TERMINATED":
                print(f"VM '{self.name}' is terminated. Starting...")
                self.start()
                return self.info()
            print(f"VM '{self.name}' exists (state: {info.state})")
            return info

        print(f"Creating VM '{self.name}' ({self.machine_type}) in {self.zone}...")
        args = [
            "compute", "instances", "create", self.name,
            f"--zone={self.zone}",
            f"--machine-type={self.machine_type}",
            f"--boot-disk-size={self.boot_disk_size}",
            f"--image-family={self.image_family}",
            f"--image-project={self.image_project}",
            "--maintenance-policy=TERMINATE",
        ]
        if self.gpu:
            args += [f"--accelerator=type={self.gpu},count={self.gpu_count}"]
        if self.preemptible:
            args += ["--provisioning-model=SPOT", "--instance-termination-action=STOP"]

        self._gcloud(args, timeout=300)

        if wait:
            self._wait_running()

        info = self.info()
        print(f"VM '{self.name}' ready! IP: {info.external_ip}")
        return info

    def down(self):
        """Delete GCE VM."""
        print(f"Deleting VM '{self.name}'...")
        self._gcloud(
            ["compute", "instances", "delete", self.name,
             f"--zone={self.zone}", "--quiet"],
            check=False,
        )
        print("Deleted.")

    def stop(self):
        """Stop VM (keeps disk, stops billing for compute)."""
        self._gcloud(["compute", "instances", "stop", self.name, f"--zone={self.zone}"])
        print(f"VM '{self.name}' stopped.")

    def start(self):
        """Start a stopped VM."""
        self._gcloud(["compute", "instances", "start", self.name, f"--zone={self.zone}"])
        self._wait_running()
        print(f"VM '{self.name}' started.")

    def info(self):
        """Get VM info. Returns None if not found."""
        result = self._gcloud(
            ["compute", "instances", "describe", self.name,
             f"--zone={self.zone}", "--format=json"],
            check=False,
        )
        if result.returncode != 0:
            return None
        data = json.loads(result.stdout)
        state = data.get("status", "UNKNOWN")
        mt = data.get("machineType", "").split("/")[-1]
        ext_ip = ""
        int_ip = ""
        for iface in data.get("networkInterfaces", []):
            int_ip = iface.get("networkIP", "")
            for ac in iface.get("accessConfigs", []):
                ext_ip = ac.get("natIP", "")
        return VMInfo(self.name, state, mt, self.zone, ext_ip, int_ip)

    def _wait_running(self, timeout=300, poll=10):
        t0 = time.time()
        while time.time() - t0 < timeout:
            info = self.info()
            if info and info.state == "RUNNING":
                return
            time.sleep(poll)
        raise TimeoutError(f"VM not running after {timeout}s")

    # ----------------------------------------------------------------
    # SSH / SCP
    # ----------------------------------------------------------------
    def ssh(self, cmd, timeout=120):
        """Run a command on the VM."""
        args = [
            "compute", "ssh", self.name,
            f"--zone={self.zone}",
            "--command", cmd,
        ]
        result = self._gcloud(args, timeout=timeout)
        return result.stdout.strip()

    def scp_to(self, local, remote):
        """Copy file/dir to VM."""
        args = ["compute", "scp", local, f"{self.name}:{remote}", f"--zone={self.zone}"]
        if os.path.isdir(local):
            args.append("--recurse")
        self._gcloud(args, timeout=300)

    def scp_from(self, remote, local):
        """Copy file/dir from VM."""
        args = ["compute", "scp", f"{self.name}:{remote}", local, f"--zone={self.zone}"]
        self._gcloud(args, timeout=300)

    # ----------------------------------------------------------------
    # Training
    # ----------------------------------------------------------------
    def setup(self, extra_pip=""):
        """Install common ML deps."""
        print("Installing deps...")
        cmds = [
            "pip install -q jax[cuda12] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html",
            "pip install -q flax optax orbax-checkpoint datasets pyarrow pyyaml wandb",
        ]
        if extra_pip:
            cmds.append(f"pip install -q {extra_pip}")
        for cmd in cmds:
            print(f"  {cmd[:60]}...")
            self.ssh(cmd, timeout=300)
        print("Setup done!")

    def run(self, cmd, env=None, secrets=None, sync=None):
        """Launch training in detached mode."""
        if sync:
            print(f"Uploading {sync}...")
            self.scp_to(sync, self.workdir)

        if secrets:
            from tpuz.secrets import SecretManager
            load_cmd = SecretManager.load_env_command(secrets, self.project)
            self.ssh(f'echo "{load_cmd}" > {self.workdir}/.load_secrets.sh', timeout=10)

        env_str = ""
        if env:
            import tempfile
            env_content = "\n".join(f"export {k}={v}" for k, v in env.items())
            with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
                f.write(env_content)
                env_file = f.name
            self.scp_to(env_file, f"{self.workdir}/.env")
            os.unlink(env_file)
            env_str = f"source {self.workdir}/.env && "

        secrets_str = f"source {self.workdir}/.load_secrets.sh && " if secrets else ""
        log_path = f"{self.workdir}/{self.log_file}"
        launch = (
            f"mkdir -p {self.workdir} && cd {self.workdir} && "
            f"{env_str}{secrets_str}"
            f"nohup {cmd} > {log_path} 2>&1 & "
            f"echo $! > {self.workdir}/train.pid"
        )
        self.ssh(launch, timeout=30)
        print(f"Launched: {cmd}")

    def logs(self, lines=50, follow=True):
        """Stream training logs."""
        log_path = f"{self.workdir}/{self.log_file}"
        cmd = f"tail -n {lines}"
        if follow:
            cmd += " -f"
        cmd += f" {log_path}"

        if follow:
            args = [
                "gcloud", "compute", "ssh", self.name,
                f"--zone={self.zone}", "--command", cmd,
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

    def kill(self):
        """Kill training process."""
        try:
            self.ssh(f"kill $(cat {self.workdir}/train.pid) 2>/dev/null", timeout=10)
        except Exception:
            pass

    def is_running(self):
        """Check if training is alive."""
        try:
            pid = self.ssh(f"cat {self.workdir}/train.pid 2>/dev/null", timeout=10)
            alive = self.ssh(f"kill -0 {pid.strip()} 2>/dev/null && echo y || echo n", timeout=10)
            return alive.strip() == "y"
        except Exception:
            return False

    def tunnel(self, remote_port, local_port=None):
        """SSH tunnel for TensorBoard/Jupyter."""
        if local_port is None:
            local_port = remote_port
        print(f"Tunnel localhost:{local_port} → {self.name}:{remote_port}")
        args = [
            "gcloud", "compute", "ssh", self.name,
            f"--zone={self.zone}", "--",
            "-NL", f"{local_port}:localhost:{remote_port}",
        ]
        if self.project:
            args += [f"--project={self.project}"]
        try:
            subprocess.run(args)
        except KeyboardInterrupt:
            print("\nTunnel closed.")

    def collect(self, remote_files, local_dir="./outputs"):
        """Download artifacts."""
        os.makedirs(local_dir, exist_ok=True)
        for f in remote_files:
            remote = f if f.startswith("/") else f"{self.workdir}/{f}"
            local = os.path.join(local_dir, os.path.basename(f))
            try:
                self.scp_from(remote, local)
                print(f"  Downloaded: {local}")
            except Exception as e:
                print(f"  Failed: {f} ({e})")

    # ----------------------------------------------------------------
    # Static helpers
    # ----------------------------------------------------------------
    @staticmethod
    def list(zone="us-central1-a", project=None):
        """List all GCE VMs in a zone."""
        args = ["gcloud", "compute", "instances", "list", f"--zones={zone}", "--format=json"]
        if project:
            args += [f"--project={project}"]
        result = subprocess.run(args, capture_output=True, text=True)
        if result.returncode != 0:
            return []
        return json.loads(result.stdout)

    def __repr__(self):
        gpu_str = f", gpu={self.gpu}x{self.gpu_count}" if self.gpu else ""
        return f"GCE(name={self.name!r}, type={self.machine_type!r}{gpu_str})"
