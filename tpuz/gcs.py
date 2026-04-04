"""
GCS (Google Cloud Storage) integration for checkpoint persistence.

Wraps gsutil for upload/download/sync of training checkpoints.
Zero deps — calls gsutil CLI.
"""

import os
import subprocess
import json
import re


class GCS:
    """
    GCS checkpoint sync for TPU training.

    Usage:
        gcs = GCS("gs://my-bucket/training")
        gcs.upload("./checkpoints", "run-01/step-1000")
        gcs.download("run-01/step-1000", "./checkpoints")
        step = gcs.latest_step("run-01")
    """

    def __init__(self, bucket):
        """
        Args:
            bucket: GCS bucket path (e.g. "gs://my-bucket" or "gs://my-bucket/subdir")
        """
        self.bucket = bucket.rstrip("/")

    def path(self, *parts):
        """Join path under bucket root."""
        return "/".join([self.bucket] + [p.strip("/") for p in parts if p])

    def _gsutil(self, args, timeout=600, check=True):
        cmd = ["gsutil"] + args
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if check and result.returncode != 0:
            raise RuntimeError(f"gsutil error: {result.stderr.strip()}")
        return result

    # ----------------------------------------------------------------
    # Core operations
    # ----------------------------------------------------------------
    def upload(self, local, remote_path="", recursive=True):
        """Upload file/dir to GCS."""
        dest = self.path(remote_path)
        args = ["-m", "cp"]
        if recursive and os.path.isdir(local):
            args.append("-r")
        args += [local, dest]
        self._gsutil(args)
        return dest

    def download(self, remote_path, local, recursive=True):
        """Download from GCS to local."""
        src = self.path(remote_path)
        os.makedirs(local, exist_ok=True)
        args = ["-m", "cp"]
        if recursive:
            args.append("-r")
        args += [src, local]
        self._gsutil(args)
        return local

    def sync(self, local_dir, remote_path="", delete=False):
        """Rsync-style sync local → GCS."""
        dest = self.path(remote_path)
        args = ["-m", "rsync", "-r"]
        if delete:
            args.append("-d")
        args += [local_dir, dest]
        self._gsutil(args)

    def ls(self, remote_path=""):
        """List objects at a GCS path."""
        src = self.path(remote_path)
        result = self._gsutil(["ls", src], check=False)
        if result.returncode != 0:
            return []
        return [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]

    def exists(self, remote_path):
        """Check if a GCS object/prefix exists."""
        result = self._gsutil(["ls", self.path(remote_path)], check=False)
        return result.returncode == 0

    # ----------------------------------------------------------------
    # Checkpoint operations
    # ----------------------------------------------------------------
    def upload_checkpoint(self, local_dir, run_name, step):
        """
        Upload a checkpoint directory to GCS.

        Stored as: gs://bucket/checkpoints/{run_name}/step-{step:06d}/
        """
        remote = f"checkpoints/{run_name}/step-{step:06d}"
        dest = self.upload(local_dir, remote)
        print(f"Checkpoint uploaded: {dest}")
        return dest

    def download_checkpoint(self, run_name, step, local_dir="./checkpoints"):
        """Download a specific checkpoint step."""
        remote = f"checkpoints/{run_name}/step-{step:06d}"
        if not self.exists(remote):
            raise FileNotFoundError(f"Checkpoint not found: {self.path(remote)}")
        local = os.path.join(local_dir, f"step-{step:06d}")
        self.download(remote, local)
        print(f"Checkpoint downloaded: {local}")
        return local

    def latest_step(self, run_name):
        """
        Find the highest checkpoint step number for a run.
        Returns step number (int) or None if no checkpoints.
        """
        remote = f"checkpoints/{run_name}/"
        items = self.ls(remote)
        steps = []
        for item in items:
            match = re.search(r"step-(\d+)", item)
            if match:
                steps.append(int(match.group(1)))
        return max(steps) if steps else None

    def checkpoint_path(self, run_name):
        """Return GCS path root for Orbax to use directly."""
        return self.path(f"checkpoints/{run_name}")

    def list_runs(self):
        """List all training runs with checkpoints."""
        items = self.ls("checkpoints/")
        runs = []
        for item in items:
            name = item.rstrip("/").split("/")[-1]
            if name:
                runs.append(name)
        return runs

    def __repr__(self):
        return f"GCS({self.bucket!r})"
