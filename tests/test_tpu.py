"""Unit tests for TPU class — no GCP access needed."""

import subprocess
import sys
import pytest
from tpuz.tpu import TPU, TPUInfo


class TestTPUInit:
    def test_defaults(self):
        t = TPU("test", accelerator="v4-8")
        assert t.name == "test"
        assert t.accelerator == "v4-8"
        assert t.num_workers == 1

    def test_runtime_detection(self):
        assert TPU._detect_runtime("v4-8") == "v2-alpha-tpuv4"
        assert TPU._detect_runtime("v5litepod-8") == "v2-alpha-tpuv5-lite"
        assert TPU._detect_runtime("v6e-8") == "v2-alpha-tpuv6e"
        assert TPU._detect_runtime("v5p-8") == "v2-alpha-tpuv5"
        assert TPU._detect_runtime("unknown-8") == "tpu-vm-base"

    def test_worker_count(self):
        assert TPU._worker_count("v4-8") == 1
        assert TPU._worker_count("v4-16") == 2
        assert TPU._worker_count("v4-32") == 4
        assert TPU._worker_count("v5litepod-64") == 8
        assert TPU._worker_count("v6e-128") == 16
        assert TPU._worker_count("unknown") == 1

    def test_repr(self):
        r = repr(TPU("my-tpu", "v4-8", "us-central2-b"))
        assert "my-tpu" in r and "v4-8" in r

    def test_zone(self):
        assert TPU("x", zone="europe-west4-a").zone == "europe-west4-a"

    def test_preemptible_default(self):
        assert TPU("x").preemptible is True

    def test_custom_runtime(self):
        assert TPU("x", runtime="custom-v1").runtime == "custom-v1"


class TestTPUInfo:
    def test_fields(self):
        info = TPUInfo("test", "READY", "v4-8", "us-central2-b", ["1.2.3.4"], ["10.0.0.1"])
        assert info.state == "READY"
        assert info.external_ips == ["1.2.3.4"]
        assert info.preemptible is False


class TestScale:
    def test_updates_fields(self):
        t = TPU("test", accelerator="v4-8")
        t.accelerator = "v4-32"
        t.num_workers = t._worker_count("v4-32")
        assert t.num_workers == 4

    def test_scale_generations(self):
        t = TPU("test")
        t.accelerator = "v5litepod-64"
        t.runtime = t._detect_runtime("v5litepod-64")
        assert t.runtime == "v2-alpha-tpuv5-lite"
        assert t._worker_count("v5litepod-64") == 8


class TestCLI:
    def _cli(self, *a):
        return subprocess.run([sys.executable, "-m", "tpuz.cli"] + list(a),
                              capture_output=True, text=True)

    def test_help(self):
        assert self._cli("--help").returncode == 0

    def test_up_help(self):
        assert self._cli("up", "--help").returncode == 0

    def test_train_help(self):
        r = self._cli("train", "--help")
        assert r.returncode == 0 and "recover" in r.stdout

    def test_repl_help(self):
        assert self._cli("repl", "--help").returncode == 0

    def test_debug_help(self):
        r = self._cli("debug", "--help")
        assert r.returncode == 0 and "port" in r.stdout

    def test_health_help(self):
        assert self._cli("health", "--help").returncode == 0

    def test_scale_help(self):
        assert self._cli("scale", "--help").returncode == 0

    def test_logs_all_help(self):
        assert self._cli("logs-all", "--help").returncode == 0

    def test_unknown_command(self):
        assert self._cli("nonexistent").returncode != 0


class TestCostTracker:
    def test_basic(self):
        from tpuz.costs import CostTracker, hourly_rate
        ct = CostTracker("v4-8", preemptible=True)
        assert ct.rate == hourly_rate("v4-8", True)
        assert ct.cost == 0.0

    def test_elapsed(self):
        from tpuz.costs import CostTracker
        import time
        ct = CostTracker("v4-8")
        ct.start()
        time.sleep(0.1)
        ct.stop()
        assert ct.elapsed_hours > 0
        assert ct.cost > 0

    def test_summary(self):
        from tpuz.costs import CostTracker
        ct = CostTracker("v4-8")
        s = ct.summary()
        assert "$" in s
        assert "v4-8" in s

    def test_hourly_rates(self):
        from tpuz.costs import hourly_rate
        assert hourly_rate("v4-8", True) > 0
        assert hourly_rate("v6e-8", True) > 0
        assert hourly_rate("nonexistent", True) == 0.0
        assert hourly_rate("v4-8", False) > hourly_rate("v4-8", True)


class TestGCS:
    def test_path(self):
        from tpuz.gcs import GCS
        gcs = GCS("gs://my-bucket")
        assert gcs.path("a", "b") == "gs://my-bucket/a/b"
        assert gcs.path() == "gs://my-bucket"

    def test_checkpoint_path(self):
        from tpuz.gcs import GCS
        gcs = GCS("gs://bucket")
        assert gcs.checkpoint_path("run1") == "gs://bucket/checkpoints/run1"

    def test_repr(self):
        from tpuz.gcs import GCS
        assert "gs://bucket" in repr(GCS("gs://bucket"))


class TestNewCLI:
    def _cli(self, *a):
        return subprocess.run([sys.executable, "-m", "tpuz.cli"] + list(a),
                              capture_output=True, text=True)

    def test_cost_help(self):
        assert self._cli("cost", "--help").returncode == 0

    def test_tunnel_help(self):
        assert self._cli("tunnel", "--help").returncode == 0

    def test_avail_help(self):
        assert self._cli("avail", "--help").returncode == 0

    def test_collect_help(self):
        assert self._cli("collect", "--help").returncode == 0

    def test_run_once_help(self):
        r = self._cli("run-once", "--help")
        assert r.returncode == 0
        assert "notify" in r.stdout

    def test_verify_help(self):
        assert self._cli("verify", "--help").returncode == 0

    def test_wait_help(self):
        assert self._cli("wait", "--help").returncode == 0


class TestSecretManager:
    def test_load_env_command(self):
        from tpuz.secrets import SecretManager
        cmd = SecretManager.load_env_command(["WANDB_API_KEY", "HF_TOKEN"], project="my-proj")
        assert "WANDB_API_KEY" in cmd
        assert "HF_TOKEN" in cmd
        assert "gcloud secrets versions access" in cmd
        assert "--project=my-proj" in cmd

    def test_load_env_command_no_project(self):
        from tpuz.secrets import SecretManager
        cmd = SecretManager.load_env_command(["KEY"])
        assert "KEY" in cmd
        assert "--project" not in cmd


class TestGCE:
    def test_init(self):
        from tpuz.gce import GCE
        vm = GCE("test", machine_type="n1-standard-8", zone="us-central1-a")
        assert vm.name == "test"
        assert vm.machine_type == "n1-standard-8"

    def test_gpu_shorthand(self):
        from tpuz.gce import GCE
        vm = GCE.gpu("test", gpu="a100")
        assert "a100" in vm.gpu
        assert vm.gpu_count == 1

    def test_gpu_multi(self):
        from tpuz.gce import GCE
        vm = GCE.gpu("test", gpu="a100x4")
        assert vm.gpu_count == 4

    def test_repr(self):
        from tpuz.gce import GCE
        vm = GCE.gpu("test", gpu="a100")
        r = repr(vm)
        assert "test" in r
        assert "gpu" in r

    def test_preemptible_default(self):
        from tpuz.gce import GCE
        assert GCE("x").preemptible is True


class TestHealthParser:
    def test_parse_step_loss(self):
        from tpuz.health import parse_training_progress
        m = parse_training_progress("step 100 | loss 3.71 | dt 0.55s | tok/s 56,000")
        assert m["step"] == 100
        assert m["loss"] == 3.71
        assert m["tok_per_sec"] == 56000

    def test_parse_total_steps(self):
        from tpuz.health import parse_training_progress
        m = parse_training_progress("step 100/5000 (2.0%) | loss 3.71")
        assert m["step"] == 100
        assert m["total_steps"] == 5000
        assert m["percent"] == 2.0

    def test_parse_lr(self):
        from tpuz.health import parse_training_progress
        m = parse_training_progress("lr=3e-4 mfu=45.2")
        assert m["lr"] == 3e-4
        assert m["mfu"] == 45.2

    def test_parse_empty(self):
        from tpuz.health import parse_training_progress
        assert parse_training_progress("nothing here") == {}

    def test_eta(self):
        from tpuz.health import estimate_eta
        eta = estimate_eta({"step": 100, "total_steps": 1000, "dt": 0.5})
        assert eta == 450.0  # 900 steps * 0.5s


class TestProfiles:
    def test_save_load(self):
        import tempfile, os
        from tpuz import profiles
        old_dir = profiles.PROFILE_DIR
        profiles.PROFILE_DIR = tempfile.mkdtemp()
        try:
            profiles.save_profile("test", {"accelerator": "v4-8", "zone": "us-central2-b"})
            loaded = profiles.load_profile("test")
            assert loaded["accelerator"] == "v4-8"
            assert "test" in [p["name"] for p in profiles.list_profiles()]
            profiles.delete_profile("test")
            assert profiles.load_profile("test") is None
        finally:
            profiles.PROFILE_DIR = old_dir

    def test_load_missing(self):
        from tpuz.profiles import load_profile
        assert load_profile("nonexistent_profile_xyz") is None


class TestAudit:
    def test_log_and_read(self):
        import tempfile
        from tpuz import audit
        old_path = audit.AUDIT_PATH
        audit.AUDIT_PATH = tempfile.mktemp(suffix=".jsonl")
        try:
            audit.log_action("up", "test-tpu", {"accelerator": "v4-8"})
            audit.log_action("down", "test-tpu")
            history = audit.get_history()
            assert len(history) == 2
            assert history[0]["action"] == "up"
            assert history[1]["action"] == "down"
            history_filtered = audit.get_history("test-tpu")
            assert len(history_filtered) == 2
            audit.clear_history()
            assert audit.get_history() == []
        finally:
            audit.AUDIT_PATH = old_path


class TestSSHResult:
    def test_ok(self):
        from tpuz.tpu import SSHResult
        r = SSHResult("output", "", 0)
        assert r.ok
        assert str(r) == "output"

    def test_failed(self):
        from tpuz.tpu import SSHResult
        r = SSHResult("", "error msg", 1)
        assert not r.ok

class TestSSHTimeout:
    def test_default_timeout(self):
        t = TPU("test", accelerator="v4-8")
        assert t.ssh_timeout == 120

    def test_custom_timeout(self):
        t = TPU("test", accelerator="v4-8", ssh_timeout=300)
        assert t.ssh_timeout == 300

    def test_none_timeout(self):
        t = TPU("test", accelerator="v4-8", ssh_timeout=None)
        assert t.ssh_timeout is None

class TestNumWorkersFor:
    def test_static(self):
        assert TPU.num_workers_for("v4-8") == 1
        assert TPU.num_workers_for("v4-32") == 4
        assert TPU.num_workers_for("v5litepod-64") == 8
