"""Unit tests for TPU class — no GCP access needed."""

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
        assert TPU._worker_count("v4-4") == 1
        assert TPU._worker_count("unknown") == 1

    def test_repr(self):
        t = TPU("my-tpu", "v4-8", "us-central2-b")
        r = repr(t)
        assert "my-tpu" in r
        assert "v4-8" in r

    def test_zone(self):
        t = TPU("x", zone="europe-west4-a")
        assert t.zone == "europe-west4-a"

    def test_preemptible_default(self):
        t = TPU("x")
        assert t.preemptible is True

    def test_custom_runtime(self):
        t = TPU("x", runtime="custom-runtime-v1")
        assert t.runtime == "custom-runtime-v1"


class TestTPUInfo:
    def test_dataclass(self):
        info = TPUInfo(
            name="test", state="READY", accelerator="v4-8",
            zone="us-central2-b", external_ips=["1.2.3.4"], internal_ips=["10.0.0.1"],
        )
        assert info.state == "READY"
        assert info.external_ips == ["1.2.3.4"]

    def test_preemptible_default(self):
        info = TPUInfo("t", "READY", "v4-8", "z", [], [])
        assert info.preemptible is False


class TestCLI:
    def test_help(self):
        import subprocess, sys
        r = subprocess.run([sys.executable, "-m", "tpuz.cli", "--help"],
                           capture_output=True, text=True)
        assert r.returncode == 0
        assert "tpuz" in r.stdout.lower()

    def test_up_help(self):
        import subprocess, sys
        r = subprocess.run([sys.executable, "-m", "tpuz.cli", "up", "--help"],
                           capture_output=True, text=True)
        assert r.returncode == 0

    def test_train_help(self):
        import subprocess, sys
        r = subprocess.run([sys.executable, "-m", "tpuz.cli", "train", "--help"],
                           capture_output=True, text=True)
        assert r.returncode == 0
        assert "recover" in r.stdout
