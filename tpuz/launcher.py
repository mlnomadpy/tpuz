"""
High-level launcher — one-command training on TPU.

Usage:
    from tpuz import Launcher

    l = Launcher("my-tpu", accelerator="v4-8")
    l.train(
        command="python train.py",
        sync="./src",
        setup_pip="flaxchat",
        env={"WANDB_API_KEY": "..."},
    )
"""

from tpuz.tpu import TPU


class Launcher:
    """
    One-command TPU training orchestrator.

    Handles: provision → setup → sync → run → logs → teardown.
    """

    def __init__(self, name, accelerator="v4-8", zone="us-central2-b",
                 project=None, preemptible=True):
        self.tpu = TPU(name, accelerator, zone, project, preemptible)

    def train(
        self,
        command,
        sync=None,
        setup_pip="",
        env=None,
        follow_logs=True,
        teardown_after=False,
        auto_recover=False,
    ):
        """
        Full training lifecycle.

        Args:
            command: Training command (e.g. "python train.py")
            sync: Local directory to upload
            setup_pip: Extra pip packages to install
            env: Environment variables dict
            follow_logs: Stream logs after launch
            teardown_after: Delete VM when done
            auto_recover: Watch for preemption and auto-restart
        """
        # 1. Create
        self.tpu.up()

        # 2. Setup
        self.tpu.setup(extra_pip=setup_pip)

        # 3. Launch
        self.tpu.run(command, env=env, sync=sync)

        # 4. Monitor
        if auto_recover:
            self.tpu.watch(command)
        elif follow_logs:
            self.tpu.logs(follow=True)

        # 5. Teardown
        if teardown_after:
            self.tpu.down()

    def __repr__(self):
        return f"Launcher({self.tpu!r})"
