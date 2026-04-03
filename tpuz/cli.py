"""
tpuz CLI — manage TPU VMs from your terminal.

Usage:
    tpuz up my-tpu --accelerator=v4-8
    tpuz run my-tpu "python train.py" --sync=./src
    tpuz logs my-tpu
    tpuz status my-tpu
    tpuz ssh my-tpu "nvidia-smi"
    tpuz down my-tpu
    tpuz list
    tpuz train my-tpu "python train.py" --sync=. --recover
"""

import argparse
import json
import os


def main():
    p = argparse.ArgumentParser(prog="tpuz", description="Manage GCP TPU VMs")
    p.add_argument("--project", default=os.environ.get("GCLOUD_PROJECT", ""))
    p.add_argument("--zone", default="us-central2-b")
    sub = p.add_subparsers(dest="cmd", required=True)

    # tpuz up NAME
    s = sub.add_parser("up", help="Create TPU VM")
    s.add_argument("name")
    s.add_argument("--accelerator", "-a", default="v4-8")
    s.add_argument("--preemptible", action="store_true", default=True)
    s.add_argument("--runtime", default=None)

    # tpuz down NAME
    s = sub.add_parser("down", help="Delete TPU VM")
    s.add_argument("name")

    # tpuz status NAME
    s = sub.add_parser("status", help="Show VM status")
    s.add_argument("name")

    # tpuz ssh NAME "cmd"
    s = sub.add_parser("ssh", help="Run SSH command")
    s.add_argument("name")
    s.add_argument("command")
    s.add_argument("--worker", "-w", type=int, default=0)

    # tpuz run NAME "cmd"
    s = sub.add_parser("run", help="Launch detached training")
    s.add_argument("name")
    s.add_argument("command")
    s.add_argument("--sync", default=None, help="Local dir to upload")
    s.add_argument("--env", default=None, help="KEY=VAL,KEY2=VAL2")

    # tpuz logs NAME
    s = sub.add_parser("logs", help="Stream training logs")
    s.add_argument("name")
    s.add_argument("--lines", "-n", type=int, default=50)
    s.add_argument("--no-follow", action="store_true")

    # tpuz setup NAME
    s = sub.add_parser("setup", help="Install JAX + deps")
    s.add_argument("name")
    s.add_argument("--pip", default="", help="Extra pip packages")

    # tpuz kill NAME
    s = sub.add_parser("kill", help="Kill training process")
    s.add_argument("name")

    # tpuz watch NAME "cmd"
    s = sub.add_parser("watch", help="Auto-recover from preemption")
    s.add_argument("name")
    s.add_argument("command")
    s.add_argument("--retries", type=int, default=5)

    # tpuz upload NAME local remote
    s = sub.add_parser("upload", help="Upload file to VM")
    s.add_argument("name"); s.add_argument("local"); s.add_argument("remote")

    # tpuz download NAME remote local
    s = sub.add_parser("download", help="Download file from VM")
    s.add_argument("name"); s.add_argument("remote"); s.add_argument("local")

    # tpuz list
    s = sub.add_parser("list", help="List TPU VMs")

    # tpuz train NAME "cmd" (all-in-one)
    s = sub.add_parser("train", help="Full lifecycle: up → setup → run → logs")
    s.add_argument("name")
    s.add_argument("command")
    s.add_argument("--accelerator", "-a", default="v4-8")
    s.add_argument("--sync", default=None)
    s.add_argument("--pip", default="")
    s.add_argument("--env", default=None)
    s.add_argument("--recover", action="store_true")
    s.add_argument("--teardown", action="store_true")

    args = p.parse_args()

    from tpuz.tpu import TPU
    from tpuz.launcher import Launcher

    def _make_tpu(name, accelerator="v4-8", runtime=None):
        return TPU(name, accelerator, args.zone, args.project, runtime=runtime)

    if args.cmd == "up":
        t = _make_tpu(args.name, args.accelerator, args.runtime)
        t.up()

    elif args.cmd == "down":
        _make_tpu(args.name).down()

    elif args.cmd == "status":
        info = _make_tpu(args.name).info()
        if info:
            print(f"Name:        {info.name}")
            print(f"State:       {info.state}")
            print(f"Accelerator: {info.accelerator}")
            print(f"IPs:         {info.external_ips}")
        else:
            print("Not found")

    elif args.cmd == "ssh":
        print(_make_tpu(args.name).ssh(args.command, worker=args.worker))

    elif args.cmd == "run":
        t = _make_tpu(args.name)
        env = dict(kv.split("=", 1) for kv in args.env.split(",")) if args.env else None
        t.run(args.command, env=env, sync=args.sync)

    elif args.cmd == "logs":
        _make_tpu(args.name).logs(lines=args.lines, follow=not args.no_follow)

    elif args.cmd == "setup":
        _make_tpu(args.name).setup(extra_pip=args.pip)

    elif args.cmd == "kill":
        _make_tpu(args.name).kill()
        print("Killed")

    elif args.cmd == "watch":
        _make_tpu(args.name).watch(args.command, max_retries=args.retries)

    elif args.cmd == "upload":
        _make_tpu(args.name).scp_to(args.local, args.remote)
        print("Uploaded")

    elif args.cmd == "download":
        _make_tpu(args.name).scp_from(args.remote, args.local)
        print("Downloaded")

    elif args.cmd == "list":
        vms = TPU.list(zone=args.zone, project=args.project)
        if not vms:
            print("No TPU VMs found")
        for vm in vms:
            state = vm.get("state", "?")
            name = vm.get("name", "?")
            accel = vm.get("acceleratorType", "?").split("/")[-1]
            print(f"  {name:20s} {accel:15s} {state}")

    elif args.cmd == "train":
        env = dict(kv.split("=", 1) for kv in args.env.split(",")) if args.env else None
        l = Launcher(args.name, args.accelerator, args.zone, args.project)
        l.train(
            command=args.command,
            sync=args.sync,
            setup_pip=args.pip,
            env=env,
            auto_recover=args.recover,
            teardown_after=args.teardown,
        )


if __name__ == "__main__":
    main()
