#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="PlayPen/Clem eval entrypoint with Adapter-BAR routing modes.")
    p.add_argument("--model_path", type=str, default=None)
    p.add_argument("--model_name", type=str, default="llama3-8b")
    p.add_argument("--model_registry", type=Path, default=Path("model_registry.json"))
    p.add_argument("--suite", choices=["clem", "static", "all"], default="all")
    p.add_argument("--game", type=str, default=None)
    p.add_argument("--output_dir", type=Path, default=None)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_tokens", type=int, default=300)
    p.add_argument("--skip_gameplay", action="store_true")
    p.add_argument("--moe", type=str, default=None)
    p.add_argument("--routing_mode", choices=["default", "adapter_bar_sequence_cluster", "adapter_bar_sequence_game"], default="default")
    p.add_argument("--adapter_bar_router_path", type=str, default=None)
    p.add_argument("--adapter_bar_config", type=str, default="configs/adapter_bar_sequence.yaml")
    p.add_argument("--max_instances", type=int, default=None)
    args = p.parse_args()

    cmd = [
        sys.executable,
        "-m",
        "playpen.cli",
        "eval",
        args.model_name,
        "--suite",
        args.suite,
        "--model-registry",
        str(args.model_registry),
        "--routing_mode",
        args.routing_mode,
        "--adapter_bar_config",
        str(args.adapter_bar_config),
        "-T",
        str(args.temperature),
        "-L",
        str(args.max_tokens),
    ]
    if args.output_dir is not None:
        cmd.extend(["-r", str(args.output_dir)])
    if args.game:
        cmd.extend(["-g", args.game])
    if args.skip_gameplay:
        cmd.append("--skip_gameplay")
    if args.moe:
        cmd.extend(["--moe", args.moe])
    if args.adapter_bar_router_path:
        cmd.extend(["--adapter_bar_router_path", args.adapter_bar_router_path])
    if args.max_instances is not None:
        cmd.extend(["--max_instances", str(args.max_instances)])

    subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
