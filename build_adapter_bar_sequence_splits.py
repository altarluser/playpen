#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from playpen.adapter_bar_sequence import build_router_splits


def main():
    p = argparse.ArgumentParser(description="Build sequence-level Adapter-BAR router splits.")
    p.add_argument("--config", required=True, type=str)
    p.add_argument("--router_type", required=True, choices=["cluster", "game"])
    p.add_argument("--debug_num_examples", type=int, default=None)
    args = p.parse_args()

    summary = build_router_splits(args.config, args.router_type, args.debug_num_examples)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
