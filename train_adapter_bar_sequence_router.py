#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from playpen.adapter_bar_sequence import train_router


def main():
    p = argparse.ArgumentParser(description="Train sequence-level Adapter-BAR router.")
    p.add_argument("--config", required=True, type=str)
    p.add_argument("--router_type", required=True, choices=["cluster", "game"])
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    metrics = train_router(args.config, args.router_type, debug=args.debug)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
