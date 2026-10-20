"""JSON command interface for future Orchestrator integration."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from .adapters import NebiusAdapter
from .controller import Controller
from .service import run_forever


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m llm_optimizer.remote")
    parser.add_argument(
        "--database",
        default=os.getenv("LLM_OPTIMIZER_CONTROLLER_DB", "optimizer-controller.sqlite"),
    )
    parser.add_argument("--concurrency", type=int, default=2)
    sub = parser.add_subparsers(dest="operation", required=True)
    submit = sub.add_parser("submit")
    submit.add_argument("request", type=Path, help="JSON request file")
    submit.add_argument("--idempotency-key")
    for name in ("status", "results", "cancel"):
        command = sub.add_parser(name)
        command.add_argument("run_id")
        if name == "cancel":
            command.add_argument("--framework")
    sub.add_parser("tick")
    serve = sub.add_parser("serve")
    serve.add_argument("--interval", type=float, default=15)
    args = parser.parse_args()
    controller = Controller(
        args.database,
        NebiusAdapter(os.getenv("NB_PROJECT_ID")),
        concurrency=args.concurrency,
    )
    if args.operation == "submit":
        result = controller.submit(
            json.loads(args.request.read_text()), args.idempotency_key
        )
    elif args.operation == "status":
        result = controller.status(args.run_id)
    elif args.operation == "results":
        result = controller.results(args.run_id)
    elif args.operation == "cancel":
        result = controller.cancel(args.run_id, args.framework)
    elif args.operation == "tick":
        controller.tick()
        result = {"status": "ok"}
    else:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
        run_forever(controller, args.interval)
    json.dump(result, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
