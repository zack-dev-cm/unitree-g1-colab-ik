from __future__ import annotations

import argparse
import json
import sys

from .benchmark import run_benchmark
from .urdf import DEFAULT_G1_URDF_URL


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a Unitree G1 batched IK benchmark.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--side", action="append", choices=["left", "right"], help="Repeat to run selected sides.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=220)
    parser.add_argument("--urdf-path")
    parser.add_argument("--urdf-url", default=DEFAULT_G1_URDF_URL)
    parser.add_argument("--success-threshold-m", type=float, default=0.02)
    parser.add_argument("--min-success-rate", type=float, default=0.98)
    parser.add_argument("--max-mean-error-m", type=float, default=0.01)
    parser.add_argument("--json", action="store_true", help="Only print JSON metrics.")
    args = parser.parse_args(argv)

    sides = tuple(args.side or ["left", "right"])
    results = run_benchmark(
        sides=sides,
        batch_size=args.batch_size,
        steps=args.steps,
        device=args.device,
        urdf_path=args.urdf_path,
        urdf_url=args.urdf_url,
        success_threshold_m=args.success_threshold_m,
    )
    payload = [result.as_dict() for result in results]

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        for result in results:
            print(
                f"{result.side:5s} {result.device:4s} "
                f"mean={result.mean_error_m:.5f}m p95={result.p95_error_m:.5f}m "
                f"max={result.max_error_m:.5f}m success={result.success_rate:.3f} "
                f"limit_violation={result.limit_violation_rad:.2e} "
                f"throughput={result.targets_per_s:.1f} targets/s"
            )
        print(json.dumps(payload, indent=2))

    failed = [
        result
        for result in results
        if result.success_rate < args.min_success_rate
        or result.mean_error_m > args.max_mean_error_m
        or result.limit_violation_rad > 1e-6
    ]
    if failed:
        print(f"Benchmark failed for: {', '.join(result.side for result in failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

