from __future__ import annotations

import argparse
import json
from pathlib import Path

from .hermes_lab import LabConfig, run_lab


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate a safety-gated Hermes Agent lab report for Unitree repositories."
    )
    parser.add_argument("--workdir", default="./unitree-hermes-lab")
    parser.add_argument("--install-hermes", action="store_true", help="Install Hermes from the pinned GitHub ref.")
    parser.add_argument("--skip-clone", action="store_true", help="Do not clone or update Unitree repositories.")
    parser.add_argument("--run-hermes-agent", action="store_true", help="Run one Hermes one-shot prompt.")
    parser.add_argument("--provider", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--repo-limit", type=int, default=3)
    parser.add_argument("--json", action="store_true", help="Print the review JSON to stdout.")
    args = parser.parse_args(argv)

    report = run_lab(
        LabConfig(
            workdir=args.workdir,
            install_hermes=args.install_hermes,
            clone_unitree_repos=not args.skip_clone,
            run_hermes_agent=args.run_hermes_agent,
            provider=args.provider,
            model=args.model,
            repo_limit=args.repo_limit,
        )
    )

    if args.json:
        print(json.dumps(report.to_json_dict(), indent=2))
    else:
        print(f"Unitree Hermes report: {Path(report.artifacts['markdown_report']).resolve()}")
        print(f"Review JSON: {Path(report.artifacts['review_json']).resolve()}")
        print(f"Usefulness score: {report.usefulness_score}/10")
        for gate in report.gates:
            detail_lines = gate.details.splitlines()
            detail = detail_lines[0] if detail_lines else "no details"
            print(f"{gate.status:4s} {gate.name}: {detail}")
    return 0 if all(gate.status != "FAIL" for gate in report.gates if gate.severity == "required") else 1


if __name__ == "__main__":
    raise SystemExit(main())
