from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import textwrap
import time


HERMES_REPO_URL = "https://github.com/NousResearch/hermes-agent"
HERMES_INSTALL_REF = "338c07433699569c24c32df4a2d1a8b9472400a8"

IMPORTANT_REFERENCES: tuple[tuple[str, str], ...] = (
    ("Hermes Agent", "https://github.com/NousResearch/hermes-agent"),
    ("Hermes documentation", "https://hermes-agent.nousresearch.com/docs/"),
    ("Unitree xr_teleoperate", "https://github.com/unitreerobotics/xr_teleoperate"),
    ("Unitree SDK2 Python", "https://github.com/unitreerobotics/unitree_sdk2_python"),
    ("Unitree MuJoCo", "https://github.com/unitreerobotics/unitree_mujoco"),
    ("OpenAI Codex skills", "https://developers.openai.com/codex/skills"),
    ("OpenAI Codex subagents", "https://developers.openai.com/codex/subagents"),
)

DANGEROUS_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bssh\b.*\b192\.168\.123\.", re.IGNORECASE),
    re.compile(r"\bscp\b.*\b192\.168\.123\.", re.IGNORECASE),
    re.compile(r"\bunitree_sdk2py\b.*\b(lowcmd|sport|wireless|motion|publish)\b", re.IGNORECASE),
    re.compile(r"\bdds\b.*\b(pub|publish|write|send)\b", re.IGNORECASE),
    re.compile(r"\bros2\s+topic\s+pub\b", re.IGNORECASE),
    re.compile(r"\bteleop(erate)?\b.*\b(real|robot|hardware)\b", re.IGNORECASE),
)


@dataclass(frozen=True)
class RepoSpec:
    name: str
    url: str
    description: str
    sparse_paths: tuple[str, ...] = ("README.md",)


UNITREE_REPOS = (
    RepoSpec(
        name="xr_teleoperate",
        url="https://github.com/unitreerobotics/xr_teleoperate.git",
        description="XR teleoperation stack and G1 assets.",
        sparse_paths=("README.md", "README_zh.md", "assets/g1"),
    ),
    RepoSpec(
        name="unitree_sdk2_python",
        url="https://github.com/unitreerobotics/unitree_sdk2_python.git",
        description="Python SDK examples and DDS-facing client code.",
        sparse_paths=("README.md", "example"),
    ),
    RepoSpec(
        name="unitree_mujoco",
        url="https://github.com/unitreerobotics/unitree_mujoco.git",
        description="MuJoCo simulator bridge for Unitree robots.",
        sparse_paths=("README.md", "simulate_python", "robots"),
    ),
)


@dataclass(frozen=True)
class LabConfig:
    workdir: str = "/content/unitree-hermes-lab"
    install_hermes: bool = True
    clone_unitree_repos: bool = True
    run_hermes_agent: bool = False
    provider: str = ""
    model: str = ""
    repo_limit: int = 3


@dataclass(frozen=True)
class RepoInspection:
    name: str
    url: str
    path: str
    present: bool
    commit: str
    file_count: int
    readme_lines: int
    signals: tuple[str, ...]


@dataclass(frozen=True)
class UseCase:
    title: str
    job: str
    prompt: str
    artifact: str
    value_score: int
    requires_model: bool


@dataclass(frozen=True)
class GateResult:
    name: str
    status: str
    details: str
    severity: str = "required"


@dataclass(frozen=True)
class HermesRunResult:
    title: str
    status: str
    output: str
    elapsed_s: float


@dataclass(frozen=True)
class LabReport:
    config: dict[str, object]
    runtime: dict[str, str]
    repos: tuple[RepoInspection, ...]
    use_cases: tuple[UseCase, ...]
    gates: tuple[GateResult, ...]
    hermes_runs: tuple[HermesRunResult, ...]
    artifacts: dict[str, str]
    usefulness_score: float

    def to_json_dict(self) -> dict[str, object]:
        return {
            "config": self.config,
            "runtime": self.runtime,
            "repos": [asdict(repo) for repo in self.repos],
            "use_cases": [asdict(use_case) for use_case in self.use_cases],
            "gates": [asdict(gate) for gate in self.gates],
            "hermes_runs": [asdict(run) for run in self.hermes_runs],
            "artifacts": self.artifacts,
            "usefulness_score": self.usefulness_score,
        }


def runtime_versions() -> dict[str, str]:
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "executable": sys.executable,
        "cuda": "not checked",
        "gpu": "not detected",
        "hermes_cli": "not installed",
    }

    if shutil.which("nvidia-smi"):
        gpu = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if gpu.stdout.strip():
            info["gpu"] = gpu.stdout.strip().splitlines()[0]
            info["cuda"] = "nvidia-smi available"

    if shutil.which("hermes"):
        version = subprocess.run(
            ["hermes", "--version"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=20,
        )
        info["hermes_cli"] = version.stdout.strip() or f"exit {version.returncode}"
    return info


def hermes_install_command(ref: str = HERMES_INSTALL_REF) -> list[str]:
    package = f"git+{HERMES_REPO_URL}.git@{ref}"
    return [sys.executable, "-m", "pip", "install", "-q", package]


def install_hermes_if_requested(config: LabConfig) -> GateResult:
    if not config.install_hermes:
        return GateResult("Hermes install", "SKIP", "INSTALL_HERMES is disabled.", "optional")
    if sys.version_info < (3, 11):
        return GateResult("Hermes install", "FAIL", "Hermes Agent requires Python 3.11 or newer.")
    if shutil.which("hermes"):
        return GateResult("Hermes install", "PASS", "Hermes CLI is already available.")

    command = hermes_install_command()
    result = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=900,
    )
    if result.returncode == 0 and shutil.which("hermes"):
        return GateResult("Hermes install", "PASS", f"Installed Hermes from {HERMES_INSTALL_REF[:12]}.")

    tail = "\n".join(result.stdout.strip().splitlines()[-12:])
    return GateResult("Hermes install", "FAIL", f"Install command failed. Last output:\n{tail}")


def clone_or_update_repo(spec: RepoSpec, root: Path) -> RepoInspection:
    root.mkdir(parents=True, exist_ok=True)
    path = root / spec.name
    env = os.environ.copy()
    env["GIT_LFS_SKIP_SMUDGE"] = "1"
    if path.exists():
        subprocess.run(
            ["git", "-C", str(path), "pull", "--ff-only"],
            check=False,
            stdout=subprocess.PIPE,
            env=env,
            timeout=240,
        )
    else:
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--filter=blob:none",
                "--sparse",
                spec.url,
                str(path),
            ],
            check=False,
            stdout=subprocess.PIPE,
            env=env,
            timeout=300,
        )
        if path.exists():
            subprocess.run(
                ["git", "-C", str(path), "sparse-checkout", "set", *spec.sparse_paths],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                env=env,
                timeout=120,
            )
    return inspect_repo(spec, path)


def inspect_repo(spec: RepoSpec, path: Path) -> RepoInspection:
    present = path.exists()
    commit = ""
    file_count = 0
    readme_lines = 0
    signals: list[str] = []

    if present:
        commit_proc = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        commit = commit_proc.stdout.strip()
        files = [p for p in path.rglob("*") if p.is_file() and ".git" not in p.parts]
        file_count = len(files)
        readme = path / "README.md"
        if readme.exists():
            text = readme.read_text(encoding="utf-8", errors="replace")
            readme_lines = len(text.splitlines())
            lower = text.lower()
            for needle, label in (
                ("--sim", "simulation flag documented"),
                ("--ipc", "agent IPC flag documented"),
                ("g1", "G1 mentioned"),
                ("mujoco", "MuJoCo mentioned"),
                ("cyclonedds", "CycloneDDS mentioned"),
            ):
                if needle in lower:
                    signals.append(label)

    return RepoInspection(
        name=spec.name,
        url=spec.url,
        path=str(path),
        present=present,
        commit=commit,
        file_count=file_count,
        readme_lines=readme_lines,
        signals=tuple(signals),
    )


def build_use_cases() -> tuple[UseCase, ...]:
    base_rule = (
        "You are reviewing Unitree open-source material in a Colab notebook. "
        "Do not execute robot-control commands, do not publish DDS messages, "
        "do not SSH to robot LAN addresses, and do not claim hardware validation."
    )
    return (
        UseCase(
            title="G1 simulation runbook",
            job="Turn xr_teleoperate setup notes into a short simulation-first runbook.",
            prompt=base_rule
            + "\nInspect the cloned xr_teleoperate README context and produce a G1 simulation runbook with manual review gates.",
            artifact="unitree-g1-sim-runbook.md",
            value_score=8,
            requires_model=True,
        ),
        UseCase(
            title="Teleoperation safety checklist",
            job="Generate a human-run checklist for PC2/image-service setup without remote actuation.",
            prompt=base_rule
            + "\nCreate a preflight checklist for Unitree XR teleoperation image service setup. Return commands as text only.",
            artifact="unitree-teleop-preflight.md",
            value_score=8,
            requires_model=True,
        ),
        UseCase(
            title="Log triage",
            job="Classify uploaded xr_teleoperate, teleimager, or SDK logs into likely causes.",
            prompt=base_rule
            + "\nGiven log excerpts, classify failures into camera, certificate, network, DDS, dependency, or simulator setup buckets.",
            artifact="unitree-log-triage.md",
            value_score=7,
            requires_model=True,
        ),
        UseCase(
            title="Repo contribution scout",
            job="Find small, testable Unitree documentation or tooling contributions.",
            prompt=base_rule
            + "\nReview cloned Unitree repositories and propose one small contribution with tests, rollback, and reviewer evidence.",
            artifact="unitree-contribution-candidate.md",
            value_score=7,
            requires_model=True,
        ),
        UseCase(
            title="Colab IK evidence bridge",
            job="Connect the existing G1 IK benchmark outputs to a broader Unitree review report.",
            prompt=base_rule
            + "\nExplain how a hardware-free IK smoke test can support, but not prove, teleoperation readiness.",
            artifact="unitree-ik-evidence-note.md",
            value_score=6,
            requires_model=True,
        ),
    )


def build_agents_md() -> str:
    return textwrap.dedent(
        """\
        # Unitree Hermes Colab Safety Rules

        This workspace is for read-only Unitree repository analysis, runbook generation,
        simulation planning, and log triage.

        Required behavior:
        - Treat Colab as an offline analysis environment.
        - Do not execute robot-control commands.
        - Do not publish DDS, ROS, motor, sport-mode, or low-level commands.
        - Do not SSH, SCP, tunnel, or scan Unitree robot LAN addresses such as 192.168.123.0/24.
        - Do not claim real hardware validation unless the user provides explicit external evidence.
        - Return risky commands as quoted text for a human to run on the correct local host.
        - Prefer short artifacts with assumptions, pass/fail gates, and rollback notes.
        """
    )


def has_dangerous_text(text: str) -> bool:
    return any(pattern.search(text) for pattern in DANGEROUS_PATTERNS)


def build_review_gates(
    *,
    config: LabConfig,
    install_gate: GateResult,
    repos: tuple[RepoInspection, ...],
    use_cases: tuple[UseCase, ...],
    hermes_runs: tuple[HermesRunResult, ...],
) -> tuple[GateResult, ...]:
    gates: list[GateResult] = [install_gate]

    python_ok = sys.version_info >= (3, 11)
    gates.append(
        GateResult(
            "Runtime",
            "PASS" if python_ok else "FAIL",
            f"Python {sys.version.split()[0]} detected; Hermes expects Python 3.11 or newer.",
        )
    )

    cloned = [repo for repo in repos if repo.present]
    gates.append(
        GateResult(
            "Unitree repository provenance",
            "PASS" if cloned else "FAIL",
            f"{len(cloned)} Unitree repositories available with commit SHAs.",
        )
    )

    secrets_present = any(
        os.environ.get(name)
        for name in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "NOUS_API_KEY", "ANTHROPIC_API_KEY")
    )
    gates.append(
        GateResult(
            "Model credential",
            "PASS" if secrets_present else "SKIP",
            "At least one provider key is present." if secrets_present else "No model key found; agent runs are skipped.",
            "optional",
        )
    )

    dangerous_prompts = [case.title for case in use_cases if has_dangerous_text(case.prompt)]
    gates.append(
        GateResult(
            "No-actuation prompt safety",
            "FAIL" if dangerous_prompts else "PASS",
            "Dangerous prompt pattern found in: " + ", ".join(dangerous_prompts)
            if dangerous_prompts
            else "Generated prompts prohibit robot actuation and robot-LAN access.",
        )
    )

    gates.append(
        GateResult(
            "Use-case coverage",
            "PASS" if len(use_cases) >= 5 else "FAIL",
            f"{len(use_cases)} use cases generated.",
        )
    )

    if config.run_hermes_agent:
        ran_ok = [run for run in hermes_runs if run.status == "PASS"]
        gates.append(
            GateResult(
                "Hermes execution",
                "PASS" if ran_ok else "FAIL",
                f"{len(ran_ok)} Hermes use-case prompts completed.",
            )
        )
    else:
        gates.append(
            GateResult(
                "Hermes execution",
                "SKIP",
                "RUN_HERMES_AGENT is false; prompts are prepared but not executed.",
                "optional",
            )
        )

    return tuple(gates)


def build_hermes_command(use_case: UseCase, config: LabConfig) -> list[str]:
    command = ["hermes", "--oneshot", use_case.prompt]
    if config.provider:
        command[1:1] = ["--provider", config.provider]
    if config.model:
        command[1:1] = ["--model", config.model]
    return command


def run_hermes_use_cases(use_cases: tuple[UseCase, ...], config: LabConfig) -> tuple[HermesRunResult, ...]:
    if not config.run_hermes_agent:
        return ()
    if not shutil.which("hermes"):
        return (
            HermesRunResult(
                title="Hermes execution",
                status="FAIL",
                output="Hermes CLI is not installed.",
                elapsed_s=0.0,
            ),
        )

    runs: list[HermesRunResult] = []
    # Run one prompt by default to bound cost and risk in Colab.
    for use_case in use_cases[:1]:
        command = build_hermes_command(use_case, config)
        start = time.perf_counter()
        result = subprocess.run(
            command,
            cwd=config.workdir,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=240,
        )
        elapsed = time.perf_counter() - start
        runs.append(
            HermesRunResult(
                title=use_case.title,
                status="PASS" if result.returncode == 0 else "FAIL",
                output=result.stdout.strip()[-6000:],
                elapsed_s=elapsed,
            )
        )
    return tuple(runs)


def usefulness_score(gates: tuple[GateResult, ...], repos: tuple[RepoInspection, ...]) -> float:
    required = [gate for gate in gates if gate.severity == "required"]
    passed = sum(1 for gate in required if gate.status == "PASS")
    repo_bonus = min(sum(repo.file_count for repo in repos) / 5000, 1.0)
    base = 2.5 + (passed / max(len(required), 1)) * 4.0
    return round(min(base + repo_bonus * 1.5, 8.0), 1)


def render_cards_html(report: LabReport) -> str:
    gate_colors = {"PASS": "#177245", "FAIL": "#b42318", "SKIP": "#8a6d1d"}
    gate_cards = "".join(
        f"""
        <div style="border:1px solid #d9e2ec;border-left:6px solid {gate_colors.get(gate.status, '#52606d')};border-radius:8px;padding:12px;background:white">
          <div style="font-size:12px;color:#627d98;text-transform:uppercase;letter-spacing:.08em">{gate.status}</div>
          <div style="font-weight:800;color:#102a43;margin-top:3px">{gate.name}</div>
          <div style="font-size:13px;color:#486581;line-height:1.45;margin-top:5px;white-space:pre-wrap">{_html_escape(gate.details)}</div>
        </div>
        """
        for gate in report.gates
    )
    repo_rows = "".join(
        f"""
        <tr>
          <td><b>{_html_escape(repo.name)}</b></td>
          <td>{_html_escape(repo.commit or 'missing')}</td>
          <td>{repo.file_count}</td>
          <td>{_html_escape(', '.join(repo.signals) or 'no README signal')}</td>
        </tr>
        """
        for repo in report.repos
    )
    use_case_cards = "".join(
        f"""
        <div style="border:1px solid #d9e2ec;border-radius:8px;padding:13px;background:#fbfdff">
          <div style="font-weight:800;color:#102a43">{_html_escape(use_case.title)}</div>
          <div style="font-size:13px;color:#486581;line-height:1.45;margin-top:5px">{_html_escape(use_case.job)}</div>
          <div style="font-size:12px;color:#627d98;margin-top:8px">Artifact: {_html_escape(use_case.artifact)} | Value: {use_case.value_score}/10</div>
        </div>
        """
        for use_case in report.use_cases
    )
    return f"""
<div style="font-family:system-ui,-apple-system,Segoe UI,sans-serif;max-width:1100px;color:#243b53">
  <div style="border:1px solid #d9e2ec;border-radius:10px;padding:18px 20px;background:#f8fbff">
    <div style="font-size:13px;color:#627d98;text-transform:uppercase;letter-spacing:.08em">Unitree Hermes Agent Safety Lab</div>
    <div style="font-size:30px;font-weight:900;color:#102a43;margin-top:4px">Colab read-only agent workflow</div>
    <div style="display:flex;gap:18px;flex-wrap:wrap;margin-top:12px;font-size:14px;color:#334e68">
      <div><b>Python</b> {_html_escape(report.runtime.get('python', ''))}</div>
      <div><b>GPU</b> {_html_escape(report.runtime.get('gpu', ''))}</div>
      <div><b>Hermes</b> {_html_escape(report.runtime.get('hermes_cli', ''))}</div>
      <div><b>Usefulness</b> {report.usefulness_score}/10</div>
    </div>
  </div>
  <h2 style="font-size:22px;margin:20px 0 10px;color:#102a43">Review Gates</h2>
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:12px">{gate_cards}</div>
  <h2 style="font-size:22px;margin:20px 0 10px;color:#102a43">Unitree Repository Map</h2>
  <table style="border-collapse:collapse;width:100%;font-size:14px;background:white;border:1px solid #d9e2ec">
    <thead style="background:#edf2f7;color:#243b53"><tr><th style="text-align:left;padding:9px">Repo</th><th style="text-align:left;padding:9px">Commit</th><th style="text-align:left;padding:9px">Files</th><th style="text-align:left;padding:9px">Signals</th></tr></thead>
    <tbody>{repo_rows}</tbody>
  </table>
  <h2 style="font-size:22px;margin:20px 0 10px;color:#102a43">Use Cases</h2>
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px">{use_case_cards}</div>
</div>
"""


def render_flow_svg() -> str:
    return """
<svg viewBox="0 0 920 230" width="100%" style="max-width:980px;background:#fbfdff;border:1px solid #d9e2ec;border-radius:10px">
  <defs>
    <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#486581"/></marker>
  </defs>
  <style>
    .box{fill:#ffffff;stroke:#bcccdc;stroke-width:1.5}
    .title{font:bold 15px system-ui, sans-serif;fill:#102a43}
    .body{font:12px system-ui, sans-serif;fill:#486581}
    .line{stroke:#486581;stroke-width:2;marker-end:url(#arrow)}
  </style>
  <rect class="box" x="30" y="48" width="150" height="92" rx="8"/>
  <text class="title" x="50" y="78">Colab runtime</text>
  <text class="body" x="50" y="104">versions, secrets</text>
  <text class="body" x="50" y="124">no robot LAN</text>
  <line class="line" x1="180" y1="94" x2="250" y2="94"/>
  <rect class="box" x="250" y="48" width="150" height="92" rx="8"/>
  <text class="title" x="270" y="78">Hermes CLI</text>
  <text class="body" x="270" y="104">install, health</text>
  <text class="body" x="270" y="124">agent opt-in</text>
  <line class="line" x1="400" y1="94" x2="470" y2="94"/>
  <rect class="box" x="470" y="48" width="170" height="92" rx="8"/>
  <text class="title" x="490" y="78">Unitree repos</text>
  <text class="body" x="490" y="104">xr, SDK, MuJoCo</text>
  <text class="body" x="490" y="124">commit provenance</text>
  <line class="line" x1="640" y1="94" x2="710" y2="94"/>
  <rect class="box" x="710" y="48" width="170" height="92" rx="8"/>
  <text class="title" x="730" y="78">Artifacts</text>
  <text class="body" x="730" y="104">runbooks, triage</text>
  <text class="body" x="730" y="124">review JSON</text>
  <rect x="250" y="168" width="390" height="34" rx="17" fill="#fff4e6" stroke="#f59f00"/>
  <text x="275" y="190" style="font:bold 13px system-ui,sans-serif;fill:#8a4b00">Safety boundary: generated commands are text, not robot execution.</text>
</svg>
"""


def render_markdown_report(report: LabReport) -> str:
    lines = [
        "# Unitree Hermes Agent Safety Lab",
        "",
        "This Colab workflow installs or checks Hermes Agent, clones selected Unitree repositories,",
        "and produces read-only setup, simulation, contribution, and log-triage artifacts.",
        "",
        "It does not control a physical robot from Colab.",
        "",
        f"Usefulness score: **{report.usefulness_score}/10**",
        "",
        "## Runtime",
        "",
    ]
    for key, value in report.runtime.items():
        lines.append(f"- **{key}**: {value}")
    lines.extend(["", "## Review Gates", ""])
    for gate in report.gates:
        lines.append(f"- **{gate.status}** {gate.name}: {gate.details}")
    lines.extend(["", "## Unitree Repositories", ""])
    for repo in report.repos:
        lines.append(
            f"- **{repo.name}** `{repo.commit or 'missing'}`: {repo.file_count} files; "
            f"signals: {', '.join(repo.signals) or 'none'}"
        )
    lines.extend(["", "## Use Cases", ""])
    for use_case in report.use_cases:
        lines.append(f"### {use_case.title}")
        lines.append("")
        lines.append(f"- Job: {use_case.job}")
        lines.append(f"- Artifact: `{use_case.artifact}`")
        lines.append(f"- Value score: {use_case.value_score}/10")
        lines.append("")
        lines.append("Prompt:")
        lines.append("")
        lines.append("```text")
        lines.append(use_case.prompt)
        lines.append("```")
        lines.append("")
    if report.hermes_runs:
        lines.extend(["## Hermes Runs", ""])
        for run in report.hermes_runs:
            lines.append(f"### {run.title}")
            lines.append(f"- Status: {run.status}")
            lines.append(f"- Elapsed: {run.elapsed_s:.1f}s")
            lines.append("")
            lines.append(run.output)
            lines.append("")
    lines.extend(["## References", ""])
    for label, url in IMPORTANT_REFERENCES:
        lines.append(f"- [{label}]({url})")
    lines.append("")
    return "\n".join(lines)


def write_artifacts(report: LabReport, artifact_dir: Path) -> dict[str, str]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "markdown_report": artifact_dir / "unitree-hermes-report.md",
        "review_json": artifact_dir / "unitree-hermes-review.json",
        "agents_md": artifact_dir / "AGENTS.md",
        "flow_svg": artifact_dir / "unitree-hermes-flow.svg",
    }
    artifact_paths = {name: str(path) for name, path in files.items()}
    report_with_artifacts = LabReport(
        config=report.config,
        runtime=report.runtime,
        repos=report.repos,
        use_cases=report.use_cases,
        gates=report.gates,
        hermes_runs=report.hermes_runs,
        artifacts=artifact_paths,
        usefulness_score=report.usefulness_score,
    )
    files["markdown_report"].write_text(render_markdown_report(report_with_artifacts), encoding="utf-8")
    files["review_json"].write_text(json.dumps(report_with_artifacts.to_json_dict(), indent=2), encoding="utf-8")
    files["agents_md"].write_text(build_agents_md(), encoding="utf-8")
    files["flow_svg"].write_text(render_flow_svg(), encoding="utf-8")
    return artifact_paths


def run_lab(config: LabConfig) -> LabReport:
    workdir = Path(config.workdir)
    repos_root = workdir / "repos"
    artifact_dir = workdir / "artifacts"

    install_gate = install_hermes_if_requested(config)
    runtime = runtime_versions()

    specs = UNITREE_REPOS[: max(0, min(config.repo_limit, len(UNITREE_REPOS)))]
    if config.clone_unitree_repos:
        repos = tuple(clone_or_update_repo(spec, repos_root) for spec in specs)
    else:
        repos = tuple(inspect_repo(spec, repos_root / spec.name) for spec in specs)

    use_cases = build_use_cases()
    safety_file = workdir / "AGENTS.md"
    workdir.mkdir(parents=True, exist_ok=True)
    safety_file.write_text(build_agents_md(), encoding="utf-8")

    hermes_runs = run_hermes_use_cases(use_cases, config)
    gates = build_review_gates(
        config=config,
        install_gate=install_gate,
        repos=repos,
        use_cases=use_cases,
        hermes_runs=hermes_runs,
    )

    report = LabReport(
        config=asdict(config),
        runtime=runtime,
        repos=repos,
        use_cases=use_cases,
        gates=gates,
        hermes_runs=hermes_runs,
        artifacts={},
        usefulness_score=usefulness_score(gates, repos),
    )
    artifacts = write_artifacts(report, artifact_dir)
    return LabReport(
        config=report.config,
        runtime=report.runtime,
        repos=report.repos,
        use_cases=report.use_cases,
        gates=report.gates,
        hermes_runs=report.hermes_runs,
        artifacts=artifacts,
        usefulness_score=report.usefulness_score,
    )


def _html_escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
