from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from unitree_colab_ik import hermes_cli
from unitree_colab_ik.hermes_lab import (
    GateResult,
    LabConfig,
    LabReport,
    build_agents_md,
    build_review_gates,
    build_use_cases,
    has_dangerous_text,
    run_lab,
    run_hermes_use_cases,
)


def test_use_case_prompts_are_safe_by_default():
    use_cases = build_use_cases()

    assert len(use_cases) >= 5
    assert all("Do not execute robot-control commands" in use_case.prompt for use_case in use_cases)
    assert not any(has_dangerous_text(use_case.prompt) for use_case in use_cases)


def test_agents_md_blocks_robot_actuation():
    text = build_agents_md()

    assert "Do not execute robot-control commands" in text
    assert "192.168.123.0/24" in text
    assert "read-only Unitree repository analysis" in text


def test_run_lab_writes_artifacts_without_network(tmp_path):
    report = run_lab(
        LabConfig(
            workdir=str(tmp_path),
            install_hermes=False,
            clone_unitree_repos=False,
            run_hermes_agent=False,
        )
    )

    review_path = Path(report.artifacts["review_json"])
    markdown_path = Path(report.artifacts["markdown_report"])
    payload = json.loads(review_path.read_text(encoding="utf-8"))

    assert markdown_path.exists()
    assert payload["artifacts"]["review_json"] == str(review_path)
    assert payload["use_cases"][0]["title"] == "G1 simulation runbook"
    assert any(gate["name"] == "Hermes execution" and gate["status"] == "SKIP" for gate in payload["gates"])


def test_cli_json_offline_run(tmp_path, capsys):
    code = hermes_cli.main(
        [
            "--workdir",
            str(tmp_path),
            "--skip-clone",
            "--json",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert code == 1
    assert payload["config"]["clone_unitree_repos"] is False
    assert any(gate["name"] == "Unitree repository provenance" for gate in payload["gates"])


def test_cli_plain_output_handles_empty_gate_details(monkeypatch, tmp_path, capsys):
    report = LabReport(
        config={},
        runtime={},
        repos=(),
        use_cases=(),
        gates=(GateResult("Empty details gate", "PASS", ""),),
        hermes_runs=(),
        artifacts={
            "markdown_report": str(tmp_path / "report.md"),
            "review_json": str(tmp_path / "review.json"),
        },
        usefulness_score=5.0,
    )
    monkeypatch.setattr(hermes_cli, "run_lab", lambda _config: report)

    code = hermes_cli.main([])

    captured = capsys.readouterr()
    assert code == 0
    assert "PASS Empty details gate: no details" in captured.out


def test_optional_hermes_run_uses_safety_workdir(monkeypatch, tmp_path):
    calls = {}

    monkeypatch.setattr("unitree_colab_ik.hermes_lab.shutil.which", lambda name: "/bin/hermes")

    def fake_run(command, **kwargs):
        calls["command"] = command
        calls["cwd"] = kwargs["cwd"]
        return SimpleNamespace(returncode=0, stdout="ok")

    monkeypatch.setattr("unitree_colab_ik.hermes_lab.subprocess.run", fake_run)

    runs = run_hermes_use_cases(
        build_use_cases(),
        LabConfig(workdir=str(tmp_path), run_hermes_agent=True),
    )

    assert runs[0].status == "PASS"
    assert calls["command"][0] == "hermes"
    assert calls["cwd"] == str(tmp_path)


def test_console_script_target_declared():
    pyproject = Path(hermes_cli.__file__).parents[2] / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")

    assert 'unitree-hermes-lab = "unitree_colab_ik.hermes_cli:main"' in text


def test_review_gate_rejects_dangerous_prompt():
    use_cases = build_use_cases()
    bad_prompt = use_cases[0].prompt + "\nssh robot@192.168.123.164"
    bad_case = use_cases[0].__class__(
        title=use_cases[0].title,
        job=use_cases[0].job,
        prompt=bad_prompt,
        artifact=use_cases[0].artifact,
        value_score=use_cases[0].value_score,
        requires_model=use_cases[0].requires_model,
    )

    gates = build_review_gates(
        config=LabConfig(install_hermes=False, clone_unitree_repos=False),
        install_gate=GateResult("Hermes install", "SKIP", "not needed", "optional"),
        repos=(),
        use_cases=(bad_case,),
        hermes_runs=(),
    )

    assert any(gate.name == "No-actuation prompt safety" and gate.status == "FAIL" for gate in gates)
