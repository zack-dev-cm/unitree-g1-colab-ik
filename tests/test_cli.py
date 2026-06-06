from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from unitree_colab_ik import cli


@dataclass(frozen=True)
class FakeResult:
    side: str = "left"
    device: str = "cpu"
    batch_size: int = 8
    steps: int = 10
    mean_error_m: float = 0.002
    p95_error_m: float = 0.004
    max_error_m: float = 0.006
    success_rate: float = 1.0
    limit_violation_rad: float = 0.0
    elapsed_s: float = 0.1
    targets_per_s: float = 80.0
    joint_names: tuple[str, ...] = ("joint",)

    def as_dict(self) -> dict[str, object]:
        return self.__dict__


def test_cli_json_output_and_repeated_side(monkeypatch, capsys):
    calls = {}

    def fake_run_benchmark(**kwargs):
        calls.update(kwargs)
        return [FakeResult(side=side) for side in kwargs["sides"]]

    monkeypatch.setattr(cli, "run_benchmark", fake_run_benchmark)

    code = cli.main(["--json", "--side", "left", "--side", "right", "--batch-size", "8", "--steps", "10"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert code == 0
    assert calls["sides"] == ("left", "right")
    assert [entry["side"] for entry in payload] == ["left", "right"]
    assert captured.err == ""


def test_cli_threshold_failure_returns_nonzero(monkeypatch, capsys):
    monkeypatch.setattr(
        cli,
        "run_benchmark",
        lambda **_kwargs: [FakeResult(side="left", success_rate=0.5, mean_error_m=0.2)],
    )

    code = cli.main(["--min-success-rate", "0.99", "--max-mean-error-m", "0.01"])

    captured = capsys.readouterr()
    assert code == 1
    assert "Benchmark failed for: left" in captured.err


def test_console_script_target_declared():
    pyproject = Path(cli.__file__).parents[2] / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")

    assert 'unitree-g1-ik-bench = "unitree_colab_ik.cli:main"' in text
