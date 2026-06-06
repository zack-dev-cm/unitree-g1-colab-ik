"""One-cell Colab runner for the Unitree Hermes Agent Safety Lab."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys

from IPython.display import HTML, Markdown, display


REPO_URL = "https://github.com/zack-dev-cm/unitree-g1-colab-ik.git"
WORKDIR = Path("/content/unitree-g1-colab-ik")

# Configuration: keep this block near the top when copying into a notebook.
INSTALL_HERMES = True
RUN_HERMES_AGENT = False
CLONE_UNITREE_REPOS = True
PROVIDER = ""
MODEL = ""
LAB_WORKDIR = "/content/unitree-hermes-lab"


def ensure_repo() -> None:
    if WORKDIR.exists():
        shutil.rmtree(WORKDIR)
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(WORKDIR)], check=True)
    sys.path.insert(0, str(WORKDIR / "src"))


def load_colab_secrets() -> None:
    try:
        from google.colab import userdata  # type: ignore
    except Exception:
        return

    for name in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "NOUS_API_KEY", "ANTHROPIC_API_KEY"):
        if os.environ.get(name):
            continue
        try:
            value = userdata.get(name)
        except Exception:
            value = None
        if value:
            os.environ[name] = value


def show_intro() -> None:
    display(
        HTML(
            """
<div style="font-family:system-ui,-apple-system,Segoe UI,sans-serif;max-width:1050px;border:1px solid #d9e2ec;border-radius:10px;background:#f8fbff;padding:18px 20px;color:#243b53">
  <div style="font-size:13px;color:#627d98;text-transform:uppercase;letter-spacing:.08em">Unitree Hermes Agent Safety Lab</div>
  <div style="font-size:30px;font-weight:900;color:#102a43;margin-top:4px">E2E Colab workflow for read-only Unitree agent use cases</div>
  <p style="font-size:15px;line-height:1.55;color:#486581;max-width:850px">
    This notebook installs or checks Hermes Agent, clones selected Unitree repositories, prepares safe agent prompts,
    and writes review artifacts. It treats Colab as an analysis environment, not a robot-control host.
  </p>
</div>
"""
        )
    )


def show_references() -> None:
    refs = (
        ("Hermes Agent", "https://github.com/NousResearch/hermes-agent"),
        ("Hermes documentation", "https://hermes-agent.nousresearch.com/docs/"),
        ("Unitree xr_teleoperate", "https://github.com/unitreerobotics/xr_teleoperate"),
        ("Unitree SDK2 Python", "https://github.com/unitreerobotics/unitree_sdk2_python"),
        ("Unitree MuJoCo", "https://github.com/unitreerobotics/unitree_mujoco"),
        ("OpenAI Codex skills", "https://developers.openai.com/codex/skills"),
        ("OpenAI Codex subagents", "https://developers.openai.com/codex/subagents"),
    )
    links = "".join(
        f'<li><a href="{url}" target="_blank" style="color:#0b7285;text-decoration:none">{label}</a></li>'
        for label, url in refs
    )
    display(
        HTML(
            f"""
<div style="font-family:system-ui,-apple-system,Segoe UI,sans-serif;max-width:1050px;margin-top:16px;border:1px solid #d9e2ec;border-radius:10px;background:white;padding:16px 20px;color:#243b53">
  <div style="font-size:13px;color:#627d98;text-transform:uppercase;letter-spacing:.08em">References</div>
  <ul style="margin:8px 0 0 18px;line-height:1.6">{links}</ul>
</div>
"""
        )
    )


def show_artifacts(report) -> None:
    rows = "".join(
        f"""
        <tr>
          <td style="padding:8px 10px;border-top:1px solid #edf2f7"><b>{name}</b></td>
          <td style="padding:8px 10px;border-top:1px solid #edf2f7"><code>{path}</code></td>
        </tr>
        """
        for name, path in report.artifacts.items()
    )
    display(
        HTML(
            f"""
<div style="font-family:system-ui,-apple-system,Segoe UI,sans-serif;max-width:1050px;margin-top:18px;border:1px solid #d9e2ec;border-radius:10px;background:white;padding:16px 20px;color:#243b53">
  <div style="font-size:13px;color:#627d98;text-transform:uppercase;letter-spacing:.08em">Saved Artifacts</div>
  <table style="border-collapse:collapse;width:100%;font-size:14px;margin-top:8px">{rows}</table>
</div>
"""
        )
    )


ensure_repo()
if RUN_HERMES_AGENT:
    load_colab_secrets()
show_intro()

from unitree_colab_ik.hermes_lab import LabConfig, render_cards_html, render_flow_svg, render_markdown_report, run_lab

report = run_lab(
    LabConfig(
        workdir=LAB_WORKDIR,
        install_hermes=INSTALL_HERMES,
        clone_unitree_repos=CLONE_UNITREE_REPOS,
        run_hermes_agent=RUN_HERMES_AGENT,
        provider=PROVIDER,
        model=MODEL,
    )
)

display(HTML(render_flow_svg()))
display(HTML(render_cards_html(report)))
show_artifacts(report)
show_references()

display(Markdown("## Generated Report Preview"))
display(Markdown(render_markdown_report(report)))
