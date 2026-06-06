"""One-cell Colab runner for the Unitree G1 GPU IK benchmark."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import HTML, display


REPO_URL = "https://github.com/zack-dev-cm/unitree-g1-colab-ik.git"
WORKDIR = Path("/content/unitree-g1-colab-ik")
BATCH_SIZE = 256
STEPS = 220


def ensure_repo() -> None:
    if WORKDIR.exists():
        shutil.rmtree(WORKDIR)
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(WORKDIR)], check=True)
    sys.path.insert(0, str(WORKDIR / "src"))


def runtime_card() -> None:
    assert torch.cuda.is_available(), "Enable Runtime > Change runtime type > GPU before running this notebook."
    gpu_name = torch.cuda.get_device_name(0)
    display(
        HTML(
            f"""
<div style="font-family:system-ui,-apple-system,Segoe UI,sans-serif;border:1px solid #d9e2ec;border-radius:12px;padding:18px 20px;background:#f8fbff;max-width:920px">
  <div style="font-size:13px;color:#486581;text-transform:uppercase;letter-spacing:.08em">Unitree G1 Colab GPU IK</div>
  <div style="font-size:28px;font-weight:800;color:#102a43;margin-top:4px">Runtime: {gpu_name}</div>
  <div style="display:flex;gap:18px;margin-top:10px;color:#334e68;font-size:14px;flex-wrap:wrap">
    <div><b>torch</b> {torch.__version__}</div>
    <div><b>cuda</b> {torch.version.cuda}</div>
    <div><b>source</b> {REPO_URL}</div>
  </div>
</div>
"""
        )
    )


def run_solver():
    from unitree_colab_ik.benchmark import build_chain, load_robot, solve_tracking_ik

    robot = load_robot()
    details = {}
    for index, side in enumerate(("left", "right")):
        chain = build_chain(robot, side, device="cuda")
        solved = solve_tracking_ik(chain, batch_size=BATCH_SIZE, steps=STEPS, seed=2026 + index)
        q = solved["q"].detach()
        margin = torch.minimum(q - chain.lower, chain.upper - q) / chain.half_range
        details[side] = {
            "chain": chain,
            "q": q.cpu(),
            "target_positions": solved["target_positions"].cpu(),
            "final_positions": solved["final_positions"].cpu(),
            "errors": solved["errors"].detach().cpu(),
            "margin": margin.detach().cpu(),
            "elapsed_s": solved["elapsed_s"],
            "success_rate": solved["success_rate"],
            "limit_violation_rad": solved["limit_violation_rad"],
        }
    return details


def summarize(details):
    rows = []
    for side, item in details.items():
        errors = item["errors"]
        rows.append(
            {
                "side": side,
                "mean_cm": float(errors.mean() * 100),
                "p95_cm": float(torch.quantile(errors, 0.95) * 100),
                "max_cm": float(errors.max() * 100),
                "success_rate": item["success_rate"],
                "elapsed_s": item["elapsed_s"],
                "targets_per_s": BATCH_SIZE / item["elapsed_s"],
                "limit_violation_rad": item["limit_violation_rad"],
            }
        )
    return rows


def summary_cards(summary) -> None:
    cards = []
    for row in summary:
        accent = "#1f9d8a" if row["side"] == "left" else "#d9480f"
        cards.append(
            f"""
    <div style="border:1px solid #d9e2ec;border-top:5px solid {accent};border-radius:12px;background:white;padding:16px;min-width:260px;box-shadow:0 4px 14px rgba(16,42,67,.07)">
      <div style="font-size:13px;color:#627d98;text-transform:uppercase;letter-spacing:.08em">{row['side']} arm</div>
      <div style="font-size:32px;font-weight:850;color:#102a43;margin:4px 0">{row['mean_cm']:.2f} cm</div>
      <div style="color:#486581">mean wrist position error</div>
      <hr style="border:none;border-top:1px solid #edf2f7;margin:12px 0">
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:9px;font-size:13px;color:#334e68">
        <div><b>p95</b><br>{row['p95_cm']:.2f} cm</div>
        <div><b>max</b><br>{row['max_cm']:.2f} cm</div>
        <div><b>success</b><br>{row['success_rate']*100:.1f}%</div>
        <div><b>throughput</b><br>{row['targets_per_s']:.1f}/s</div>
      </div>
    </div>
    """
        )

    display(
        HTML(
            """
<div style="font-family:system-ui,-apple-system,Segoe UI,sans-serif;max-width:980px;margin-top:12px">
  <h2 style="margin:0 0 10px;color:#102a43">Benchmark Summary</h2>
  <div style="display:flex;gap:16px;flex-wrap:wrap">%s</div>
</div>
"""
            % "".join(cards)
        )
    )


def configure_plots() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#fbfdff",
            "axes.edgecolor": "#bcccdc",
            "axes.labelcolor": "#243b53",
            "axes.titleweight": "bold",
            "font.size": 11,
            "grid.color": "#d9e2ec",
        }
    )


def error_plots(summary, details) -> None:
    colors = {"left": "#1f9d8a", "right": "#d9480f"}
    labels = [row["side"] for row in summary]
    x = np.arange(len(labels))
    width = 0.24

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.bar(x - width, [row["mean_cm"] for row in summary], width, label="mean", color="#1f9d8a")
    ax.bar(x, [row["p95_cm"] for row in summary], width, label="p95", color="#f59f00")
    ax.bar(x + width, [row["max_cm"] for row in summary], width, label="max", color="#d9480f")
    ax.axhline(2.0, color="#9b2c2c", linestyle="--", linewidth=1.3, label="2 cm target")
    ax.set_xticks(x, labels)
    ax.set_ylabel("position error (cm)")
    ax.set_title("Unitree G1 wrist IK error by side")
    ax.grid(axis="y", alpha=0.75)
    ax.legend(frameon=False, ncols=4, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=3, fontsize=9)
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), sharey=True)
    for ax, side in zip(axes, labels):
        err_cm = details[side]["errors"].numpy() * 100
        ax.hist(err_cm, bins=28, color=colors[side], alpha=0.88, edgecolor="white")
        ax.axvline(2.0, color="#9b2c2c", linestyle="--", linewidth=1.2)
        ax.set_title(f"{side.title()} arm error distribution")
        ax.set_xlabel("error (cm)")
        ax.grid(axis="y", alpha=0.6)
    axes[0].set_ylabel("target count")
    plt.tight_layout()
    plt.show()


def pose_plot(summary, details) -> None:
    labels = [row["side"] for row in summary]
    fig = plt.figure(figsize=(12, 5.4))
    for plot_index, side in enumerate(labels, start=1):
        ax = fig.add_subplot(1, 2, plot_index, projection="3d")
        target = details[side]["target_positions"].numpy()
        solved = details[side]["final_positions"].numpy()
        sample = np.linspace(0, len(target) - 1, min(80, len(target)), dtype=int)
        ax.scatter(
            target[sample, 0],
            target[sample, 1],
            target[sample, 2],
            s=20,
            color="#1f9d8a",
            alpha=0.72,
            label="target",
        )
        ax.scatter(
            solved[sample, 0],
            solved[sample, 1],
            solved[sample, 2],
            s=13,
            color="#d9480f",
            alpha=0.64,
            label="solved",
        )
        for i in sample[::8]:
            ax.plot(
                [target[i, 0], solved[i, 0]],
                [target[i, 1], solved[i, 1]],
                [target[i, 2], solved[i, 2]],
                color="#829ab1",
                alpha=0.35,
                linewidth=0.8,
            )
        ax.set_title(f"{side.title()} wrist targets vs solved poses")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        ax.legend(frameon=False, loc="upper left")
        ax.view_init(elev=22, azim=-55 if side == "left" else -125)
    plt.tight_layout()
    plt.show()


def joint_margin_plot(summary, details) -> None:
    labels = [row["side"] for row in summary]
    joint_labels = [
        name.replace("left_", "").replace("right_", "").replace("_joint", "")
        for name in details["left"]["chain"].joint_names
    ]
    mean_margin = np.vstack([details[side]["margin"].mean(dim=0).numpy() for side in labels]) * 100
    min_margin = np.vstack([details[side]["margin"].min(dim=0).values.numpy() for side in labels]) * 100
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.7), constrained_layout=True)
    for ax, matrix, title in zip(
        axes,
        [mean_margin, min_margin],
        ["Mean joint-limit margin", "Worst-case joint-limit margin"],
    ):
        image = ax.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=100, aspect="auto")
        ax.set_title(title)
        ax.set_yticks(np.arange(len(labels)), labels)
        ax.set_xticks(np.arange(len(joint_labels)), joint_labels, rotation=35, ha="right")
        for row_i in range(matrix.shape[0]):
            for col_i in range(matrix.shape[1]):
                ax.text(
                    col_i,
                    row_i,
                    f"{matrix[row_i, col_i]:.0f}%",
                    ha="center",
                    va="center",
                    color="#102a43",
                    fontsize=9,
                )
        ax.set_xlabel("G1 arm joint")
    fig.colorbar(image, ax=axes, shrink=0.85, label="remaining range before limit (%)")
    plt.show()


def assert_pass(summary) -> None:
    for row in summary:
        assert row["success_rate"] >= 0.98, row
        assert row["mean_cm"] <= 1.0, row
        assert row["limit_violation_rad"] <= 1e-6, row

    display(
        HTML(
            """
<div style="font-family:system-ui,-apple-system,Segoe UI,sans-serif;border-radius:12px;background:#edfdf7;border:1px solid #8eedc7;padding:18px 20px;max-width:920px;color:#014d40;margin-top:10px">
  <div style="font-size:26px;font-weight:850">PASS</div>
  <div style="margin-top:5px;font-size:15px">Unitree G1 Colab GPU IK benchmark passed for both arms with zero joint-limit violations.</div>
</div>
"""
        )
    )


ensure_repo()
runtime_card()
details = run_solver()
summary = summarize(details)
summary_cards(summary)
configure_plots()
error_plots(summary, details)
pose_plot(summary, details)
joint_margin_plot(summary, details)
assert_pass(summary)

