# Unitree G1 Colab IK

GPU-backed Colab benchmark for Unitree G1 arm inverse kinematics.

The notebook downloads the G1 `g1_body29_hand14.urdf` from
`unitreerobotics/xr_teleoperate`, extracts the left and right arm chains, and
uses PyTorch to solve a batched tracking-style wrist IK problem. It reports
mean, p95, and max end-effector position error, success rate, joint-limit
violations, and runtime throughput.

This is a hardware-free smoke test for teleoperation and retargeting work:
it checks whether candidate wrist targets can be solved under the actual G1
URDF limits before touching a robot.

## Colab

Open the notebook on a GPU runtime:

https://colab.research.google.com/github/zack-dev-cm/unitree-g1-colab-ik/blob/main/notebooks/unitree_g1_colab_ik_bench.ipynb

The notebook is designed to fail fast if CUDA is not available.

## Notebook Structure

The Colab is organized as a compact data-science report:

- Description and scope: what the benchmark validates, and what it does not.
- Environment: GPU, PyTorch, CUDA, source repository, and fixed runtime inputs.
- Inputs and provenance: the upstream Unitree G1 URDF URL and generated target set.
- Execution: a single parameterized IK workload for both arms.
- Metrics: mean, p95, max error, success rate, throughput, and joint-limit margin.
- Visual checks: summary cards, error plots, 3D target-vs-solved plots, and limit heatmaps.
- Validation: assertive pass/fail criteria plus interpretation and next limitations.
- References: upstream robot assets, Colab/Jupyter documentation, notebook practice, and review workflow links.

Reusable logic stays in `src/unitree_colab_ik`; the notebook should remain an
executable report rather than the only implementation.

## Local Quickstart

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
unitree-g1-ik-bench --device cpu --batch-size 24 --steps 80
pytest
```

Use a GPU when available:

```bash
unitree-g1-ik-bench --device cuda --batch-size 512 --steps 240
```

## What It Tests

- Parses the upstream Unitree G1 URDF without Pinocchio, MuJoCo, or ROS.
- Builds torso-to-palm chains for both arms.
- Samples reachable wrist targets from joint-limit-aware G1 poses.
- Solves batched IK with PyTorch Adam using a bounded joint parameterization.
- Verifies end-effector error and zero joint-limit violations.

## Limits

This is a position-only, hardware-free smoke test. It does not validate wrist
orientation tracking, self-collision, external collision, torque limits,
network latency, controller behavior, or safety for robot operation.

## Upstream Assets

The benchmark downloads the URDF from:

`https://raw.githubusercontent.com/unitreerobotics/xr_teleoperate/main/assets/g1/g1_body29_hand14.urdf`

The URDF remains owned by Unitree Robotics and is not vendored in this repo.

## References

- Unitree `xr_teleoperate`: https://github.com/unitreerobotics/xr_teleoperate
- Google Colab FAQ: https://research.google.com/colaboratory/faq.html
- Project Jupyter documentation: https://docs.jupyter.org/en/latest/
- Ten simple rules for writing and sharing computational analyses in Jupyter Notebooks: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007007
- OpenAI Codex subagents: https://developers.openai.com/codex/concepts/subagents
- OpenAI harness engineering: https://openai.com/index/harness-engineering/
