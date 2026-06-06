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

## Review Gates

The Colab output should be accepted only when the report clears these gates:

- Runtime gate: CUDA is available and the GPU, PyTorch, and CUDA versions are visible.
- Provenance gate: the source repository, upstream G1 URDF URL, batch size, steps, and fixed seeds are declared in the run.
- Metric gate: mean wrist error stays below 1 cm and success rate stays at or above 98%.
- Limit gate: optimized joints report zero joint-limit violation.
- Visual gate: summary cards, error tails, 3D residuals, and joint-limit margins are reviewed together.
- Scope gate: the report explicitly labels hardware, orientation, collision, latency, and controller behavior as out of scope.

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

Robot model:

- [Unitree `xr_teleoperate`](https://github.com/unitreerobotics/xr_teleoperate)
- [G1 URDF asset folder](https://github.com/unitreerobotics/xr_teleoperate/tree/main/assets/g1)

Runtime:

- [Google Colab FAQ](https://research.google.com/colaboratory/faq.html)
- [PyTorch CUDA notes](https://docs.pytorch.org/docs/stable/notes/cuda.html)
