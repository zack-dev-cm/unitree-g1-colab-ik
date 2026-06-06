import pytest

torch = pytest.importorskip("torch")

from unitree_colab_ik.benchmark import build_chain, run_benchmark
from unitree_colab_ik.urdf import load_urdf_path


LOCAL_G1_URDF = "/home/z/Github/xr_teleoperate/assets/g1/g1_body29_hand14.urdf"


def test_g1_arm_chain_has_expected_dof():
    robot = load_urdf_path(LOCAL_G1_URDF)
    left = build_chain(robot, "left")
    right = build_chain(robot, "right")

    assert left.dof == 7
    assert right.dof == 7
    assert left.joint_names[0] == "left_shoulder_pitch_joint"
    assert right.joint_names[-1] == "right_wrist_yaw_joint"


def test_small_cpu_benchmark_converges():
    results = run_benchmark(
        sides=("left",),
        batch_size=12,
        steps=70,
        device="cpu",
        urdf_path=LOCAL_G1_URDF,
        success_threshold_m=0.03,
    )

    assert len(results) == 1
    assert results[0].success_rate >= 0.90
    assert results[0].mean_error_m < 0.02
    assert results[0].limit_violation_rad <= 1e-6

