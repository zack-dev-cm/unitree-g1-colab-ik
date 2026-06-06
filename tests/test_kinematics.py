import math

import pytest

torch = pytest.importorskip("torch")

from unitree_colab_ik.kinematics import ChainModel
from unitree_colab_ik.urdf import Joint


def test_single_revolute_joint_forward_position():
    chain = ChainModel(
        (
            Joint(
                name="joint",
                joint_type="revolute",
                parent="base",
                child="link",
                xyz=(0.0, 0.0, 0.0),
                rpy=(0.0, 0.0, 0.0),
                axis=(0.0, 0.0, 1.0),
                lower=-math.pi,
                upper=math.pi,
            ),
            Joint(
                name="tip",
                joint_type="fixed",
                parent="link",
                child="tip",
                xyz=(1.0, 0.0, 0.0),
                rpy=(0.0, 0.0, 0.0),
                axis=(1.0, 0.0, 0.0),
                lower=None,
                upper=None,
            ),
        )
    )

    q = torch.tensor([[math.pi / 2]], dtype=torch.float32)
    position = chain.position(q)[0]

    assert torch.allclose(position, torch.tensor([0.0, 1.0, 0.0]), atol=1e-5)

