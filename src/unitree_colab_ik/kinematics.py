from __future__ import annotations

from dataclasses import dataclass
import math

from .urdf import Joint

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - exercised in minimal local envs.
    torch = None  # type: ignore[assignment]


@dataclass
class ChainModel:
    joints: tuple[Joint, ...]
    device: str = "cpu"
    dtype: object | None = None

    def __post_init__(self) -> None:
        if torch is None:
            raise ModuleNotFoundError("unitree_colab_ik.kinematics requires torch")
        dtype = self.dtype or torch.float32
        self.dtype = dtype
        self.active_joints = tuple(joint for joint in self.joints if joint.joint_type in {"revolute", "continuous"})
        self.joint_names = tuple(joint.name for joint in self.active_joints)
        self.lower = torch.tensor(
            [joint.lower if joint.lower is not None else -math.pi for joint in self.active_joints],
            device=self.device,
            dtype=dtype,
        )
        self.upper = torch.tensor(
            [joint.upper if joint.upper is not None else math.pi for joint in self.active_joints],
            device=self.device,
            dtype=dtype,
        )
        self.mid = (self.lower + self.upper) * 0.5
        self.half_range = (self.upper - self.lower) * 0.5
        self.home = self.clamp(torch.zeros(len(self.active_joints), device=self.device, dtype=dtype))
        if torch.any(self.upper <= self.lower):
            raise ValueError("All active joints must have upper > lower")

    @property
    def dof(self) -> int:
        return len(self.active_joints)

    def clamp(self, q):
        return torch.minimum(torch.maximum(q, self.lower), self.upper)

    def sample_centered(self, batch_size: int, span: float = 0.45, seed: int = 0):
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)
        noise = (torch.rand((batch_size, self.dof), generator=generator, device=self.device, dtype=self.dtype) - 0.5) * 2.0
        return self.clamp(self.home + noise * self.half_range * span)

    def bounded_from_unconstrained(self, u):
        return self.mid + self.half_range * torch.tanh(u)

    def unconstrained_from_bounded(self, q):
        normalized = (q - self.mid) / self.half_range
        normalized = torch.clamp(normalized, -0.999, 0.999)
        return 0.5 * torch.log((1.0 + normalized) / (1.0 - normalized))

    def forward(self, q):
        if q.ndim == 1:
            q = q.unsqueeze(0)
        if q.shape[-1] != self.dof:
            raise ValueError(f"Expected q with {self.dof} columns, got shape {tuple(q.shape)}")

        batch = q.shape[0]
        transform = _identity(batch, self.device, self.dtype)
        active_index = 0
        for joint in self.joints:
            transform = transform @ _origin_transform(joint.xyz, joint.rpy, batch, self.device, self.dtype)
            if joint.joint_type in {"revolute", "continuous"}:
                transform = transform @ _axis_rotation(joint.axis, q[:, active_index], self.device, self.dtype)
                active_index += 1
            elif joint.joint_type != "fixed":
                raise NotImplementedError(f"Unsupported joint type: {joint.joint_type}")
        return transform

    def position(self, q):
        return self.forward(q)[:, :3, 3]


def _identity(batch: int, device: str, dtype):
    matrix = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(batch, 1, 1)
    return matrix


def _origin_transform(xyz: tuple[float, float, float], rpy: tuple[float, float, float], batch: int, device: str, dtype):
    transform = _identity(batch, device, dtype)
    transform[:, :3, :3] = _rpy_matrix(rpy, device, dtype).unsqueeze(0)
    transform[:, :3, 3] = torch.tensor(xyz, device=device, dtype=dtype)
    return transform


def _rpy_matrix(rpy: tuple[float, float, float], device: str, dtype):
    roll, pitch, yaw = (torch.tensor(value, device=device, dtype=dtype) for value in rpy)
    cr, sr = torch.cos(roll), torch.sin(roll)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    return torch.stack(
        [
            torch.stack([cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr]),
            torch.stack([sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr]),
            torch.stack([-sp, cp * sr, cp * cr]),
        ]
    )


def _axis_rotation(axis: tuple[float, float, float], theta, device: str, dtype):
    axis_tensor = torch.tensor(axis, device=device, dtype=dtype)
    axis_tensor = axis_tensor / torch.linalg.norm(axis_tensor).clamp_min(1e-12)
    x, y, z = axis_tensor
    c = torch.cos(theta)
    s = torch.sin(theta)
    one_c = 1.0 - c

    batch = theta.shape[0]
    rot = torch.zeros((batch, 4, 4), device=device, dtype=dtype)
    rot[:, 0, 0] = c + x * x * one_c
    rot[:, 0, 1] = x * y * one_c - z * s
    rot[:, 0, 2] = x * z * one_c + y * s
    rot[:, 1, 0] = y * x * one_c + z * s
    rot[:, 1, 1] = c + y * y * one_c
    rot[:, 1, 2] = y * z * one_c - x * s
    rot[:, 2, 0] = z * x * one_c - y * s
    rot[:, 2, 1] = z * y * one_c + x * s
    rot[:, 2, 2] = c + z * z * one_c
    rot[:, 3, 3] = 1.0
    return rot

