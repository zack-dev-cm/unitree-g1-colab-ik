from __future__ import annotations

from dataclasses import dataclass
import time

from .kinematics import ChainModel
from .urdf import DEFAULT_G1_URDF_URL, RobotModel, download_urdf, load_urdf_path

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


CHAIN_SPECS = {
    "left": ("torso_link", "left_hand_palm_link"),
    "right": ("torso_link", "right_hand_palm_link"),
}


@dataclass(frozen=True)
class BenchmarkResult:
    side: str
    device: str
    batch_size: int
    steps: int
    mean_error_m: float
    p95_error_m: float
    max_error_m: float
    success_rate: float
    limit_violation_rad: float
    elapsed_s: float
    targets_per_s: float
    joint_names: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "side": self.side,
            "device": self.device,
            "batch_size": self.batch_size,
            "steps": self.steps,
            "mean_error_m": self.mean_error_m,
            "p95_error_m": self.p95_error_m,
            "max_error_m": self.max_error_m,
            "success_rate": self.success_rate,
            "limit_violation_rad": self.limit_violation_rad,
            "elapsed_s": self.elapsed_s,
            "targets_per_s": self.targets_per_s,
            "joint_names": self.joint_names,
        }


def resolve_device(requested: str = "auto") -> str:
    if torch is None:
        raise ModuleNotFoundError("unitree_colab_ik.benchmark requires torch")
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return requested


def load_robot(urdf_path: str | None = None, urdf_url: str = DEFAULT_G1_URDF_URL) -> RobotModel:
    if urdf_path:
        return load_urdf_path(urdf_path)
    return download_urdf(urdf_url)


def build_chain(robot: RobotModel, side: str, device: str = "cpu") -> ChainModel:
    if side not in CHAIN_SPECS:
        raise KeyError(f"Unknown side {side!r}; expected one of {sorted(CHAIN_SPECS)}")
    base, tip = CHAIN_SPECS[side]
    return ChainModel(robot.chain(base, tip), device=device)


def solve_tracking_ik(
    chain: ChainModel,
    *,
    batch_size: int = 256,
    steps: int = 220,
    lr: float = 0.08,
    seed: int = 0,
    success_threshold_m: float = 0.02,
):
    if torch is None:
        raise ModuleNotFoundError("unitree_colab_ik.benchmark requires torch")

    target_q = chain.sample_centered(batch_size, span=0.42, seed=seed)
    target_positions = chain.position(target_q).detach()

    generator = torch.Generator(device=chain.device)
    generator.manual_seed(seed + 10_000)
    initial_q = chain.clamp(
        target_q
        + torch.randn(target_q.shape, generator=generator, device=chain.device, dtype=target_q.dtype)
        * chain.half_range
        * 0.08
    )
    u = chain.unconstrained_from_bounded(initial_q).detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([u], lr=lr)

    if chain.device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        q = chain.bounded_from_unconstrained(u)
        positions = chain.position(q)
        position_loss = torch.mean((positions - target_positions) ** 2)
        motion_loss = torch.mean((q - initial_q) ** 2) * 1e-5
        loss = position_loss + motion_loss
        loss.backward()
        optimizer.step()

    if chain.device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    with torch.no_grad():
        q = chain.bounded_from_unconstrained(u)
        final_positions = chain.position(q)
        errors = torch.linalg.norm(final_positions - target_positions, dim=1)
        lower_violation = torch.clamp(chain.lower - q, min=0.0).max()
        upper_violation = torch.clamp(q - chain.upper, min=0.0).max()
        violation = torch.maximum(lower_violation, upper_violation)
        success = (errors <= success_threshold_m).float().mean()

    return {
        "q": q.detach(),
        "target_positions": target_positions.detach(),
        "final_positions": final_positions.detach(),
        "errors": errors.detach(),
        "elapsed_s": elapsed,
        "success_rate": float(success.item()),
        "limit_violation_rad": float(violation.item()),
    }


def run_benchmark(
    *,
    sides: tuple[str, ...] = ("left", "right"),
    batch_size: int = 256,
    steps: int = 220,
    device: str = "auto",
    urdf_path: str | None = None,
    urdf_url: str = DEFAULT_G1_URDF_URL,
    success_threshold_m: float = 0.02,
) -> list[BenchmarkResult]:
    if torch is None:
        raise ModuleNotFoundError("unitree_colab_ik.benchmark requires torch")

    resolved = resolve_device(device)
    robot = load_robot(urdf_path=urdf_path, urdf_url=urdf_url)
    results: list[BenchmarkResult] = []
    for index, side in enumerate(sides):
        chain = build_chain(robot, side, device=resolved)
        solved = solve_tracking_ik(
            chain,
            batch_size=batch_size,
            steps=steps,
            seed=1234 + index,
            success_threshold_m=success_threshold_m,
        )
        errors = solved["errors"]
        elapsed = float(solved["elapsed_s"])
        results.append(
            BenchmarkResult(
                side=side,
                device=resolved,
                batch_size=batch_size,
                steps=steps,
                mean_error_m=float(errors.mean().item()),
                p95_error_m=float(torch.quantile(errors, 0.95).item()),
                max_error_m=float(errors.max().item()),
                success_rate=float(solved["success_rate"]),
                limit_violation_rad=float(solved["limit_violation_rad"]),
                elapsed_s=elapsed,
                targets_per_s=batch_size / max(elapsed, 1e-9),
                joint_names=chain.joint_names,
            )
        )
    return results
