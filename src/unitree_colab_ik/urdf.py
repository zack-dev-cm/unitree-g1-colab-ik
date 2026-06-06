from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.request import urlopen
import xml.etree.ElementTree as ET

DEFAULT_G1_URDF_URL = (
    "https://raw.githubusercontent.com/unitreerobotics/xr_teleoperate/main/"
    "assets/g1/g1_body29_hand14.urdf"
)


@dataclass(frozen=True)
class Joint:
    name: str
    joint_type: str
    parent: str
    child: str
    xyz: tuple[float, float, float]
    rpy: tuple[float, float, float]
    axis: tuple[float, float, float]
    lower: float | None
    upper: float | None


@dataclass(frozen=True)
class RobotModel:
    name: str
    links: frozenset[str]
    joints: tuple[Joint, ...]

    def __post_init__(self) -> None:
        children = [joint.child for joint in self.joints]
        if len(children) != len(set(children)):
            duplicates = sorted({child for child in children if children.count(child) > 1})
            raise ValueError(f"URDF is not a tree; duplicate child links: {duplicates}")

    @property
    def joint_by_child(self) -> dict[str, Joint]:
        return {joint.child: joint for joint in self.joints}

    def chain(self, base_link: str, tip_link: str) -> tuple[Joint, ...]:
        if base_link not in self.links:
            raise KeyError(f"Unknown base link: {base_link}")
        if tip_link not in self.links:
            raise KeyError(f"Unknown tip link: {tip_link}")

        by_child = self.joint_by_child
        chain: list[Joint] = []
        cursor = tip_link
        while cursor != base_link:
            joint = by_child.get(cursor)
            if joint is None:
                raise ValueError(f"No path from {base_link!r} to {tip_link!r}")
            chain.append(joint)
            cursor = joint.parent
        chain.reverse()
        return tuple(chain)

    def active_joint_names(self, chain: Iterable[Joint]) -> tuple[str, ...]:
        return tuple(joint.name for joint in chain if joint.joint_type in {"revolute", "continuous"})


def parse_urdf(text: str) -> RobotModel:
    root = ET.fromstring(text)
    if root.tag != "robot":
        raise ValueError("Expected a URDF <robot> root")

    links = frozenset(link.attrib["name"] for link in root.findall("link"))
    joints = tuple(_parse_joint(node) for node in root.findall("joint"))
    return RobotModel(name=root.attrib.get("name", "robot"), links=links, joints=joints)


def load_urdf_path(path: str | Path) -> RobotModel:
    return parse_urdf(Path(path).read_text(encoding="utf-8"))


def download_urdf(url: str = DEFAULT_G1_URDF_URL, timeout: float = 30.0) -> RobotModel:
    with urlopen(url, timeout=timeout) as response:
        text = response.read().decode("utf-8")
    return parse_urdf(text)


def _parse_joint(node: ET.Element) -> Joint:
    origin = node.find("origin")
    axis = node.find("axis")
    limit = node.find("limit")
    parent = node.find("parent")
    child = node.find("child")

    if parent is None or child is None:
        raise ValueError(f"Joint {node.attrib.get('name', '<unnamed>')} is missing parent or child")

    return Joint(
        name=node.attrib["name"],
        joint_type=node.attrib.get("type", "fixed"),
        parent=parent.attrib["link"],
        child=child.attrib["link"],
        xyz=_triplet(origin.attrib.get("xyz", "0 0 0") if origin is not None else "0 0 0"),
        rpy=_triplet(origin.attrib.get("rpy", "0 0 0") if origin is not None else "0 0 0"),
        axis=_triplet(axis.attrib.get("xyz", "1 0 0") if axis is not None else "1 0 0"),
        lower=float(limit.attrib["lower"]) if limit is not None and "lower" in limit.attrib else None,
        upper=float(limit.attrib["upper"]) if limit is not None and "upper" in limit.attrib else None,
    )


def _triplet(raw: str) -> tuple[float, float, float]:
    values = tuple(float(part) for part in raw.split())
    if len(values) != 3:
        raise ValueError(f"Expected 3 values, got {raw!r}")
    return values

