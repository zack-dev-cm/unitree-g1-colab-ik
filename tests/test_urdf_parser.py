from unitree_colab_ik.urdf import parse_urdf


MINI_URDF = """
<robot name="mini">
  <link name="base"/>
  <link name="shoulder"/>
  <link name="wrist"/>
  <joint name="shoulder_joint" type="revolute">
    <origin xyz="0 0 1" rpy="0 0 0"/>
    <parent link="base"/>
    <child link="shoulder"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" effort="1" velocity="1"/>
  </joint>
  <joint name="wrist_joint" type="fixed">
    <origin xyz="1 0 0" rpy="0 0 0"/>
    <parent link="shoulder"/>
    <child link="wrist"/>
  </joint>
</robot>
"""


def test_parse_urdf_and_chain():
    robot = parse_urdf(MINI_URDF)
    chain = robot.chain("base", "wrist")

    assert robot.name == "mini"
    assert [joint.name for joint in chain] == ["shoulder_joint", "wrist_joint"]
    assert robot.active_joint_names(chain) == ("shoulder_joint",)

