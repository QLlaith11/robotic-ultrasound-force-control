import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
if (ROOT / "rusim").is_dir():
    sys.path.insert(0, str(ROOT))


def test_controller_node_entry_points_exist():
    package = Path(__file__).resolve().parents[1]
    setup_text = (package / "setup.py").read_text()
    assert "mujoco_node" in setup_text
    assert "controller_node" in setup_text
    assert "safety_node" in setup_text


def test_ros_topics_are_declared():
    source = Path(__file__).resolve().parents[1]
    text = "\n".join(
        (source / "robotic_ultrasound_force_control" / name).read_text()
        for name in ("controller_node.py", "mujoco_node.py", "safety_node.py")
    )
    for topic in (
        "/tissue/estimated_force",
        "/tissue/true_force",
        "/ultrasound/desired_normal_force",
        "/ultrasound/safe_desired_normal_force",
        "/robot/joint_position",
    ):
        assert topic in text
