from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    share = Path(get_package_share_directory("robotic_ultrasound_force_control"))
    config = share / "config" / "digital_twin.yaml"
    project_root = share.resolve().parents[4]
    project_root = next(
        (parent for parent in (project_root,) + tuple(project_root.parents)
         if (parent / "rusim").is_dir()),
        project_root,
    )
    environment = {"RUSIM_PROJECT_ROOT": str(project_root)}
    return LaunchDescription([
        Node(
            package="robotic_ultrasound_force_control",
            executable="mujoco_node",
            name="robotic_ultrasound_mujoco",
            parameters=[str(config)],
            output="screen",
            additional_env=environment,
        ),
        Node(
            package="robotic_ultrasound_force_control",
            executable="controller_node",
            name="robotic_ultrasound_controller",
            parameters=[str(config)],
            output="screen",
            additional_env=environment,
        ),
        Node(
            package="robotic_ultrasound_force_control",
            executable="safety_node",
            name="robotic_ultrasound_safety",
            parameters=[str(config)],
            output="screen",
        ),
    ])
