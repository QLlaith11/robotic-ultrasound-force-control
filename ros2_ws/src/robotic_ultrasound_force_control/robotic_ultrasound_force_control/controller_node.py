import os
from pathlib import Path
import sys

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64


def add_project_root():
    root = os.environ.get("RUSIM_PROJECT_ROOT")
    if root:
        sys.path.insert(0, root)
        return
    here = Path(__file__).resolve()
    for parent in (here,) + tuple(here.parents):
        if (parent / "rusim").is_dir():
            sys.path.insert(0, str(parent))
            return


add_project_root()
from rusim.controllers import Controller


class ControllerNode(Node):
    def __init__(self):
        super().__init__("robotic_ultrasound_controller")
        self.declare_parameter("target_force", 10.0)
        self.declare_parameter("controller", "blf_stsmc")
        self.declare_parameter("control_hz", 500.0)
        self.declare_parameter("barrier_margin", 4.0)
        self.declare_parameter("blf_st_k1", 2.0)
        self.declare_parameter("blf_st_k2", 3.5)
        target = float(self.get_parameter("target_force").value)
        name = str(self.get_parameter("controller").value)
        gains = {
            "barrier_margin": float(self.get_parameter("barrier_margin").value),
            "blf_st_k1": float(self.get_parameter("blf_st_k1").value),
            "blf_st_k2": float(self.get_parameter("blf_st_k2").value),
        }
        self.controller = Controller(name, target=target, gains=gains)
        self.estimated_force = target
        self.surface_velocity = 0.0
        hz = float(self.get_parameter("control_hz").value)
        self.dt = 1.0 / hz
        self.create_subscription(Float64, "/tissue/estimated_force", self.force_cb, 10)
        self.create_subscription(Float64, "/tissue/surface_velocity", self.velocity_cb, 10)
        self.command_pub = self.create_publisher(Float64, "/ultrasound/desired_normal_force", 10)
        self.create_timer(self.dt, self.update)

    def force_cb(self, msg):
        self.estimated_force = float(msg.data)

    def velocity_cb(self, msg):
        self.surface_velocity = float(msg.data)

    def update(self):
        command = self.controller.command(
            self.estimated_force, self.dt, self.surface_velocity, safe=True
        )
        msg = Float64()
        msg.data = command
        self.command_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ControllerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
