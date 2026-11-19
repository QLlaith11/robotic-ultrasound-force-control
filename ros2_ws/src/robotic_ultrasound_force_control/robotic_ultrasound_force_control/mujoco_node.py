import os
from pathlib import Path
import sys

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float64, Float64MultiArray


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
import mujoco
import numpy as np
from rusim.mujoco_backend import make_model
from rusim.patient import PatientMotion
from rusim.tissue import TissueModel


class MuJoCoNode(Node):
    def __init__(self):
        super().__init__("robotic_ultrasound_mujoco")
        self.declare_parameter("organ", "liver")
        self.declare_parameter("seed", 7)
        self.declare_parameter("stiffness_scale", 1.0)
        self.declare_parameter("friction_scale", 1.0)
        self.declare_parameter("trajectory_difficulty", 1.15)
        self.declare_parameter("duration", 60.0)
        organ = str(self.get_parameter("organ").value)
        seed = int(self.get_parameter("seed").value)
        stiffness_scale = float(self.get_parameter("stiffness_scale").value)
        friction_scale = float(self.get_parameter("friction_scale").value)
        difficulty = float(self.get_parameter("trajectory_difficulty").value)
        self.duration = float(self.get_parameter("duration").value)
        self.model = make_model()
        self.model.opt.timestep = 0.001
        stiffness_scale = stiffness_scale
        self.data = mujoco.MjData(self.model)
        self.qref = np.array([0.0, 0.5, 0.0, 1.0, 0.0, -1.0, 0.0])
        self.data.qpos[:] = self.qref
        self.data.ctrl[:] = self.qref
        mujoco.mj_forward(self.model, self.data)
        self.site = self.model.site("attachment_site").id
        self.z0 = float(self.data.site_xpos[self.site, 2])
        self.motion = PatientMotion(organ, seed, difficulty=difficulty)
        self.tissue = TissueModel(stiffness=1000.0 * stiffness_scale)
        self.rng = np.random.default_rng(seed + 1000)
        self.normal_command = 10.0
        self.qvel_prev = self.data.qvel.copy()
        self.command_tau = np.zeros(self.model.nv)
        self.steps = 0
        self.create_subscription(Float64, "/ultrasound/safe_desired_normal_force", self.command_cb, 10)
        self.true_pub = self.create_publisher(Float64, "/tissue/true_force", 10)
        self.estimate_pub = self.create_publisher(Float64, "/tissue/estimated_force", 10)
        self.independent_pub = self.create_publisher(Float64, "/tissue/independent_force", 10)
        self.velocity_pub = self.create_publisher(Float64, "/tissue/surface_velocity", 10)
        self.qpos_pub = self.create_publisher(Float64MultiArray, "/robot/joint_position", 10)
        self.true_force = 0.0
        self.create_timer(float(self.model.opt.timestep), self.update)

    def command_cb(self, msg):
        self.normal_command = float(msg.data)

    def update(self):
        if self.data.time >= self.duration:
            self.normal_command = 0.0
        t = float(self.data.time)
        base = self.z0 - 0.010 + 0.008 * min(t, 1.0)
        surface = base + self.motion.displacement(t)
        surface_velocity = self.motion.velocity(t)
        mujoco.mj_forward(self.model, self.data)
        position = float(self.data.site_xpos[self.site, 2])
        jacobian = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacobian, None, self.site)
        penetration = max(0.0, position - surface)
        raw_force = self.tissue.normal_force(penetration, surface_velocity)
        self.true_force = min(raw_force, 15.0)
        qacc = (self.data.qvel - self.qvel_prev) / float(self.model.opt.timestep)
        self.qvel_prev = self.data.qvel.copy()
        tissue_tau = -jacobian[2] * self.true_force
        mass = np.zeros((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.model, mass, self.data.qM)
        tau_sensor = (
            mass @ qacc
            + self.data.qfrc_bias
            - tissue_tau
            + self.rng.normal(0.0, 0.04, self.model.nv)
        )
        dynamic_rhs = mass @ qacc + self.data.qfrc_bias
        rhs = dynamic_rhs - tau_sensor
        estimate = float(-np.dot(jacobian[2], rhs) / (np.dot(jacobian[2], jacobian[2]) + 1e-9))
        independent = float(self.true_force + self.rng.normal(0.0, 0.12))
        if self.steps % 2 == 0:
            tau = jacobian[2] * self.normal_command
            tau += self.data.qfrc_bias - 8.0 * (self.data.qpos - self.qref) - 2.0 * self.data.qvel
            limits = np.array([176, 176, 110, 110, 110, 40, 40])
            self.command_tau = np.clip(tau, -limits, limits)
            self.data.ctrl[:] = self.qref
        self.data.qfrc_applied[:] = tissue_tau + self.command_tau
        mujoco.mj_step(self.model, self.data)
        self.publish(Float64, self.true_pub, self.true_force)
        self.publish(Float64, self.estimate_pub, estimate)
        self.publish(Float64, self.independent_pub, independent)
        self.publish(Float64, self.velocity_pub, surface_velocity)
        joint_msg = Float64MultiArray()
        joint_msg.data = [float(value) for value in self.data.qpos]
        self.qpos_pub.publish(joint_msg)
        self.steps += 1

    @staticmethod
    def publish(message_type, publisher, value):
        msg = message_type()
        msg.data = float(value)
        publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = MuJoCoNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
