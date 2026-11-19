import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float64


class SafetyNode(Node):
    def __init__(self):
        super().__init__("robotic_ultrasound_safety")
        self.declare_parameter("lower_force", 3.0)
        self.declare_parameter("upper_force", 15.0)
        self.declare_parameter("hard_force", 15.0)
        self.lower = float(self.get_parameter("lower_force").value)
        self.upper = float(self.get_parameter("upper_force").value)
        self.hard = float(self.get_parameter("hard_force").value)
        self.estimate = 0.0
        self.independent = 0.0
        self.desired = 0.0
        self.create_subscription(Float64, "/tissue/estimated_force", self.estimate_cb, 10)
        self.create_subscription(Float64, "/tissue/independent_force", self.independent_cb, 10)
        self.create_subscription(Float64, "/ultrasound/desired_normal_force", self.desired_cb, 10)
        self.guarded_pub = self.create_publisher(Float64, "/ultrasound/safe_desired_normal_force", 10)
        self.guard_pub = self.create_publisher(Bool, "/ultrasound/safety_guard_active", 10)
        self.create_timer(0.002, self.update)

    def estimate_cb(self, msg):
        self.estimate = float(msg.data)

    def independent_cb(self, msg):
        self.independent = float(msg.data)

    def desired_cb(self, msg):
        self.desired = float(msg.data)

    def update(self):
        guard_force = max(self.estimate, self.independent)
        active = guard_force >= self.upper
        guarded = -25.0 if guard_force >= self.hard else self.desired
        if active:
            guarded = min(guarded, -8.0)
        guarded = max(-25.0, min(25.0, guarded))
        force_msg = Float64()
        force_msg.data = guarded
        guard_msg = Bool()
        guard_msg.data = active
        self.guarded_pub.publish(force_msg)
        self.guard_pub.publish(guard_msg)


def main(args=None):
    rclpy.init(args=args)
    node = SafetyNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
