import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, Int16
from auvc_pid.pid_loop import *
import numpy as np

class LineNode(Node):
    def __init__(self):
        super().__init__('line_node')

        #heading is in degrees
        self.heading = None
        self.target_heading = 270
        self.thrust = 50

        #publishes surge and sway values to maintain the line
        self.line_pub = self.create_publisher(Float64MultiArray, "/line_commands", 10)
        self.target_direction_sub = self.create_subscription(Int16, "/target_direction", self.target_heading_callback, 10)
        self.heading_sub = self.create_subscription(Int16, "/heading", self.heading_callback, 10)

        # run loop at 20 hz
        self.timer_period = 0.05
        self.timer = self.create_timer(self.timer_period, self.line)
        
        self.get_logger().info(f"going {self.target_heading} degrees")

    def heading_callback(self, msg):
        self.heading = msg.data

    def target_heading_callback(self, msg):
        self.target_heading = msg.data

    def line(self):
        if (self.heading == None):
            return

        #calculate values
        surge_sway = calculate_surge_sway(self.thrust, self.target_heading, self.heading)

        #print statements for debugging
        print(f"current heading: {self.heading}")
        print(f"surge and sway: {surge_sway}")
        
        # Publish the active step's joystick values
        msg = Float64MultiArray()
        msg.data = surge_sway
        self.line_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = LineNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received, shutting down...")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

def calculate_surge_sway(thrust, target_heading, heading):
    theta_degrees = target_heading - heading
    if (theta_degrees > 180):
        theta_degrees = -(360 - theta_degrees)

    theta = np.radians(theta_degrees)
    surge = thrust * np.cos(theta)
    sway = thrust * np.sin(theta)

    return [surge, sway]