import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, Float64, Int16
from auvc_pid.pid_loop import *
import numpy as np

class CircularNode(Node):
    def __init__(self):
        super().__init__('angular_node')

        self.desired_period = 4.0 #seconds
        self.target_radius = 2.0 #meters

        self.latest_heading = None
        self.previous_heading = None

        #all angular velocity units are in radians per second
        self.latest_omega = 0.0
        self.target_omega = -2 * np.pi / self.desired_period

        #publishes yaw, surge, and sway values to maintain the circle
        self.circular_pub = self.create_publisher(Float64MultiArray, "/circle_commands", 10)
        self.radius_sub = self.create_subscription(Float64, "/target_radius", self.radius_callback, 10)
        self.heading_sub = self.create_subscription(Int16, "/heading", self.heading_callback, 10)

        self.omega_errors = []

        # run loop at 20 hz
        self.timer_period = 0.05
        self.timer = self.create_timer(self.timer_period, self.circle)
        
        self.get_logger().info(f"Circling with radius: {self.target_radius} meters")

    def heading_callback(self, msg):
        self.previous_heading = self.latest_heading
        self.latest_heading = msg.data

    def radius_callback(self, msg):
        self.target_radius = msg.data

    def circle(self):
        if ((self.latest_heading == None) or (self.previous_heading == None)):
            return

        self.latest_omega = calculate_omega(self.latest_heading, self.previous_heading, self.timer_period)
        #update errors
        omega_error = calculate_omega_error(self.latest_omega, self.target_omega)
        self.omega_errors.append(omega_error)

        #calculate values
        yaw = calculate_yaw(self.omega_errors, self.timer_period)
        sway = 100

        #print statements for debugging
        print(f"current heading: {self.latest_heading}")
        print(f"previous heading: {self.previous_heading}")
        print(f"yaw: {yaw}")
        print(f"omega: {self.latest_omega}")
        print(f"error: {omega_error}")

        # Publish the active step's joystick values
        msg = Float64MultiArray()
        msg.data = [sway, yaw]
        self.circular_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = CircularNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received, shutting down...")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

def calculate_omega(current_theta, previous_theta, dt):
    omega_degrees = (current_theta - previous_theta) / dt
    return omega_degrees * 2 * np.pi / 360

def calculate_omega_error(current_omega, target_omega):
    return target_omega - current_omega

def calculate_yaw(errors, dt):
    Kp = 20.0
    Ki = 0.0
    Kd = 0.0

    yaw = run_pid(errors, dt, Kp, Ki, Kd)
    return output_cap(yaw, 300)

def output_cap(output, maximum):
    if (abs(output) > maximum):
        if (output > 0):
            return maximum
        return -maximum
    return output