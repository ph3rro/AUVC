import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64MultiArray, Float64
import time
from auvc_pid.pid_loop import *
import numpy as np

class CircularNode(Node):
    def __init__(self):
        super().__init__('angular_node')

        self.desired_period = 4.0 #seconds
        self.target_radius = 2.0 #meters

        #all angular velocity units are in radians per second
        self.latest_angular_velocity = 0.0
        self.target_angular_velocity = 2 * np.pi / self.desired_period

        self.latest_centripital_acceleration = 0.0
        self.target_centripital_acceleration = (self.target_angular_velocity ** 2) * self.target_radius

        #publishes yaw, surge, and sway values to maintain the circle
        self.circular_pub = self.create_publisher(Float64MultiArray, '/circle_commands', 10)
        self.imu_sub = self.create_subscription(Imu, "/imu", self.imu_callback, 10)        
        self.radius_sub = self.create_subscription(Float64, "/target_radius", self.radius_callback, 10)    

        self.omega_errors = []

        # run loop at 20 hz
        self.timer_period = 0.05
        self.timer = self.create_timer(self.timer_period, self.circle)
        
        self.get_logger().info(f"Circling with radius: {self.target_radius} meters")

    def imu_callback(self, msg):
        self.latest_angular_velocity = msg.angular_velocity.z

    def radius_callback(self, msg):
        self.target_radius = msg.data

    def circle(self):
        #update errors
        omega_error = calculate_omega_error(self.latest_angular_velocity, self.target_angular_velocity)
        self.omega_errors.append(omega_error)

        #calculate values
        yaw = -calculate_yaw(self.omega_errors, self.timer_period)
        sway = 75

        #print statements for debugging
        print(f"yaw: {yaw}")
        print(f"omega: {self.latest_angular_velocity}")
        #print(f"error: {error}")

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

def calculate_omega_error(current_omega, target_omega):
    return target_omega - current_omega

def calculate_yaw(errors, dt):
    Kp = 1.1
    Ki = 0.0
    Kd = 0.5
    multiplier = 1

    pid_final = run_pid(errors, dt, Kp, Ki, Kd)
    yaw = pid_final * multiplier

    return output_cap(yaw, 300)

def output_cap(output, maximum):
    if (abs(output) > maximum):
        if (output > 0):
            return maximum
        return -maximum
    return output