import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64
import time
from auvc_pid.pid_loop import *
from geometry_msgs.msg import Vector3  

class CircularNode(Node):
    def __init__(self):
        super().__init__('angular_node')
        
        #all angular velocity units are in radians per second
        self.latest_angular_velocity = 0.0
        self.target_angular_velocity = 0.0

        self.latest_centripital_acceleration = 0.0
        self.target_centripital_acceleration = 0.0

        self.target_radius = 2.0 #units in meters

        "publishes yaw, surge, and sway values to maintain the circle"
        self.circular_pub = self.create_publisher(Vector3, '/circle_commands', 10)
        self.imu_sub = self.create_subscription(Imu, "/imu", self.imu_callback, 10)        
        self.radius_sub = self.create_subscription(Float64, "/target_radius", self.radius_callback, 10)    

        self.omega_errors = []
        self.centripetal_errors = []

        # run loop at 20 hz
        self.timer_period = 0.05
        self.timer = self.create_timer(self.timer_period, self.circle)
        
        self.get_logger().info(f"Circling with radius: {self.target_radius} meters")

    def imu_callback(self, msg):
        self.latest_angular_velocity = msg.angular_velocity.z
        self.latest_centripital_acceleration = msg.acceleration.x

    def radius_callback(self, msg):
        self.target_radius = msg.data

    def circle(self):
        omega_error = calculate_omega_error(self.latest_heading, self.target_heading)
        centripital_error = calculate_centripital_error(self.latest_ac, self.target_ac)
        self.omega_errors.append(omega_error)
        
        yaw = calculate_yaw(self.omega_errors, self.timer_period)
        surge = calculate_surge(self.centripetal_errors, self.timer_period)

        #print statements for debugging
        print(f"yaw: {yaw}")
        print(f"omega: {self.latest_omega}")
        #print(f"error: {error}")

        # Publish the active step's joystick values
        msg = Vector3()
        msg.x = surge
        msg.y = sway
        msg.z = yaw
        self.pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = AngularNode()
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

def calculate_centripetal_error(current_ac, target_ac):
    return target_ac - current_ac

def calculate_yaw(errors, dt):
    Kp = 1.1
    Ki = 0.0
    Kd = 0.5
    multiplier = 1

    pid_final = run_pid(errors, dt, Kp, Ki, Kd)
    yaw = pid_final * multiplier

    if (abs(yaw) > 300.0):
        if (yaw > 0):
            return -300.0
        else:
            return 300.0

    return yaw

def calculate_surge(errors, dt):
