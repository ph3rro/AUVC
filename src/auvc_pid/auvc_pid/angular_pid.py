import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64, Int16
import time
from auvc_pid.pid_loop import *
from auvc_pid.calculate_depth import *

class AngularNode(Node):
    def __init__(self):
        super().__init__('angular_node')
        
        #all heading units are in degrees
        self.latest_heading = 0.0
        self.target_heading = 245.0 
        self.latest_angular_velocity = 0.0

        '''self.manual_pub publishes the movements so the auv can read them'''
        self.angular_pub = self.create_publisher(Float64, '/current_torque', 10)
        self.imu_sub = self.create_subscription(Imu, "/imu", self.imu_callback, 10)
        self.heading_sub = self.create_subscription(Int16, "/heading", self.heading_callback, 10)        
        self.theta_sub = self.create_subscription(Int16, "/target_theta", self.theta_callback, 10)    

        self.errors = []

        # run loop at 20 hz
        self.timer_period = 0.05
        self.timer = self.create_timer(self.timer_period, self.turnToHeading)
        
        self.get_logger().info(f"Turning to heading: {self.target_heading} degrees")

    def imu_callback(self, msg):
        self.latest_angular_velocity = msg.angular_velocity.z
        
    def heading_callback(self, msg):
        self.latest_heading = msg.data
    
    def theta_callback(self, msg):
        self.target_heading = msg.data

    #def depth_callback(self, msg):
        #self.target_depth = msg.data
        #self.get_logger().info(f'New target depth: {self.target_depth:.2f} m')

    def turnToHeading(self):
        error = calculate_minimum_heading_error(self.latest_heading, self.target_heading)
        self.errors.append(error)
        
        yaw = calculate_yaw(self.errors, self.timer_period, self.latest_angular_velocity)

        #print statements for debugging
        print(f"yaw: {yaw}")
        print(f"heading: {self.latest_heading}")
        #print(f"error: {error}")

        # Publish the active step's joystick values
        msg = Float64()
        msg.data = yaw
        self.angular_pub.publish(msg)

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

def calculate_minimum_heading_error(current_heading, target_heading):
    base_error = target_heading - current_heading
    if (base_error > 180):
        return current_heading - target_heading
    return base_error

def calculate_yaw(errors, dt, angular_velocity):
    Kp = 1.1
    Ki = 0.0
    Kd = 0.5
    multiplier = 1

    pid_PI = run_pid(errors, dt, Kp, Ki, 0)

    pid_final = pid_PI + (Kd * angular_velocity)
    yaw = pid_final * multiplier

    if (abs(yaw) > 300.0):
        if (yaw > 0):
            return -300.0
        else:
            return 300.0

    return yaw

