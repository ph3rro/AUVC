import rclpy
from rclpy.node import Node
from mavros_msgs.msg import ManualControl
from std_msgs.msg import Float64, Float64MultiArray
import time

class PublisherNode(Node):
    def __init__(self):
        super().__init__('publisher_node')
        
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.angular = 0.0

        '''self.manual_pub publishes the movements so the auv can read them'''
        self.manual_pub = self.create_publisher(ManualControl, "/manual_control", 10)
        self.heave_sub = self.create_subscription(Float64, "/current_heave", self.heave_callback, 10)
        self.angular_sub = self.create_subscription(Float64, "/current_torque", self.angular_callback, 10)
        self.thrust_sub = self.create_subscription(Float64, "/forward", self.forward_callback, 10)   
        self.circle_sub = self.create_subscription(Float64MultiArray, "/circle_commands", self.circle_callback, 10)
        
        self.current_step_index = 0
        self.elapsed_time = 0.0

        # run loop at 20 hz
        self.timer_period = 0.05
        self.timer = self.create_timer(self.timer_period, self.manual_control_publisher)
        
        #self.get_logger().info(f"approaching depth: {self.target_depth} meters")
        #self.get_logger().info(f"facing heading: {self.target_heading} degrees")

    def heave_callback(self, msg):
        self.z = msg.data
    
    def angular_callback(self, msg):
        self.angular = msg.data

    def forward_callback(self, msg):
        self.x = msg.data

    def circle_callback(self, msg):
        self.y = msg.data[0]
        self.get_logger().info(f"data[1]: {msg.data[1]}")
        self.angular = msg.data[1]

    def manual_control_publisher(self):
        if self.elapsed_time > 300.0:
            self.send_neutral_command()
            print("timed out")
            return

        # Publish the active step's joystick values
        msg = ManualControl()

        self.get_logger().info(f"Y: {self.y}")
        self.get_logger().info(f"R: {self.angular}")
        msg.x = float(self.x)
        msg.y = float(self.y)
        msg.z = float(self.z)
        msg.r = float(self.angular)
        self.manual_pub.publish(msg)

        self.elapsed_time += self.timer_period

    def send_neutral_command(self):
        msg = ManualControl()
        msg.x, msg.y, msg.r = 0.0, 0.0, 0.0
        msg.z = 0.0 
        self.manual_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = PublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received, shutting down...")
    finally:
        node.send_neutral_command()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
