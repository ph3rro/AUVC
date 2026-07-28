import rclpy
from rclpy.node import Node
from mavros_msgs.msg import ManualControl, OverrideRCIn
from std_msgs.msg import Float64, Float64MultiArray
import time

class BrainNode(Node):
    def __init__(self):
        super().__init__('brain_node')
        
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.angular = 0.0

        self.distance = None
        self.fire = False
        self.start_sequence = True

        self.go_to_heading_value = 0.0
        

        '''self.manual_pub publishes the movements so the auv can read them'''
        self.manual_pub = self.create_publisher(ManualControl, "/manual_control", 10)
        self.light_pub = self.create_publisher(OverrideRCIn, '/mavros/rc/override', 10)

        self.heave_sub = self.create_subscription(Float64, "/current_heave", self.heave_callback, 10)
        self.angular_sub = self.create_subscription(Float64, "/current_torque", self.angular_callback, 10)
        #self.thrust_sub = self.create_subscription(Float64, "/forward", self.forward_callback, 10)   
        self.circle_sub = self.create_subscription(Float64MultiArray, "/circle_commands", self.circle_callback, 10)
        self.line_sub = self.create_subscription(Float64MultiArray, "/line_commands", self.line_callback, 10)
        #self.pose_sub = self.create_subscription(Float64MultiArray, "/pose", self.pose_callback, 10)
        self.bearing_sub = self.create_subscription(Float64, "/target_bearing", self.bearing_callback, 10)
        self.height_sub = self.create_subscription(Float64, "/target_height", self.height_callback, 10)
        # run loop at 20 hz
        self.timer = self.create_timer(0.05, self.manual_control_publisher)
        
        #self.get_logger().info(f"approaching depth: {self.target_depth} meters")
        #self.get_logger().info(f"facing heading: {self.target_heading} degrees")

    def heave_callback(self, msg):
        self.z = msg.data
    
    def angular_callback(self, msg):
        self.angular = msg.data

    def forward_callback(self, msg):
        self.x = msg.data

    def line_callback(self, msg):
        self.x = msg.data[0]
        self.y = msg.data[1]

    def circle_callback(self, msg):
        self.y = msg.data[0]
        #self.get_logger().info(f"data[1]: {msg.data[1]}")
        self.angular = msg.data[1]

    def manual_control_publisher(self):

        override_msg = OverrideRCIn()
        channels = [65535] * 8  # 65535 means "no change" for unlisted channels

        if (self.found_auv and self.distance <= 1.0):
            self.x = -100
            channels[4] = 1900
            self.get_logger().info(f"Within range! FIREEEEEEEEEEE!")
            self.fire = True

        if(self.fire):
            self.fire = False
            channels[4] = 1100

        override_msg.channels = channels
        self.light_pub.publish(override_msg)

        if(self.start_sequence):
            self.y = self.go_to_heading_value
            if(self.go_to_heading_value <= 1):
                self.x = 25.0

        if(self.found_auv and not(self.distance <= 1.0)):
            self.x = 50.0
            self.angular = self.weird heading thing that aiden will put in
        else:
            self.x = 0.0
            self.y = 0.0
            self.angular = 15

        
        
        msg = ManualControl()
        msg.x = float(self.x)
        msg.y = float(self.y)
        msg.z = float(self.z)
        msg.r = float(self.angular)
        self.manual_pub.publish(msg)

    def bearing_callback(self, msg):
        self.angular = msg.data

    def height_callback(self, msg):
        self.z = msg.data

    def send_neutral_command(self):
        msg = ManualControl()
        msg.x, msg.y, msg.r = 0.0, 0.0, 0.0
        msg.z = 0.0 
        self.manual_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = BrainNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received, shutting down...")
    finally:
        node.send_neutral_command()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()