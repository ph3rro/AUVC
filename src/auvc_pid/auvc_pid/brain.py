import rclpy
from rclpy.node import Node
from mavros_msgs.msg import ManualControl
from std_msgs.msg import Float64, Float64MultiArray, Bool, String
import time
from auvc_pid.light_controller import *

class BrainNode(Node):
    def __init__(self):
        super().__init__('brain_node')
        self.declare_parameter("qualifier_mode", True)

        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.angular = 0.0

        self.elapsed_time = 0.0
        self.latest_fire = 0.0

        self.distance = 1e5
        self.fire = False
        self.start_sequence = True
        self.found_auv = False

        self.go_to_heading_value = 0.0
        self.bearing = 0.0


        '''self.manual_pub publishes the movements so the auv can read them'''
        self.manual_pub = self.create_publisher(ManualControl, "/manual_control", 10)
        self.light_pub = self.create_publisher(String, "/fire_control", 10)
        self.lights = LightController(self)

        self.yolo_detected_sub = self.create_subscription(Bool, "/yolo_detected", self.yolo_detected_callback, 10)
        self.heave_sub = self.create_subscription(Float64, "/current_heave", self.heave_callback, 10)
        self.angular_sub = self.create_subscription(Float64, "/current_torque", self.angular_callback, 10)
        #self.thrust_sub = self.create_subscription(Float64, "/forward", self.forward_callback, 10)   
        self.circle_sub = self.create_subscription(Float64MultiArray, "/circle_commands", self.circle_callback, 10)
        self.line_sub = self.create_subscription(Float64MultiArray, "/line_commands", self.line_callback, 10)
        #self.pose_sub = self.create_subscription(Float64MultiArray, "/pose", self.pose_callback, 10)
        self.bearing_sub = self.create_subscription(Float64, "/target_bearing", self.bearing_callback, 10)
        self.height_sub = self.create_subscription(Float64, "/target_height", self.height_callback, 10)
        self.distance_sub = self.create_subscription(Float64, "/distance", self.distance_callback, 10)

        # run loop at 20 hz
        self.timer = self.create_timer(0.05, self.manual_control_publisher)
        
    def distance_callback(self, msg):
        self.distance = msg.data

    def heave_callback(self, msg):
        self.z = msg.data
    
    def angular_callback(self, msg):
        self.go_to_heading_value = msg.data

    def forward_callback(self, msg):
        self.x = msg.data

    def line_callback(self, msg):
        self.x = msg.data[0]
        self.y = msg.data[1]

    def yolo_detected_callback(self, msg):
        self.found_auv = msg.data
        if not self.found_auv:
            self.angular = 0.0

    def circle_callback(self, msg):
        self.y = msg.data[0]
        #self.get_logger().info(f"data[1]: {msg.data[1]}")
        self.angular = msg.data[1]

    def manual_control_publisher(self):
        
        if (not self.fire and (self.found_auv and self.distance <= 1.0)):
            msg_lights = String()
            msg_lights.data = "UNDER ONE METER! FIREEEEEEEEEEEEEEE!"
            self.light_pub.publish(msg_lights)
            self.x = -100
            self.lights.full()
            self.fire = True
            self.angular = 0.0
            self.latest_fire = self.elapsed_time
            

        elif(self.fire):
            self.x = -100
            if (self.elapsed_time - self.latest_fire > 0.25):
                self.lights.off()
                self.x = 0
                self.fire = False
        
        if(self.start_sequence and not self.fire):
            self.angular = self.go_to_heading_value
            if(self.elapsed_time >= 3.0):
                self.x = 25

        if(self.found_auv and not(self.distance <= 1.0)):
            self.start_sequence = False
            self.x = 60.0
            self.angular = self.bearing

        elif(not self.start_sequence):
            self.x = 0.0
            self.y = 0.0
            self.angular = 25
        

        
        print(f"r: {self.angular}")
        print(f"z: {self.z}")
        print(f"x: {self.x}")

        msg = ManualControl()
        msg.x = float(self.x)
        msg.y = float(self.y)
        msg.z = float(self.z)
        msg.r = float(self.angular)
        self.manual_pub.publish(msg)

        self.elapsed_time += 0.05

    def bearing_callback(self, msg):
        self.bearing = msg.data

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