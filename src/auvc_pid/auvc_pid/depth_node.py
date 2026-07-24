import rclpy
from rclpy.node import Node
from sensor_msgs.msg import FluidPressure as Pressure
from std_msgs.msg import Float64
from auvc_pid.pid_loop import *

class DepthNode(Node):
    def __init__(self):
        super().__init__('depth_node')
        
        self.latest_pressure = None
        self.target_depth = 2.0 #units in meters

        '''self.manual_pub publishes the movements so the auv can read them'''
        self.heave_pub = self.create_publisher(Float64, '/current_heave', 10)
        self.pressure_sub = self.create_subscription(Pressure, "/pressure", self.pressure_callback, 10)
        self.depth_sub = self.create_subscription(Float64, "/target_depth", self.depth_callback, 10)    
        
        self.errors = []

        # run loop at 20 hz
        self.timer_period = 0.05
        self.timer = self.create_timer(self.timer_period, self.goToDepth)
        
        self.get_logger().info(f"approaching depth: {self.target_depth} meters")

    def pressure_callback(self, msg):
        self.latest_pressure = msg.fluid_pressure

    def calculate_depth(self, measured_pressure):
        output = (measured_pressure - 101325) / (9.81 * 1000)
        return output

    def calculate_heave(errors, dt):
        Kp = 3.6
        Ki = 0.0
        Kd = 1.2

        multiplier = 25.0
        pid_raw = run_pid(errors, dt, Kp, Ki, Kd)

        heave = pid_raw * multiplier

        if (abs(heave) > 400.0):
            if (heave > 0):
                return -400.0
            else:
                return 400.0

        return -heave


    def depth_callback(self, msg):
        self.target_depth = msg.data
        self.get_logger().info(f'New target depth: {self.target_depth:.2f} m')

    def goToDepth(self):
        
        if self.latest_pressure is None:
            print("pressure is none")
            return
        
        #calculation returns a positive value, so positive depth = down
        depth = self.calculate_depth(self.latest_pressure) 

        error = self.target_depth - depth
        self.errors.append(error)

        #accounting for the buoyancy of the rov
        offset = 0

        heave = self.calculate_heave(self.errors, self.timer_period) - offset

        #print statements for debugging
        print(f"heave: {heave}")
        print(f"depth: {depth}")
        #print(f"error: {error}")

        # Publish the active step's joystick values
        msg = Float64()
        msg.data = heave
        self.heave_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = DepthNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received, shutting down...")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
