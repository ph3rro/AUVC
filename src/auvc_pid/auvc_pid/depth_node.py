import rclpy
from rclpy.node import Node
from sensor_msgs.msg import FluidPressure as Pressure
from std_msgs.msg import Float64
from auvc_pid.pid_loop import *

T = 26.6666667 # celsius 
WATER_DENSITY = 1000 * (1- (T+288.9414)/ (508929.2 * (T+68.12963)) * (T-3.9863)**2)
G = 9.80665 
ATMOSPHERIC = 101325.0


class DepthNode(Node):
    def __init__(self):
        super().__init__('depth_node')
        
        self.latest_pressure = None
        self.latest_pressure_time = None
        self.has_new_pressure = False
        self.pressure_timed_out = False
        self.target_depth_range = [0, 1.0] #units in meters
        self.target_depth = abs((self.target_depth_range[1] + self.target_depth_range[0]) / 2)

        '''self.manual_pub publishes the movements so the auv can read them'''
        self.heave_pub = self.create_publisher(Float64, '/current_heave', 10)
        self.depth_pub = self.create_publisher(Float64, '/current_depth', 10)
        self.pressure_sub = self.create_subscription(Pressure, "/pressure", self.pressure_callback, 10)
        self.depth_sub = self.create_subscription(Float64, "/target_depth", self.depth_callback, 10)    
        
        self.integral = 0
        self.prev_error = 0
        self.current_error = 0
        self.last_pid_time = None
        self.pid_initialized = False
        self.latest_heave = 0.0
        self.latest_depth = None

        # run loop at 20 hz
        self.timer_period = 0.05
        self.max_derivative_dt = 1.0
        self.pressure_timeout = 2.0
        self.timer = self.create_timer(self.timer_period, self.goToDepth)
        
        self.get_logger().info(f"approaching depth: {self.target_depth} meters")

    def pressure_callback(self, msg):
        self.latest_pressure = msg.fluid_pressure
        self.latest_pressure_time = self.get_clock().now()
        self.has_new_pressure = True

    def calculate_depth(self, measured_pressure):
        output = (measured_pressure - ATMOSPHERIC) / (G * WATER_DENSITY)
        return output

    def calculate_heave(self, dt):
        Kp = 3.0
        Ki = 0.0
        Kd = 0
        Kf = 0.62

        multiplier = 25.0
        self.integral, pid_raw = run_pid(self.prev_error, self.current_error, dt, Kp, Ki, Kd, Kf, self.integral)

        heave = pid_raw * multiplier

        if (abs(heave) > 300.0):
            if (heave > 0):
                return -300.0
            else:
                return 300.0

        return -heave


    def depth_callback(self, msg):
        if not msg.data == self.target_depth:
            self.integral = 0
            self.prev_error = 0
            self.current_error = 0
            self.last_pid_time = None
            self.pid_initialized = False
        self.target_depth = msg.data
        
        self.get_logger().info(f'New target depth: {self.target_depth:.2f} m')

    def goToDepth(self):
        now = self.get_clock().now()

        if self.latest_pressure_time is None:
            self.publish_outputs()
            return

        pressure_age = (now - self.latest_pressure_time).nanoseconds * 1e-9
        if pressure_age > self.pressure_timeout:
            if not self.pressure_timed_out:
                self.get_logger().warn(
                    f"No pressure update for {pressure_age:.2f} s; commanding neutral heave"
                )
                self.integral = 0
                self.prev_error = 0
                self.current_error = 0
                self.last_pid_time = None
                self.pid_initialized = False
                self.latest_heave = 0.0
                self.pressure_timed_out = True
            self.has_new_pressure = False
            self.publish_outputs()
            return

        if not self.has_new_pressure:
            # Keep the command topic alive without integrating stale measurements.
            self.publish_outputs()
            return

        if self.pressure_timed_out:
            self.get_logger().info("Pressure updates resumed")
            self.pressure_timed_out = False

        measurement_time = self.latest_pressure_time
        skip_derivative = self.last_pid_time is None
        if skip_derivative:
            dt = self.timer_period
        else:
            dt = (measurement_time - self.last_pid_time).nanoseconds * 1e-9
            if dt <= 0.0:
                dt = self.timer_period
                skip_derivative = True
            elif dt > self.max_derivative_dt:
                skip_derivative = True
        self.last_pid_time = measurement_time
        self.has_new_pressure = False
        
        #calculation returns a positive value, so positive depth = down
        depth = self.calculate_depth(self.latest_pressure) 

        error = self.target_depth - depth
        if self.pid_initialized and not skip_derivative:
            self.prev_error = self.current_error
        else:
            # Matching samples make the derivative zero after startup or a long pause.
            self.prev_error = error
            self.pid_initialized = True
        self.current_error = error

        #accounting for the buoyancy of the rov
        offset = 0

        self.latest_heave = self.calculate_heave(dt) - offset
        self.latest_depth = depth

        #print statements for debugging
        print(f"heave: {self.latest_heave}")
        print(f"depth: {depth}")
        print(f"error: {error}")

        self.publish_outputs()

    def publish_outputs(self):
        self.heave_pub.publish(Float64(data=self.latest_heave))
        if self.latest_depth is not None:
            self.depth_pub.publish(Float64(data=self.latest_depth))


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
