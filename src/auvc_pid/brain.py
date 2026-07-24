#brain

import threading
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64


class brain_node(Node):

    def __init__(self):
        super().__init__("brain_node")

        # ==========================
        # Current State Variables
        # ==========================
        self.current_heading = 0.0
        self.current_depth = 0.0

        self.heading_received = False
        self.depth_received = False

        # ==========================
        # Target Setpoints
        # ==========================
        self.target_heading = 0.0
        self.target_depth = 0.0

        # Desired dive depth (meters)
        self.dive_depth = 2.0

        # ==========================
        # Subscribers
        # ==========================
        self.heading_sub = self.create_subscription(
            Float64,
            "/heading",
            self.heading_callback,
            10
        )

        self.depth_sub = self.create_subscription(
            Float64,
            "/depth",
            self.depth_callback,
            10
        )

        # ==========================
        # Publishers
        # ==========================
        self.target_heading_pub = self.create_publisher(
            Float64,
            "/target_heading",
            10
        )

        self.target_depth_pub = self.create_publisher(
            Float64,
            "/target_depth",
            10
        )

        # Publish targets continuously
        self.timer = self.create_timer(
            0.1,
            self.publish_targets
        )

        # Keyboard thread
        keyboard = threading.Thread(
            target=self.keyboard_listener,
            daemon=True
        )
        keyboard.start()

        self.get_logger().info("================================")
        self.get_logger().info(" Brain Node Started")
        self.get_logger().info(" Waiting for sensor data...")
        self.get_logger().info(" Press ENTER to execute mission.")
        self.get_logger().info("================================")

    # =====================================================
    # Callbacks
    # =====================================================

    def heading_callback(self, msg):
        self.current_heading = msg.data
        self.heading_received = True

    def depth_callback(self, msg):
        self.current_depth = msg.data
        self.depth_received = True

    # =====================================================
    # Publish desired targets
    # =====================================================

    def publish_targets(self):

        if self.heading_received:
            msg = Float64()
            msg.data = self.target_heading
            self.target_heading_pub.publish(msg)

        if self.depth_received:
            msg = Float64()
            msg.data = self.target_depth
            self.target_depth_pub.publish(msg)

    # =====================================================
    # Keyboard listener
    # =====================================================

    def keyboard_listener(self):

        while rclpy.ok():

            input()

            if not self.heading_received:
                print("Waiting for heading...")
                continue

            if not self.depth_received:
                print("Waiting for depth...")
                continue

            # Compute new heading
            self.target_heading = (
                self.current_heading + 180.0
            ) % 360.0

            # Dive to desired depth
            self.target_depth = self.dive_depth

            print("\n====================================")
            print("MISSION STARTED")
            print("------------------------------------")
            print(f"Current Heading : {self.current_heading:.2f}°")
            print(f"Target Heading  : {self.target_heading:.2f}°")
            print("")
            print(f"Current Depth   : {self.current_depth:.2f} m")
            print(f"Target Depth    : {self.target_depth:.2f} m")
            print("====================================\n")

            self.get_logger().info(
                f"Heading Target = {self.target_heading:.2f}"
            )

            self.get_logger().info(
                f"Depth Target = {self.target_depth:.2f}"
            )


# ==========================================================
# Main
# ==========================================================

def main(args=None):

    rclpy.init(args=args)

    node = brain_node()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
