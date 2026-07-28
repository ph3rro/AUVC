import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
import rosbag2_py
from rosbag2_py import SequentialWriter, StorageOptions, ConverterOptions, TopicMetadata
from rosbag2_py.serialization import serialize_message

class LoggerNode(Node):
    def __init__(self):
        super().__init__('logger_node')
        
        #Initialize the bag writer
        self.writer = SequentialWriter()
        self.writer.open(
            StorageOptions(uri='mission_bag', storage_id='mcap'),
            ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
        )
        
        #Define the specific topics and their exact message types
        self.topics_to_record = {
            '/auv/depth': 'std_msgs/msg/Float32',
            #'/auv/heading': 'std_msgs/msg/Float32',
        }
        
        # Register each topic schema with the bag writer
        for topic_name, topic_type in self.topics_to_record.items():
            self.writer.create_topic(TopicMetadata(
                name=topic_name,
                type=topic_type,
                serialization_format='cdr'
            ))

        # Create dynamic subscriptions for each target topic
        self.subscriptions = {}
        
        # Subscribe to depth
        self.subscriptions['/auv/depth'] = self.create_subscription(Float64, '/current_depth', lambda msg: self.save_message('/current_depth', msg), 10)
        
        self.get_logger().info('BlueROV Data Recorder initialized for specific telemetry topics.')

    def save_message(self, topic_name, msg):
        """Generic handler to write incoming messages from any tracked topic into the bag."""
        self.writer.write(
            topic_name,
            serialize_message(msg),
            self.get_clock().now().nanoseconds
        )

def main(args=None):
    rclpy.init(args=args)
    node = LoggerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()