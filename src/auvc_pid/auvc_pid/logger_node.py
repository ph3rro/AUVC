import rclpy
from rclpy.node import Node
from rclpy.serialization import serialize_message, deserialize_message
from std_msgs.msg import Float64, Int16, String
import rosbag2_py
from rosbag2_py import (
    SequentialWriter, SequentialReader, StorageOptions,
    ConverterOptions, TopicMetadata
)
from datetime import datetime
from mavros_msgs.msg import ManualControl

# Maps the string type names used when registering bag topics to the actual
# message classes needed to deserialize them back out for the text dump.
# Add an entry here any time you add a new topic to topics_to_record.
MSG_TYPE_MAP = {
    'std_msgs/msg/Float64': Float64,
    'std_msgs/msg/Int16': Int16,
    'mavros_msgs/msg/ManualControl': ManualControl,
    'std_msgs/msg/String': String
}


class LoggerNode(Node):
    def __init__(self):
        super().__init__('logger_node')

        # Timestamped name so repeated runs never collide with a leftover
        # bag directory from a previous attempt (rosbag2 refuses to
        # overwrite an existing bag folder). Stored on self so we can
        # reopen it for reading later.
        self.bag_uri = f'mission_bag_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        self.writer = SequentialWriter()
        self.writer.open(
            StorageOptions(uri=self.bag_uri, storage_id='mcap'),
            ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
        )
        self.get_logger().info(f'Recording to bag: {self.bag_uri}')

        # Define the specific topics and their exact message types.
        # NOTE: verify '/current_depth' actually publishes Float64 -- if it
        # publishes Float32 instead, change this AND the subscription/import
        # below to match. Whatever type you declare here MUST match what
        # you actually serialize and write, or the bag will be unreadable.
        self.topics_to_record = {
            '/auv/depth': 'std_msgs/msg/Float64',
            '/auv/heading': 'std_msgs/msg/Int16',
            '/auv/manualControl': 'mavros_msgs/msg/ManualControl',
            '/auv/fire': 'std_msgs/msg/String'
        }

        # Register each topic schema with the bag writer.
        # This rosbag2_py version requires an integer `id` per topic --
        # just assign sequential IDs starting at 0.
        for topic_id, (topic_name, topic_type) in enumerate(self.topics_to_record.items()):
            self.writer.create_topic(TopicMetadata(
                id=topic_id,
                name=topic_name,
                type=topic_type,
                serialization_format='cdr'
            ))

        # Create dynamic subscriptions for each target topic.
        # Renamed from self.subscriptions -- Node already defines a
        # read-only `subscriptions` property, so assigning to it raises
        # AttributeError.
        self.topic_subs = {}

        # Subscribe to the live ROS topic '/current_depth', but log it into
        # the bag under the registered bag-topic-name '/auv/depth' (must
        # match a key in topics_to_record above).
        self.topic_subs['/auv/depth'] = self.create_subscription(Float64,'/current_depth', lambda msg: self.save_message('/auv/depth', msg), 10)
        self.topic_subs['/auv/heading'] = self.create_subscription(Int16,'/heading', lambda msg: self.save_message('/auv/heading', msg), 10)
        self.topic_subs['/auv/manualControl'] = self.create_subscription(ManualControl,'/manual_control', lambda msg: self.save_message('/auv/manualControl', msg), 10)
        self.topic_subs['/auv/fire'] = self.create_subscription(String,'/fire_control', lambda msg: self.save_message('/auv/fire', msg), 10)

        self.get_logger().info('BlueROV Data Recorder initialized for specific telemetry topics.')

    def save_message(self, topic_name, msg):
        """Generic handler to write incoming messages from any tracked topic into the bag."""
        self.writer.write(
            topic_name,
            serialize_message(msg),
            self.get_clock().now().nanoseconds
        )

    def dump_bag_to_text(self, output_path=None):
        """Close the bag writer, then read every recorded message back out
        and write it to a plain text file for quick viewing."""
        # Release the writer so the bag is finalized on disk before we try
        # to open it for reading. There's no writer.close() exposed in this
        # rosbag2_py version -- `del` (triggering the destructor) is the
        # documented way to flush and finalize the bag's metadata.
        del self.writer

        if output_path is None:
            output_path = f'{self.bag_uri}.txt'

        reader = SequentialReader()
        reader.open(
            StorageOptions(uri=self.bag_uri, storage_id='mcap'),
            ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
        )

        with open(output_path, 'w') as f:
            while reader.has_next():
                topic, data, timestamp = reader.read_next()
                type_str = self.topics_to_record.get(topic)
                msg_class = MSG_TYPE_MAP.get(type_str)
                if msg_class is None:
                    continue  # unmapped/unknown type -- add it to MSG_TYPE_MAP above
                msg = deserialize_message(data, msg_class)
                f.write(f'{timestamp}\t{topic}\t{str(msg)}\n')

        self.get_logger().info(f'Wrote bag contents to {output_path}')


def main(args=None):
    rclpy.init(args=args)
    node = LoggerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.dump_bag_to_text()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()