import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Float64MultiArray
from sensor_msgs.msg import Image
import os, threading, time, cv2, numpy as np
from pupil_apriltags import Detector
from std_msgs.msg import Float64MultiArray, Float64, Int16
import torch
from ultralytics import YOLO

class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')
          
        self.camera_sub = self.create_subscription(Image, "/camera", self.camera_callback, 10)
        self.pose_pub = self.create_publisher(Float64MultiArray, "/pose", 10)

        self.image_size = 256
        self.detector = Detector(families="tag36h11")
        self.latest_detections = []

        self.declare_parameter("weights_path", "")
        self.declare_parameter("yolo_threads", 2)
        self.declare_parameter("yolo_rate", 2.0)
        self.declare_parameter("yolo_conf", 0.25)

        self.weights_path = self.get_parameter("weights_path").value
        self.yolo_threads = int(self.get_parameter("yolo_threads").value)
        self.yolo_period = 1.0 / float(self.get_parameter("yolo_rate").value)
        self.yolo_conf = float(self.get_parameter("yolo_conf").value)

        # leave cores free for yolo and the rest of the stack on the pi
        cv2.setNumThreads(1)

        self.frame_lock = threading.Lock()
        self.latest_frame = None
        self.latest_boxes = []

        if self.weights_path:
            self.yolo_thread = threading.Thread(target=self.yolo_worker, daemon=True)
            self.yolo_thread.start()
        else:
            self.get_logger().warn("no weights_path parameter set, running with apriltags only")

        # run loop at 20 hz
        self.timer_period = 0.05
        self.timer = self.create_timer(self.timer_period, self.pose_publisher)
        #self.img_count = 0
        #self.imgs = np.zeros((200, 480, 640, 3))

    def camera_callback(self,msg):
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(480, 640, 3)

        # center crop to a square, then downscale
        side = min(arr.shape[0], arr.shape[1])
        row = (arr.shape[0] - side) // 2
        col = (arr.shape[1] - side) // 2
        square = arr[row:row + side, col:col + side]
        resized = cv2.resize(square, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        self.latest_detections = self.detector.detect(gray)

        with self.frame_lock:
            self.latest_frame = resized

    def yolo_worker(self):
        # inference runs off the executor thread so a slow frame can't stall the camera callbacks
        os.environ.setdefault("OMP_NUM_THREADS", str(self.yolo_threads))

        torch.set_num_threads(self.yolo_threads)

        try:
            model = YOLO(self.weights_path)
        except Exception as error:
            self.get_logger().error(f"yolo disabled, could not load {self.weights_path}: {error}")
            return

        self.get_logger().info(f"loaded {self.weights_path} on {self.yolo_threads} threads")

        while rclpy.ok():
            with self.frame_lock:
                frame = self.latest_frame
                self.latest_frame = None

            if frame is None:
                time.sleep(self.yolo_period)
                continue

            try:
                results = model.predict(frame, imgsz=self.image_size, conf=self.yolo_conf,
                                        device="cpu", verbose=False)
            except Exception as error:
                self.get_logger().error(f"yolo inference failed: {error}")
                time.sleep(self.yolo_period)
                continue

            boxes = []
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    boxes.append([float(box.cls[0]), float(box.conf[0]), x1, y1, x2, y2])

            with self.frame_lock:
                self.latest_boxes = boxes

            time.sleep(self.yolo_period)

    def pose_publisher(self):
        with self.frame_lock:
            boxes = self.latest_boxes

        # [tag_count, box_count, (tag_id, cx, cy) per tag, (class_id, conf, x1, y1, x2, y2) per box]
        msg = Float64MultiArray()
        msg.data = [float(len(self.latest_detections)), float(len(boxes))]
        for detection in self.latest_detections:
            cx, cy = detection.center
            msg.data.extend([float(detection.tag_id), float(cx), float(cy)])
        for box in boxes:
            msg.data.extend(box)
        self.pose_pub.publish(msg)
        



def main(args=None):
    rclpy.init(args=args)
    node = VisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received, shutting down...")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
