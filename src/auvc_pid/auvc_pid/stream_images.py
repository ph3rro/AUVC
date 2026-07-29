import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Float64MultiArray
from sensor_msgs.msg import Image
import os, threading, time, math, cv2, numpy as np
from pupil_apriltags import Detector
from std_msgs.msg import Float64MultiArray, Float64, Int16, Bool
from sensor_msgs.msg import FluidPressure
from auvc_pid.pid_loop import run_pid
try:
    import torch
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    # torch and ultralytics are not installed on the pi yet, so the node stays usable
    # on apriltags alone until they are
    YOLO_AVAILABLE = False

# True: use the highest-confidence YOLO box for controller targets.
# False: use the selected AprilTag for controller targets.
USING_YOLO = True

# real world height in meters of each yolo class, used to turn a box height into a range.
# a class that is missing here gets a range of nan, so fill this in per competition prop.
# e.g. {0: 0.30, 1: 0.60}
KNOWN_HEIGHTS_M = {0: 0.254}

T = 26.6666667 # celsius 
WATER_DENSITY = 1000 * (1- (T+288.9414)/ (508929.2 * (T+68.12963)) * (T-3.9863)**2)
G = 9.80665 
ATMOSPHERIC = 101325.0


class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')
          
        self.camera_sub = self.create_subscription(Image, "/camera", self.camera_callback, 10)
        self.pose_pub = self.create_publisher(Float64MultiArray, "/pose", 10)

        # single number each, straight into the yaw and depth controllers
        self.bearing_pub = self.create_publisher(Float64, "/target_bearing", 10)
        self.depth_pub = self.create_publisher(Float64, "/target_depth", 10)
        self.distance_pub = self.create_publisher(Float64, "/distance", 10)
        self.pressure_sub = self.create_subscription(FluidPressure, "/pressure", self.calculate_depth, 10)
        self.forward_pub = self.create_publisher(Float64, "/forward", 10)
        self.yolo_detected_pub = self.create_publisher(Bool, "/yolo_detected", 10)
        

        self.image_size = 256
        self.detector = Detector(families="tag36h11")
        self.latest_detections = []

        self.declare_parameter("weights_path", "/home/pherro/AUVC/yolo/weights/yolo26n_auv_1class_gray.onnx")
        self.declare_parameter("yolo_threads", 3)
        self.declare_parameter("yolo_rate", 5.0)
        self.declare_parameter("yolo_conf", 0.25)
        self.declare_parameter("yolo_pixel_kp", 0.03)
        self.declare_parameter("yolo_pixel_ki", 0.0)
        self.declare_parameter("yolo_pixel_kd", 0.0)
        self.declare_parameter("yolo_yaw_limit", 80.0)

        self.weights_path = self.get_parameter("weights_path").value
        self.yolo_threads = int(self.get_parameter("yolo_threads").value)
        self.yolo_period = 1.0 / float(self.get_parameter("yolo_rate").value)
        self.yolo_conf = float(self.get_parameter("yolo_conf").value)
        self.yolo_pixel_kp = float(self.get_parameter("yolo_pixel_kp").value)
        self.yolo_pixel_ki = float(self.get_parameter("yolo_pixel_ki").value)
        self.yolo_pixel_kd = float(self.get_parameter("yolo_pixel_kd").value)
        self.yolo_yaw_limit = float(self.get_parameter("yolo_yaw_limit").value)
        self.yolo_pixel_errors = []

        self.declare_parameter("frame_width", 640)
        self.declare_parameter("frame_height", 480)
        self.declare_parameter("camera_fx", 273.25)
        self.declare_parameter("camera_fy", 261.76)
        self.declare_parameter("camera_cx", 307.89)
        self.declare_parameter("camera_cy", 153.84)

        self.declare_parameter("camera_hfov_deg", 80.0)
        self.declare_parameter("tag_size", 0.10)
        self.declare_parameter("target_tag_id", -1)
        self.declare_parameter("yolo_detected_timeout", 1.5)

        self.frame_width = int(self.get_parameter("frame_width").value)
        self.frame_height = int(self.get_parameter("frame_height").value)
        self.tag_size = float(self.get_parameter("tag_size").value)
        # -1 follows whichever tag is closest, otherwise only this id is followed
        self.target_tag_id = int(self.get_parameter("target_tag_id").value)
        self.yolo_detected_timeout = float(self.get_parameter("yolo_detected_timeout").value)
        self.yolo_detected_time = 1e20
        self.setup_intrinsics()

        # leave cores free for yolo and the rest of the stack on the pi
        cv2.setNumThreads(1)

        self.frame_lock = threading.Lock()
        self.latest_frame = None
        self.latest_boxes = []
        self.using_yolo = USING_YOLO and YOLO_AVAILABLE and bool(self.weights_path)
        self.current_depth = 0


        if not USING_YOLO:
            self.get_logger().info("using apriltags for controller targets")
        elif not YOLO_AVAILABLE:
            self.get_logger().warn(
                "yolo requested but torch/ultralytics is unavailable; using apriltags"
            )
        elif not self.weights_path:
            self.get_logger().warn("yolo requested without weights_path; using apriltags")
        else:
            self.get_logger().info("using yolo for controller targets")
            self.yolo_thread = threading.Thread(target=self.yolo_worker, daemon=True)
            self.yolo_thread.start()

        # run loop at 20 hz
        self.timer_period = 0.05
        self.timer = self.create_timer(self.timer_period, self.pose_publisher)
        
        self.yolo_detected = False

    def setup_intrinsics(self):
        fx = float(self.get_parameter("camera_fx").value)
        fy = float(self.get_parameter("camera_fy").value)
        cx = float(self.get_parameter("camera_cx").value)
        cy = float(self.get_parameter("camera_cy").value)

        if fx <= 0.0 or fy <= 0.0:
            hfov = math.radians(float(self.get_parameter("camera_hfov_deg").value))
            fx = (self.frame_width / 2.0) / math.tan(hfov / 2.0)
            fy = fx
            cx = self.frame_width / 2.0
            cy = self.frame_height / 2.0
            self.get_logger().warn(
                "no camera_fx/camera_fy set, guessing intrinsics from hfov. "
                "a flat housing port refracts by about the index of water, so calibrate "
                "in the water before trusting any range"
            )

        # camera_callback center crops to a square and then downscales, so the calibrated
        # intrinsics have to go through the same transform as the pixels they describe
        side = min(self.frame_width, self.frame_height)
        crop_x = (self.frame_width - side) // 2
        crop_y = (self.frame_height - side) // 2
        scale = self.image_size / float(side)

        self.fx = fx * scale
        self.fy = fy * scale
        self.cx = (cx - crop_x) * scale
        self.cy = (cy - crop_y) * scale

    def pixel_to_angles(self, u, v):
        xn = (u - self.cx) / self.fx
        yn = (v - self.cy) / self.fy
        yaw = math.atan(xn)
        pitch = math.atan(yn / math.hypot(1.0, xn))
        return yaw, pitch, xn, yn

    def box_to_bearing_range(self, class_id, x1, y1, x2, y2):
        yaw, pitch, xn, yn = self.pixel_to_angles(0.5 * (x1 + x2), 0.5 * (y1 + y2))

        # a box clipped by the frame edge has a meaningless height, and that happens
        # exactly when closing on a target, so refuse to guess a range from it
        margin = 2.0
        truncated = (x1 < margin or y1 < margin or
                     x2 > self.image_size - margin or y2 > self.image_size - margin)

        height_px = y2 - y1
        real_height = KNOWN_HEIGHTS_M.get(int(class_id))
        if real_height is None or truncated or height_px < 1.0:
            return yaw, pitch, float("nan")

        # pinhole gives distance along the optical axis, so lift it to a line of sight range
        z = self.fy * real_height / height_px
        return yaw, pitch, z * math.sqrt(1.0 + xn * xn + yn * yn)

    def tag_range(self, detection):
        if detection.pose_t is None:
            return float("inf")
        return float(np.linalg.norm(np.asarray(detection.pose_t)))

    def select_tag(self, detections):
        if self.target_tag_id >= 0:
            candidates = [d for d in detections if d.tag_id == self.target_tag_id]
        else:
            candidates = list(detections)

        if not candidates:
            return None

        # closest tag wins, and one with no pose solution sorts to the back
        return min(candidates, key=self.tag_range)

    def calculate_depth(self, msg):
        output = (msg.fluid_pressure - ATMOSPHERIC) / (G * WATER_DENSITY)
        self.current_depth = output

    def camera_callback(self,msg):
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(self.frame_height, self.frame_width, 3)

        # center crop to a square, then downscale
        side = min(arr.shape[0], arr.shape[1])
        row = (arr.shape[0] - side) // 2
        col = (arr.shape[1] - side) // 2
        square = arr[row:row + side, col:col + side]
        resized = cv2.resize(square, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)

        if not self.using_yolo:
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            # four known corners and a known edge length give a far better range than any
            # box height guess, so solve the tag pose outright wherever a tag is visible
            self.latest_detections = self.detector.detect(
                gray,
                estimate_tag_pose=True,
                camera_params=(self.fx, self.fy, self.cx, self.cy),
                tag_size=self.tag_size,
            )
        else:
            self.latest_detections = []

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

            # only keep the highest-confidence detection for the controller
            best = None
            for result in results:
                for box in result.boxes:
                    conf = float(box.conf[0])
                    if best is None or conf > best[1]:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        best = [float(box.cls[0]), conf, x1, y1, x2, y2]

            with self.frame_lock:
                self.latest_boxes = [] if best is None else [best]
            
            if best is not None:
                self.yolo_detected = True
                self.yolo_detected_time = time.perf_counter()
                self.yolo_detected_pub.publish(Bool(data=True))
            else:
                if (time.perf_counter() - self.yolo_detected_time) > self.yolo_detected_timeout:
                    self.yolo_detected = False
                    self.yolo_detected_pub.publish(Bool(data=False))
            # time.sleep(self.yolo_period)

    def pose_publisher(self):
        with self.frame_lock:
            boxes = self.latest_boxes
        detections = self.latest_detections

        # [tag_count, box_count,
        #  (tag_id, cx, cy, yaw, pitch, x, y, z) per tag,
        #  (class_id, conf, x1, y1, x2, y2, yaw, pitch, range) for the highest-conf box]
        # box_count is 0 or 1. angles are radians, + yaw is right of center and + pitch is
        # below center. x/y/z and range are meters, nan when unavailable.
        msg = Float64MultiArray()
        msg.data = [float(len(detections)), float(len(boxes))]

        for detection in detections:
            cx, cy = detection.center
            yaw, pitch, _, _ = self.pixel_to_angles(cx, cy)
            if detection.pose_t is None:
                x = y = z = float("nan")
            else:
                x, y, z = (float(value) for value in np.asarray(detection.pose_t).reshape(3))
            msg.data.extend([float(detection.tag_id), float(cx), float(cy), yaw, pitch, x, y, z])

        for class_id, conf, x1, y1, x2, y2 in boxes:
            yaw, pitch, distance = self.box_to_bearing_range(class_id, x1, y1, x2, y2)
            msg.data.extend([class_id, conf, x1, y1, x2, y2, yaw, pitch, distance])

        self.pose_pub.publish(msg)
        if self.using_yolo:
            self.publish_yolo_target(boxes)
        else:
            self.publish_apriltag_target(detections)

    def publish_yolo_target(self, boxes):
        if not boxes:
            return
        
        class_id, _, x1, y1, x2, y2 = boxes[0]
        target_x = 0.5 * (x1 + x2)
        pixel_error = target_x - (self.image_size / 2.0)
        self.yolo_pixel_errors.append(pixel_error)
        self.yolo_pixel_errors = self.yolo_pixel_errors[-200:]

        # The YOLO controller now works directly in pixels instead of converting
        # the horizontal pixel error to an angle.
        # yaw, _, _, _ = self.pixel_to_angles(target_x, 0.5 * (y1 + y2))
        yaw_command = run_pid(
            self.yolo_pixel_errors,
            self.timer_period,
            self.yolo_pixel_kp,
            self.yolo_pixel_ki,
            self.yolo_pixel_kd,
            p_quadratic=True
        )
        yaw_command = max(-self.yolo_yaw_limit, min(self.yolo_yaw_limit, yaw_command))
        self.bearing_pub.publish(Float64(data=yaw_command))

        # Angle/range conversion remains here only for the depth setpoint.
        _, pitch, distance = self.box_to_bearing_range(class_id, x1, y1, x2, y2)

        # Keep /target_height in meters for both target sources. Positive means the
        # target is above the camera. Range is unavailable until KNOWN_HEIGHTS_M has
        # a real height for this YOLO class.
        if math.isfinite(distance):
            self.distance_pub.publish(Float64(data=distance))
            height = -distance * math.sin(pitch)
            self.depth_pub.publish(Float64(data=self.current_depth-height))
        

    def publish_apriltag_target(self, detections):
        # nothing is published while the tag is not in view, so a controller can tell
        # "no target" apart from "target dead ahead" by how long it has been since a message
        tag = self.select_tag(detections)
        if tag is None:
            return

        cx, cy = tag.center
        yaw, _, _, _ = self.pixel_to_angles(cx, cy)
        self.bearing_pub.publish(Float64(data=math.degrees(yaw)))

        if tag.pose_t is None:
            return

        distance = self.tag_range(tag)
        self.distance_pub.publish(Float64(data=distance))

        # the tag frame is x right, y down, z forward, so negating y gives a height
        # where positive means the tag sits above the camera
        height = -float(np.asarray(tag.pose_t).reshape(3)[1])
        self.depth_pub.publish(Float64(data=self.current_depth-height))
        



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
