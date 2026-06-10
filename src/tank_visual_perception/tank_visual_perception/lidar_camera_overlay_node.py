# -*- coding: utf-8 -*-
"""
ROS2 node: LiDAR points -> turret camera image projection overlay.

This node is for calibration visualization, and uses the same projection math as
path_planning/local_path_node.py so the manually tuned calibration affects both
RViz overlay and actual YOLO-LiDAR fusion.

Subscribe:
  /tank/camera/image_compressed     sensor_msgs/CompressedImage
  /tank/api/info/raw                std_msgs/String

Publish:
  /tank/camera/lidar_projection/image       sensor_msgs/Image
  /tank/camera/lidar_projection/compressed  sensor_msgs/CompressedImage
  /tank/camera/lidar_projection/status      std_msgs/String
"""

from __future__ import annotations

import json
import threading
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import String

from tank_visual_perception.projection import (
    DEFAULT_PROJECTION_PARAMS,
    compute_camera_pose,
    extract_info_payload,
    lidar_point_raw_position,
    project_point,
    to_float,
    vec3_from_dict,
)


def compressed_msg_to_cv2(msg: CompressedImage) -> Optional[np.ndarray]:
    np_arr = np.frombuffer(msg.data, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


def cv2_to_image_msg(image_bgr: np.ndarray, stamp, frame_id: str) -> Image:
    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = int(image_bgr.shape[0])
    msg.width = int(image_bgr.shape[1])
    msg.encoding = "bgr8"
    msg.is_bigendian = False
    msg.step = int(image_bgr.shape[1] * 3)
    msg.data = image_bgr.tobytes()
    return msg


def cv2_to_compressed_msg(image_bgr: np.ndarray, stamp, frame_id: str, quality: int = 85) -> Optional[CompressedImage]:
    ok, buffer = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return None
    msg = CompressedImage()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.format = "jpeg"
    msg.data = buffer.tobytes()
    return msg


class LidarCameraOverlayNode(Node):
    def __init__(self) -> None:
        super().__init__("lidar_camera_overlay_node")

        self.declare_parameter("image_topic", "/tank/camera/image_compressed")
        self.declare_parameter("info_topic", "/tank/api/info/raw")
        self.declare_parameter("out_image_topic", "/tank/camera/lidar_projection/image")
        self.declare_parameter("out_compressed_topic", "/tank/camera/lidar_projection/compressed")
        self.declare_parameter("out_status_topic", "/tank/camera/lidar_projection/status")
        for name, value in DEFAULT_PROJECTION_PARAMS.items():
            self.declare_parameter(name, value)
        self.declare_parameter("use_only_detected", True)
        self.declare_parameter("min_distance", 1.0)
        self.declare_parameter("max_distance", 35.0)
        self.declare_parameter("point_radius", 2)
        self.declare_parameter("draw_text", True)
        self.declare_parameter("jpeg_quality", 85)

        self.image_topic = str(self.get_parameter("image_topic").value)
        self.info_topic = str(self.get_parameter("info_topic").value)
        self.out_image_topic = str(self.get_parameter("out_image_topic").value)
        self.out_compressed_topic = str(self.get_parameter("out_compressed_topic").value)
        self.out_status_topic = str(self.get_parameter("out_status_topic").value)
        self.params = {key: float(self.get_parameter(key).value) for key in DEFAULT_PROJECTION_PARAMS.keys()}
        self.use_only_detected = bool(self.get_parameter("use_only_detected").value)
        self.min_distance = float(self.get_parameter("min_distance").value)
        self.max_distance = float(self.get_parameter("max_distance").value)
        self.point_radius = int(self.get_parameter("point_radius").value)
        self.draw_text = bool(self.get_parameter("draw_text").value)
        self.jpeg_quality = int(self.get_parameter("jpeg_quality").value)

        self._lock = threading.Lock()
        self._latest_info: Optional[Dict[str, Any]] = None
        self._latest_info_stamp = None

        self.create_subscription(String, self.info_topic, self.on_info, 10)
        self.create_subscription(CompressedImage, self.image_topic, self.on_image, 10)
        self.pub_overlay_image = self.create_publisher(Image, self.out_image_topic, 10)
        self.pub_overlay_compressed = self.create_publisher(CompressedImage, self.out_compressed_topic, 10)
        self.pub_status = self.create_publisher(String, self.out_status_topic, 10)

        self.get_logger().info("LiDAR-camera overlay node started")
        self.get_logger().info(f"subscribe image: {self.image_topic}")
        self.get_logger().info(f"subscribe info : {self.info_topic}")
        self.get_logger().info(f"publish image  : {self.out_image_topic}")
        self.get_logger().info(f"projection params: {self.params}")

    def on_info(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
            info = extract_info_payload(payload)
            if info is None:
                return
            with self._lock:
                self._latest_info = info
                self._latest_info_stamp = self.get_clock().now()
        except Exception as exc:
            self.get_logger().warn(f"failed to parse info raw: {exc}")

    def on_image(self, msg: CompressedImage) -> None:
        image = compressed_msg_to_cv2(msg)
        if image is None:
            self.get_logger().warn("failed to decode compressed image")
            return

        with self._lock:
            info = self._latest_info

        if info is None:
            overlay = image.copy()
            cv2.putText(overlay, "Waiting for /tank/api/info/raw...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            self.publish_overlay(overlay, msg.header.stamp, msg.header.frame_id or "tank_camera")
            self.publish_status(0, 0, 0, "waiting_info")
            return

        overlay, projected_count, used_count, total_count = self.draw_lidar_overlay(image, info)
        if self.draw_text:
            text = f"LiDAR projection: {projected_count}/{used_count} used, total={total_count}"
            cv2.putText(overlay, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(overlay, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        self.publish_overlay(overlay, msg.header.stamp, msg.header.frame_id or "tank_camera")
        self.publish_status(projected_count, used_count, total_count, "ok")

    def draw_lidar_overlay(self, image: np.ndarray, info: Dict[str, Any]) -> Tuple[np.ndarray, int, int, int]:
        h, w = image.shape[:2]
        overlay = image.copy()
        try:
            camera_pos, camera_yaw, camera_pitch, camera_roll = compute_camera_pose(info, self.params)
        except Exception as exc:
            cv2.putText(overlay, f"Invalid info: {exc}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            return overlay, 0, 0, 0

        lidar_points = info.get("lidarPoints", [])
        if not isinstance(lidar_points, list):
            return overlay, 0, 0, 0

        projected_count = 0
        used_count = 0
        total_count = len(lidar_points)
        for p in lidar_points:
            if not isinstance(p, dict):
                continue
            if self.use_only_detected and not bool(p.get("isDetected", False)):
                continue
            distance = to_float(p.get("distance"), 9999.0)
            if distance < self.min_distance or distance > self.max_distance:
                continue
            pos_raw = lidar_point_raw_position(p)
            if pos_raw is None:
                continue
            used_count += 1
            projected = project_point(
                point_world_raw=vec3_from_dict(pos_raw),
                camera_pos_world_raw=camera_pos,
                camera_yaw_deg=camera_yaw,
                camera_pitch_deg=camera_pitch,
                camera_roll_deg=camera_roll,
                image_w=w,
                image_h=h,
                params=self.params,
            )
            if projected is None:
                continue
            u, v, _depth = projected
            if 0 <= u < w and 0 <= v < h:
                ratio = max(0.0, min(1.0, distance / max(0.001, self.max_distance)))
                b = int(255 * ratio)
                r = int(255 * (1.0 - ratio))
                cv2.circle(overlay, (u, v), self.point_radius, (b, 255, r), -1, cv2.LINE_AA)
                projected_count += 1
        return overlay, projected_count, used_count, total_count

    def publish_overlay(self, image_bgr: np.ndarray, stamp, frame_id: str) -> None:
        self.pub_overlay_image.publish(cv2_to_image_msg(image_bgr, stamp, frame_id))
        comp_msg = cv2_to_compressed_msg(image_bgr, stamp, frame_id, self.jpeg_quality)
        if comp_msg is not None:
            self.pub_overlay_compressed.publish(comp_msg)

    def publish_status(self, projected_count: int, used_count: int, total_count: int, state: str) -> None:
        msg = String()
        msg.data = json.dumps(
            {
                "state": state,
                "projected_count": projected_count,
                "used_count": used_count,
                "total_count": total_count,
                "params": self.params,
                "method": "shared_projection_math",
            },
            ensure_ascii=False,
        )
        self.pub_status.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LidarCameraOverlayNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
