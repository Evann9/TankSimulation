# -*- coding: utf-8 -*-
"""ROS2 LiDAR preprocessing node for Tank Challenge.

책임 범위:
- /tank/api/info/raw 안의 lidarOrigin, lidarRotation, lidarPoints 분리
- Unity raw 좌표를 tank_map 좌표로 변환
- isDetected=True hit point만 /tank/sensor/lidar/detected_points_map으로 publish

다른 패키지는 LiDAR raw schema를 직접 해석하지 않고 이 노드의 출력 topic을 사용한다.
"""

from __future__ import annotations

import json
from typing import Any, Dict

import rclpy
from geometry_msgs.msg import PointStamped, Vector3Stamped
from rclpy.node import Node
from std_msgs.msg import Int32, String

from lidar.config import (
    DEFAULT_LIDAR_ORIGIN_Y,
    GROUND_FILTER_ENABLED,
    MAP_FRAME,
    TOPIC_INFO_RAW,
    TOPIC_LIDAR_DETECTED_MAP,
    TOPIC_LIDAR_ORIGIN,
    TOPIC_LIDAR_ORIGIN_RAW,
    TOPIC_LIDAR_POINTS,
    TOPIC_LIDAR_POINTS_COUNT,
    TOPIC_LIDAR_ROTATION,
    UNITY_FRAME,
)
from lidar.coordinate_utils import as_xyz, dumps_compact, raw_and_map_point, to_float
from lidar.payloads import build_detected_map_payload


class LidarProcessorNode(Node):
    def __init__(self) -> None:
        super().__init__("lidar_processor_node")

        self.declare_parameter("info_raw_topic", TOPIC_INFO_RAW)
        self.declare_parameter("ground_filter_enabled", GROUND_FILTER_ENABLED)
        self.declare_parameter("default_lidar_origin_y", DEFAULT_LIDAR_ORIGIN_Y)

        self.info_raw_topic = str(self.get_parameter("info_raw_topic").value)
        self.ground_filter_enabled = bool(self.get_parameter("ground_filter_enabled").value)
        self.default_lidar_origin_y = float(self.get_parameter("default_lidar_origin_y").value)

        self.pub_points = self.create_publisher(String, TOPIC_LIDAR_POINTS, 10)
        self.pub_points_count = self.create_publisher(Int32, TOPIC_LIDAR_POINTS_COUNT, 10)
        self.pub_origin = self.create_publisher(PointStamped, TOPIC_LIDAR_ORIGIN, 10)
        self.pub_origin_raw = self.create_publisher(PointStamped, TOPIC_LIDAR_ORIGIN_RAW, 10)
        self.pub_rotation = self.create_publisher(Vector3Stamped, TOPIC_LIDAR_ROTATION, 10)
        self.pub_detected_map = self.create_publisher(String, TOPIC_LIDAR_DETECTED_MAP, 10)

        self.create_subscription(String, self.info_raw_topic, self.info_raw_cb, 10)
        self.get_logger().info(
            f"LiDAR processor started: sub={self.info_raw_topic}, "
            f"pub=/tank/sensor/lidar/*, ground_filter={self.ground_filter_enabled}"
        )

    def publish_json(self, publisher: Any, data: Any) -> None:
        msg = String()
        msg.data = dumps_compact(data)
        publisher.publish(msg)

    def publish_int(self, publisher: Any, value: int) -> None:
        msg = Int32()
        msg.data = int(value)
        publisher.publish(msg)

    def publish_point(self, publisher: Any, point: Dict[str, Any]) -> None:
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = str(point.get("frame_id", MAP_FRAME))
        msg.point.x = to_float(point.get("x"))
        msg.point.y = to_float(point.get("y"))
        msg.point.z = to_float(point.get("z"))
        publisher.publish(msg)

    def publish_vector3(self, publisher: Any, vector: Dict[str, Any], frame_id: str = UNITY_FRAME) -> None:
        msg = Vector3Stamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        msg.vector.x = to_float(vector.get("x"))
        msg.vector.y = to_float(vector.get("y"))
        msg.vector.z = to_float(vector.get("z"))
        publisher.publish(msg)

    def info_raw_cb(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
            data = payload.get("data", payload) if isinstance(payload, dict) else {}
            if not isinstance(data, dict):
                return
        except Exception as exc:
            self.get_logger().debug(f"/tank/api/info/raw parse failed: {exc}")
            return

        ts = to_float(payload.get("timestamp_wall")) if isinstance(payload, dict) else 0.0
        lidar_points = data.get("lidarPoints") if isinstance(data.get("lidarPoints"), list) else []
        lidar_count = len(lidar_points)

        points_payload = {
            "route": "/info",
            "timestamp_wall": ts,
            "source": "lidarPoints",
            "count": lidar_count,
            "points": lidar_points,
        }

        origin = data.get("lidarOrigin")
        origin_y = to_float(origin.get("y"), self.default_lidar_origin_y) if isinstance(origin, dict) else self.default_lidar_origin_y
        detected_payload = build_detected_map_payload(
            lidar_points,
            timestamp_wall=ts,
            map_frame=MAP_FRAME,
            ground_filter_enabled=self.ground_filter_enabled,
            origin_y=origin_y,
        )

        self.publish_int(self.pub_points_count, lidar_count)
        self.publish_json(self.pub_points, points_payload)
        self.publish_json(self.pub_detected_map, detected_payload)

        if isinstance(origin, dict):
            origin_raw, origin_map = raw_and_map_point(origin, "/info/lidarOrigin")
            self.publish_point(self.pub_origin_raw, origin_raw)
            self.publish_point(self.pub_origin, origin_map)

        rotation = as_xyz(data.get("lidarRotation"))
        if rotation is not None:
            self.publish_vector3(self.pub_rotation, rotation, UNITY_FRAME)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LidarProcessorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
