# -*- coding: utf-8 -*-
"""LiDAR payload parsing and obstacle preprocessing.

다른 패키지(path_planning, potential 등)는 LiDAR JSON schema를 직접 파싱하지 않고
여기 함수만 import해서 사용한다. 이렇게 해야 LiDAR schema가 바뀌어도 수정 지점이
lidar 패키지로 제한된다.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from .config import BBOX_MIN_THICKNESS
from .coordinate_utils import lidar_point_with_map_position, to_float

Point2D = Tuple[float, float]
BBox2D = Dict[str, float]


def extract_payload_list(data: Any, key: str = "points") -> List[Any]:
    if isinstance(data, list):
        return data
    if not isinstance(data, dict):
        return []
    if isinstance(data.get(key), list):
        return data[key]
    inner = data.get("data")
    if isinstance(inner, list):
        return inner
    if isinstance(inner, dict) and isinstance(inner.get(key), list):
        return inner[key]
    return []


def iter_detected_points(raw_points: Any) -> Iterable[Dict[str, Any]]:
    if not isinstance(raw_points, list):
        return []
    return (p for p in raw_points if isinstance(p, dict) and bool(p.get("isDetected", False)))


def build_detected_map_payload(
    lidar_points: Any,
    timestamp_wall: float,
    map_frame: str,
    ground_filter_enabled: bool = False,
    origin_y: float = 8.0,
) -> Dict[str, Any]:
    source_points = list(iter_detected_points(lidar_points))
    if ground_filter_enabled and source_points:
        try:
            from .perception_utils import filter_ground_points
            source_points = filter_ground_points(source_points, origin_y)
        except Exception:
            # Node code may log the exception if it needs detail; this utility stays side-effect free.
            pass

    points = []
    for point in source_points:
        converted = lidar_point_with_map_position(point)
        if converted is not None:
            points.append(converted)

    return {
        "route": "/info",
        "timestamp_wall": timestamp_wall,
        "source": "lidarPoints",
        "frame_id": map_frame,
        "coordinate_policy": "position_map: x=raw.x, y=raw.z, z=raw.y",
        "count": len(points),
        "points": points,
    }


def parse_lidar_points_payload(payload: Any) -> List[Point2D]:
    """Parse /tank/sensor/lidar/detected_points_map into map-plane (x, y)."""
    points: List[Point2D] = []
    for item in extract_payload_list(payload, "points"):
        if not isinstance(item, dict):
            continue
        pos = item.get("position_map") if isinstance(item.get("position_map"), dict) else item.get("position")
        if not isinstance(pos, dict):
            continue
        try:
            if "y" in pos:
                points.append((float(pos.get("x", 0.0)), float(pos.get("y", 0.0))))
            else:
                points.append((float(pos.get("x", 0.0)), float(pos.get("z", 0.0))))
        except Exception:
            continue
    return points


def distance(a: Point2D, b: Point2D) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def filter_lidar_points_by_distance(
    current_pos: Point2D,
    lidar_points: Sequence[Point2D],
    min_distance: float,
    max_distance: float,
) -> List[Point2D]:
    filtered: List[Point2D] = []
    for p in lidar_points:
        d = distance(current_pos, p)
        if min_distance <= d <= max_distance:
            filtered.append(p)
    return filtered


def cluster_lidar_points(points: Sequence[Point2D], eps: float = 2.0, min_samples: int = 3) -> List[List[Point2D]]:
    clusters: List[List[Point2D]] = []
    visited = set()
    pts = list(points)
    for i, _ in enumerate(pts):
        if i in visited:
            continue
        queue = [i]
        visited.add(i)
        cluster: List[Point2D] = []
        while queue:
            idx = queue.pop(0)
            p = pts[idx]
            cluster.append(p)
            for j, q in enumerate(pts):
                if j in visited:
                    continue
                if distance(p, q) <= eps:
                    visited.add(j)
                    queue.append(j)
        if len(cluster) >= min_samples:
            clusters.append(cluster)
    return clusters


def lidar_clusters_to_bboxes(clusters: Sequence[Sequence[Point2D]], min_thickness: float = BBOX_MIN_THICKNESS) -> List[BBox2D]:
    bboxes: List[BBox2D] = []
    for cluster in clusters:
        if not cluster:
            continue
        xs = [p[0] for p in cluster]
        ys = [p[1] for p in cluster]
        x_min, x_max = min(xs), max(xs)
        z_min, z_max = min(ys), max(ys)
        if x_max - x_min < min_thickness:
            pad = 0.5 * (min_thickness - (x_max - x_min))
            x_min -= pad
            x_max += pad
        if z_max - z_min < min_thickness:
            pad = 0.5 * (min_thickness - (z_max - z_min))
            z_min -= pad
            z_max += pad
        bboxes.append({"x_min": x_min, "x_max": x_max, "z_min": z_min, "z_max": z_max})
    return bboxes


def update_lidar_history(
    history: List[Point2D],
    history_set: set,
    points: Sequence[Point2D],
    resolution: float,
    max_points: int,
) -> Tuple[List[Point2D], set]:
    q = max(resolution, 0.1)
    for x, y in points:
        rounded = (round(x / q) * q, round(y / q) * q)
        if rounded not in history_set:
            history_set.add(rounded)
            history.append(rounded)
    if len(history) > max_points:
        drop = len(history) - max_points
        for p in history[:drop]:
            history_set.discard(p)
        history = history[drop:]
    return history, history_set
