# -*- coding: utf-8 -*-
"""Lightweight live camera view for ros_bridge.

This module does not run YOLO. It only displays the latest /detect frame and the
latest detection list already produced by the bridge/vision path.
"""

from __future__ import annotations

import time
from copy import deepcopy
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from flask import Response, render_template_string

try:
    import cv2
except Exception:  # pragma: no cover - runtime optional guard
    cv2 = None

_state_lock = Lock()
_latest_frame: Optional[np.ndarray] = None
_latest_frame_seq = 0
_latest_frame_timestamp = 0.0
_latest_frame_shape: Optional[List[int]] = None
_latest_detections: List[Dict[str, Any]] = []
_latest_detection_metadata: Dict[str, Any] = {}
_latest_detection_timestamp = 0.0
_latest_error: Optional[str] = None

_CLASS_COLORS_BGR = {
    "tank": (0, 0, 255),
    "rock": (0, 255, 255),
    "person": (0, 255, 0),
    "tent": (255, 255, 0),
    "wall": (255, 0, 0),
    "unknown": (255, 255, 255),
}
_COLOR_PALETTE_BGR = [
    (0, 255, 0),
    (0, 0, 255),
    (255, 0, 0),
    (0, 255, 255),
    (255, 0, 255),
    (255, 255, 0),
    (255, 255, 255),
]


def _decode_jpeg(image_bytes: bytes) -> Optional[np.ndarray]:
    if cv2 is None or not image_bytes:
        return None
    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)


def update_frame(image_bytes: bytes) -> Optional[List[int]]:
    """Store latest camera frame. Returns frame shape [h, w, c] if decoded."""
    global _latest_frame, _latest_frame_seq, _latest_frame_timestamp, _latest_frame_shape, _latest_error
    frame = _decode_jpeg(image_bytes)
    if frame is None:
        with _state_lock:
            _latest_error = "live_view: failed to decode frame or cv2 unavailable"
        return None
    shape = [int(v) for v in frame.shape]
    with _state_lock:
        _latest_frame = frame
        _latest_frame_seq += 1
        _latest_frame_timestamp = time.time()
        _latest_frame_shape = shape
        _latest_error = None
    return shape


def update_detections(detections: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
    """Store latest detection list for overlay."""
    global _latest_detections, _latest_detection_metadata, _latest_detection_timestamp
    with _state_lock:
        _latest_detections = deepcopy(detections) if isinstance(detections, list) else []
        _latest_detection_metadata = deepcopy(metadata) if isinstance(metadata, dict) else {}
        _latest_detection_timestamp = time.time()


def _class_color(class_name: str, class_id: int = 0) -> Tuple[int, int, int]:
    key = str(class_name).strip().lower()
    if key in _CLASS_COLORS_BGR:
        return _CLASS_COLORS_BGR[key]
    return _COLOR_PALETTE_BGR[int(class_id) % len(_COLOR_PALETTE_BGR)]


def _draw_detections(frame: np.ndarray, detections: List[Dict[str, Any]], metadata: Dict[str, Any]) -> np.ndarray:
    if cv2 is None:
        return frame
    drawn = frame.copy()
    for det in detections:
        if not isinstance(det, dict):
            continue
        bbox = det.get("bbox", [])
        if not isinstance(bbox, list) or len(bbox) < 4:
            continue
        try:
            x1, y1, x2, y2 = [int(float(v)) for v in bbox[:4]]
        except Exception:
            continue
        class_name = str(det.get("className", det.get("class_name", "object"))).strip().lower()
        class_id = int(det.get("classId") or 0)
        track_id = det.get("trackId", det.get("track_id"))
        fixed_id = det.get("classFixedId", det.get("id"))
        conf = float(det.get("confidence") or 0.0)
        color = _class_color(class_name, class_id)
        cv2.rectangle(drawn, (x1, y1), (x2, y2), color, 2)
        id_text = ""
        if fixed_id is not None:
            id_text += f" ID:{fixed_id}"
        if track_id is not None:
            id_text += f" T:{track_id}"
        label = f"{class_name}{id_text} {conf:.2f}"
        cv2.putText(
            drawn,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    frame_seq = metadata.get("frameSeq")
    processed_seq = metadata.get("processedFrameSeq")
    age_ms = metadata.get("resultAgeMs")
    async_flag = metadata.get("asyncYolo")
    status = f"det={len(detections)}"
    if async_flag:
        status += f" async frame={frame_seq} yolo={processed_seq} age={age_ms:.0f}ms" if isinstance(age_ms, (int, float)) else f" async frame={frame_seq} yolo={processed_seq}"
    else:
        status += " sync"
    cv2.putText(drawn, status, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
    return drawn


def _blank_frame(message: str = "Waiting for /detect image...") -> np.ndarray:
    frame = np.zeros((480, 854, 3), dtype=np.uint8)
    if cv2 is not None:
        cv2.putText(frame, message, (40, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return frame


def render_view_page() -> str:
    html = """
    <!doctype html>
    <html lang="ko">
    <head>
        <meta charset="utf-8">
        <title>Tank YOLO Live View</title>
        <style>
            body { margin: 0; background: #111; color: #eee; font-family: Arial, sans-serif; text-align: center; }
            header { padding: 12px 20px; background: #1e1e1e; border-bottom: 1px solid #333; }
            .wrap { padding: 16px; }
            img { max-width: 96vw; max-height: 82vh; border: 2px solid #00ff00; background: #000; }
            .hint { margin-top: 10px; color: #aaa; font-size: 14px; }
            a { color: #7dd3fc; }
        </style>
    </head>
    <body>
        <header><h2>Tank YOLO Live View</h2></header>
        <div class="wrap">
            <img src="/video_feed" alt="YOLO stream">
            <div class="hint">/detect로 들어온 최신 프레임 위에 bridge가 반환한 최신 bbox/trackId를 표시합니다.</div>
            <div class="hint">상태 확인: <a href="/debug/live_view">/debug/live_view</a> · <a href="/debug/yolo">/debug/yolo</a></div>
        </div>
    </body>
    </html>
    """
    return render_template_string(html)


def generate_video_stream(web_fps: float = 20.0, jpeg_quality: int = 80):
    interval = 1.0 / max(1.0, float(web_fps))
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)] if cv2 is not None else []
    while True:
        with _state_lock:
            frame = None if _latest_frame is None else _latest_frame.copy()
            detections = deepcopy(_latest_detections)
            metadata = deepcopy(_latest_detection_metadata)
        if frame is None:
            frame = _blank_frame()
        else:
            frame = _draw_detections(frame, detections, metadata)
        if cv2 is not None:
            ok, buffer = cv2.imencode(".jpg", frame, encode_params)
            if ok:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        time.sleep(interval)


def video_response(web_fps: float = 20.0, jpeg_quality: int = 80) -> Response:
    return Response(generate_video_stream(web_fps=web_fps, jpeg_quality=jpeg_quality), mimetype="multipart/x-mixed-replace; boundary=frame")


def debug_state() -> Dict[str, Any]:
    with _state_lock:
        frame_age = time.time() - _latest_frame_timestamp if _latest_frame_timestamp else None
        det_age = time.time() - _latest_detection_timestamp if _latest_detection_timestamp else None
        return {
            "enabled": True,
            "opencvAvailable": cv2 is not None,
            "latestFrameSeq": _latest_frame_seq,
            "latestFrameShape": deepcopy(_latest_frame_shape),
            "latestFrameAgeMs": None if frame_age is None else frame_age * 1000.0,
            "latestDetectionCount": len(_latest_detections),
            "latestDetectionAgeMs": None if det_age is None else det_age * 1000.0,
            "latestDetectionMetadata": deepcopy(_latest_detection_metadata),
            "latestError": _latest_error,
        }
