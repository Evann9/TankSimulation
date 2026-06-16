#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import os
import sys


def _add_source_package_paths() -> None:
    src_root = Path(__file__).resolve().parents[1]
    for package_dir in (
        src_root / "ros_bridge",
        src_root / "vision",
    ):
        package_path = str(package_dir)
        if package_dir.exists() and package_path not in sys.path:
            sys.path.insert(0, package_path)


_add_source_package_paths()


def _set_direct_run_defaults() -> None:
    src_root = Path(__file__).resolve().parents[1]
    yolo_engine_path = src_root / "vision" / "models" / "best_300.engine"
    yolo_pt_path = src_root / "vision" / "models" / "best_300.pt"
    yolo_model_path = yolo_engine_path if yolo_engine_path.exists() else yolo_pt_path
    yolo_config_path = src_root / "vision" / "config" / "yolo_detection.yaml"

    if yolo_model_path.exists():
        os.environ["TANK_YOLO_MODEL_PATH"] = str(yolo_model_path)
    if yolo_config_path.exists():
        os.environ["TANK_YOLO_CONFIG"] = str(yolo_config_path)

    os.environ.setdefault("TANK_YOLO_ASYNC", "true")
    os.environ.setdefault("TANK_YOLO_ASYNC_MIN_INTERVAL_SEC", "0.02" if yolo_engine_path.exists() else "0.08")
    os.environ.setdefault("TANK_YOLO_ASYNC_MAX_RESULT_AGE_MS", "700")
    os.environ.setdefault("TANK_YOLO_ASYNC_WAIT_FOR_FRESH_MS", "0")
    os.environ.setdefault("TANK_YOLO_ASYNC_LOG_INTERVAL_SEC", "2.0")
    os.environ.setdefault("TANK_PUBLISH_DETECT_IMAGE", "false")
    os.environ.setdefault("TANK_LIVE_VIEW_FPS", "30")
    os.environ.setdefault("TANK_LIVE_VIEW_JPEG_QUALITY", "65")
    os.environ.setdefault("TANK_LIVE_VIEW_DECODE_FPS", "4")
    os.environ.setdefault("TANK_LIVE_VIEW_MAX_SIDE", "900")
    os.environ.setdefault("TANK_REQUEST_LOG", "false")
    os.environ.setdefault("YOLO_TRACKING", "false")
    os.environ.setdefault("YOLO_IMGSZ", "416")
    os.environ.setdefault("YOLO_MIN_INTERVAL", "0.02" if yolo_engine_path.exists() else "0.08")
    os.environ.setdefault("YOLO_JPEG_REDUCED_DECODE", "true")
    os.environ.setdefault("YOLO_JPEG_DECODE_MAX_SIDE", "960")
    os.environ.setdefault("YOLO_CUDNN_BENCHMARK", "true")
    os.environ.setdefault("YOLO_RECOGNITION_LOG", "false")


_set_direct_run_defaults()

from ros_bridge.main import main


if __name__ == "__main__":
    main()
