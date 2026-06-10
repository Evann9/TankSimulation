#!/usr/bin/env python3
"""Compatibility wrapper for the standalone YOLO debug server.

The production integration is now inside ros_bridge /detect.
For standalone YOLO testing, run:
    ros2 run vision yolo_debug_server
or:
    python3 src/vision/run_yolo_server.py
"""

from vision.yolo_debug_server import main

if __name__ == "__main__":
    main()
