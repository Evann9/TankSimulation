# Project Overview

## Tech Stack & Dependencies
- **Language**: Python
- **Web Framework**: Flask, Werkzeug (for API Endpoints to communicate with the Simulator)
- **Machine Learning & Vision**: PyTorch, Ultralytics (YOLOv8), OpenCV (opencv-python), Torchvision
- **Data & Math**: NumPy, SciPy, SymPy, Matplotlib, Polars, NetworkX
- **Other**: PyYAML (for configs), Requests

## Directory Tree & Key Modules Description
- `src/`: Core logic modules.
  - `main.py`: Skeleton entry point for the tank client.
  - `simulator/`: Contains `api_client.py` for simulator communication.
  - `perception/`, `planning/`, `rl/`: Directories reserved for specific logic layers (vision, path finding, reinforcement learning).
- `scripts/`: Executable scripts and sample codes.
  - `server_sample.py`: A sample Flask server that communicates with the simulator.
  - `run_client.py`: Client runner script.
  - `yolov8n.pt`: YOLO model weights.
- `docs/`: Documentation.
  - `simulator_api.md`: API specifications for simulator endpoints.
- `configs/`: Configuration files (e.g., `simulator.yaml`).
- `data/` & `map/`: Data and map assets.
- `tests/`: Testing directory.
- `.ai/`: Contains AI specific documents and plans like `experiment_plan.md`.

## Execution & Setup Guide
1. **Dependencies**: `pip install -r requirements.txt`
2. **Configuration**: Modify parameters in `configs/simulator.yaml` if needed.
3. **Run Client**: Execute the sample server via `python scripts/server_sample.py` or the main entry `python src/main.py`. The Flask server runs on port 5000 by default and listens to requests from the Unity Simulator.

## Architectural Observations
- **Communication Paradigm**: The Python application acts as an HTTP Server (Flask), and the Unity Simulator acts as the Client. The simulator polls endpoints like `/init`, `/get_action`, and `/info` periodically based on the `Request Interval`.
- **Control Flow**: Actions are queued and popped on every `/get_action` request. There is no push capability; all state changes must be delivered as responses to the simulator's polling.
- **Next Steps**: The immediate task is to implement the experiment server as defined in `.ai/docs/experiment_plan.md`, which requires creating an automated trial harness and a CSV logger.
