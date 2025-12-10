# Robot Motion Retargeting & Visualization Project

This project provides a framework for robot motion retargeting, visualization, and deployment, primarily using ROS2 and MuJoCo. It supports motion capture data processing (e.g., from XSens), retargeting motion to various robot models (like H1, G1), and visualizing the results.

## Directory Structure

```
├── src/
│   ├── motion_retargeting/       # Core retargeting logic and robot models
│   │   ├── motion_retargeting/   # Python source for retargeting
│   │   │   ├── bvh_parser.py     # Parses BVH motion files
│   │   │   ├── mujoco_play.py    # Plays retargeted motion in MuJoCo
│   │   │   ├── data_subscriber.py# Subscribes to motion data topics
│   │   │   └── retarget/         # IK and retargeting algorithms
│   │   └── robots/               # Robot URDF/XML descriptions (e.g., H1, G1)
│   │
│   ├── deploy_controller/        # Robot deployment and control logic
│   │   └── deploy_controller/
│   │       ├── g1_controller.py  # Controller for G1 robot
│   │       └── g1_mujoco_play.py # Simulation playback for G1
│   │
│   ├── xsens_mvn_ros2/           # XSens Motion Capture ROS2 integration
│   │
│   ├── vis_robot_motion.py       # Standalone utility to visualize .pkl motion files
│   └── generate_dummy_motion.py  # Helper to generate test motion data
│
└── requirements.txt              # Python dependencies
```

## Key Components

### 1. Motion Retargeting (`src/motion_retargeting`)
This is the core package for processing motion data.
- **`bvh_parser.py`**: Reads `.bvh` files (common mocap format) and converts them into a structure usable for retargeting.
- **`retarget/`**: Contains the inverse kinematics (IK) and optimization logic to map human motion to robot joint angles.
- **`robots/`**: Stores the kinematic descriptions (MJCF/URDF) for supported robots (e.g., H1, AzureLoong, G1).

### 2. Deployment Controller (`src/deploy_controller`)
Handles the execution of motion on specific robots or simulations.
- **`g1_controller.py`**: tailored control logic for the G1 humanoid robot.

### 3. Visualization Tools
- **`src/vis_robot_motion.py`**: A standalone script to visualize robot motion stored in `.pkl` files using MuJoCo.
    - **Usage**:
      ```bash
      mjpython src/vis_robot_motion.py --xml_path <path_to_xml> --robot_motion_path <path_to_pkl>
      ```
    - Supports video recording via `--record_video`.

- **`src/generate_dummy_motion.py`**: A utility created to generate valid placeholder `.pkl` motion files for testing the visualization pipeline when real data is unavailable.

## Quick Start

### Prerequisites
- specific Conda environment (as active in the shell).
- `mujoco`, `imageio`, `numpy`.

### Visualizing Motion
To verify the setup, you can generate dummy motion data and run the visualizer:

1.  **Generate Test Data**:
    ```bash
    python src/generate_dummy_motion.py
    ```
    This creates `dummy_motion.pkl`.

2.  **Run Visualizer**:
    ```bash
    # Use mjpython on macOS to avoid rendering issues
    mjpython src/vis_robot_motion.py \
        --xml_path src/motion_retargeting/robots/h1_2/h1_2.xml \
        --robot_motion_path dummy_motion.pkl
    ```

## Development Notes
- **macOS Compatibility**: MuJoCo's passive viewer requires running scripts with `mjpython` instead of standard `python` on macOS.
- **Data Formats**: The core motion data format used for visualization is a pickled dictionary containing `root_pos`, `root_rot`, `dof_pos`, and `fps`.
