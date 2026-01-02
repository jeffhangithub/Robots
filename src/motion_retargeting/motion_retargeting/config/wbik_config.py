"""
文件作用：配置 WBIK（全身逆运动学）相关参数，包括 MJCF 路径、接触阈值、权重及映射表等，用于 motion_retargeting 的求解与控制。
作者：Jeff
日期：2026-01-02
"""

import numpy as np


class WBIKConfig:
    """
    mjcf_path (str): Path to the MJCF model of the robot
    termination_velocity (float): Terminates integration when joints reach specified velocity
    max_iters (float): Max integration steps
    skip_iters (float): Max integration steps for initial position, throws exception when reached
    contact_velocity (float): foot velocity contact threshold
    height_offset (float): global z coordinate offset for the motion data
    yaw_only_feet (bool): wether to ignore pitch and roll orientation of the foot task
    joint_vel_weight (float): joint velocity constraints weight
    joint_limit_scale (float): scales min and max limits of the joints
    contact_target_lerp (float): alpha for exponential filtering of the desired foot positions and rotations when not in contact with the ground
    com_pos_weight (float): weight for keeping the com close to the foot support line
    body_to_model_map (dict): MJCF to task name mapping of the corresponding frames
    step_dt (float): integration step size
    extra_bodies (list[string]): names of the body in the xml for which to export transform data aside from the task bodies
    """

    mjcf_path: str # Path to the MJCF model of the robot
    termination_velocity: float = 5e-1 # Terminates integration when joints reach specified velocity
    max_iters: int = 1 # Max integration steps
    skip_iters: int = 2000
    contact_velocity: float = 0.6
    height_offset: float = 0.06
    yaw_only_feet: bool = False
    joint_limit_scale: float = 0.7
    contact_target_lerp: float = 0.3
    step_dt: float = 1 / 60
    barrier_gain : float = 100.0  # 障碍增益基础值
    safety_margin : float = 0.02  # 安全距离(m)
    foot_clearance : float = 0.15  # 脚部离地高度阈值(m)
    balance_adjustment : float = 0.05  # 平衡调整幅度(m)
    task_weights = { # 任务权重配置
        "joint_velocity": 2.0,
        "max_joint_acceleration": np.inf,
        "max_root_lin_acceleration": np.inf,
        "max_root_ang_acceleration": np.inf,
        "position": {
            "root": 2.0,
            "foot": 10.0,
            "hand": 4.0,
            "knee": 4.0,
            "elbow": 4.0,
        },
        "rotation": {
            "root": 5.0,
            "foot": 1.0,
        },
    }
    body_to_model_map = {
        "root": "",
        "left_hip": "",
        "right_hip": "",
        "left_knee": "",
        "right_knee": "",
        "left_foot": "",
        "right_foot": "",
        "left_hand": "",
        "right_hand": "",
        "left_elbow": "",
        "right_elbow": "",
        "left_shoulder": "",
        "right_shoulder": "",
    }
    extra_bodies = []
