from motion_retargeting.config.bvh_retarget_config import BVHRetargetConfig

import numpy as np
import os

# 使用本地路径而非 robot_descriptions（避免网络依赖）
MJCF_PATH = os.path.join(os.path.dirname(__file__), "../../../robots/g1/urdf/g1.xml")

# 尝试用 Mujoco.MjSpec 生成带 torso_center 的编辑版；若不支持（如无 MjSpec），回退到原始 MJCF。
try:
    from motion_retargeting.utils.mujoco.model_editor import MJCFModelEditor

    editor = MJCFModelEditor.from_path(MJCF_PATH)
    editor.add_body("torso_center", "torso_link", np.array([0, 0, 0.25]), np.array([1, 0, 0, 0]))
    editor.compile()

    save_path = os.path.join(os.path.dirname(__file__), "models")
    MODEL_PATH = os.path.join(save_path, f"{os.path.basename(MJCF_PATH).split('.')[0]}_edited.xml")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    editor.save(MODEL_PATH)
except Exception:
    MODEL_PATH = MJCF_PATH

class G1_BVH_CONFIG(BVHRetargetConfig):
    template_scale = 0.8013491034507751
    mjcf_path = MODEL_PATH
    joint_limit_scale: float = 0.9
    output_fps = 60
    body_to_model_map = {
        # 若无编辑版 torso_center，则使用 pelvis 作为根
        "root": "pelvis",
        "left_hip": "left_hip_roll_link",
        "right_hip": "right_hip_roll_link",
        "left_knee": "left_knee_link",
        "right_knee": "right_knee_link",
        "left_foot": "left_ankle_roll_link",
        "right_foot": "right_ankle_roll_link",
        "left_hand": "left_wrist_roll_link",
        "right_hand": "right_wrist_roll_link",
        "left_elbow": "left_elbow_link",
        "right_elbow": "right_elbow_link",
        "left_shoulder": "left_shoulder_roll_link",
        "right_shoulder": "right_shoulder_roll_link",
    }
    scales = np.array(
    [
        1.0,
        0.9801662564277649,
        0.9795798659324646,
        0.980391800403595,
        0.9805275797843933,
        1.0,
        1.0,
        1.010497808456421,
        0.9874637722969055,
        0.9516863226890564,
        0.9719527959823608,
        1.010497808456421,
        0.9874637722969055,
        0.9516863226890564,
        0.9719527959823608,
        0.9958724975585938,
        0.9268315434455872,
        0.9630832672119141,
        1.0,
        0.9958724975585938,
        0.9268315434455872,
        0.9630832672119141,
        1.0,
    ],
    )

    # 第一个表示根节点
    body_links=[
        "pelvis",
        "left_hip_pitch_link",
        "left_hip_roll_link",
        "left_hip_yaw_link",
        "left_knee_link",
        "left_ankle_pitch_link",
        "left_ankle_roll_link",
        "right_hip_pitch_link",
        "right_hip_roll_link",
        "right_hip_yaw_link",
        "right_knee_link",
        "right_ankle_pitch_link",
        "right_ankle_roll_link",
        "waist_yaw_link",
        "waist_roll_link",
        "torso_link",
        "left_shoulder_pitch_link",
        "left_shoulder_roll_link",
        "left_shoulder_yaw_link",
        "left_elbow_link",
        "left_wrist_roll_link",
        "left_wrist_pitch_link",
        "left_wrist_yaw_link",
        "right_shoulder_pitch_link",
        "right_shoulder_roll_link",
        "right_shoulder_yaw_link",
        "right_elbow_link",
        "right_wrist_roll_link",
        "right_wrist_pitch_link",
        "right_wrist_yaw_link",
    ]

    body_parent_links=[
        "pelvis",
        "pelvis",
        "left_hip_pitch_link",
        "left_hip_roll_link",
        "left_hip_yaw_link",
        "left_knee_link",
        "left_ankle_pitch_link",
        "pelvis",
        "right_hip_pitch_link",
        "right_hip_roll_link",
        "right_hip_yaw_link",
        "right_knee_link",
        "right_ankle_pitch_link",
        "pelvis",
        "waist_yaw_link",
        "waist_roll_link",
        "torso_link",
        "left_shoulder_pitch_link",
        "left_shoulder_roll_link",
        "left_shoulder_yaw_link",
        "left_elbow_link",
        "left_wrist_roll_link",
        "left_wrist_pitch_link",
        "torso_link",
        "right_shoulder_pitch_link",
        "right_shoulder_roll_link",
        "right_shoulder_yaw_link",
        "right_elbow_link",
        "right_wrist_roll_link",
        "right_wrist_pitch_link"
    ]

    bvh_axis= {
        "left_hip_pitch_link":[0,1,0],
        "left_hip_roll_link":[1,0,0],
        "left_knee_link":[0,1,0],
        "left_ankle_pitch_link": [0,1,0],
        "left_ankle_roll_link": [1,0,0],
        "right_hip_pitch_link":[0,1,0],
        "right_hip_roll_link":[1,0,0],
        "right_knee_link":[0,1,0],
        "right_ankle_pitch_link":[0,1,0],
        "right_ankle_roll_link":[1,0,0],
        "waist_roll_link":[1,0,0],
        "torso_link":[0,1,0],
        "left_shoulder_pitch_link":[0,1,0],
        "left_shoulder_roll_link":[1,0,0],
        "left_elbow_link":[0,1,0],
        "left_wrist_roll_link":[1,0,0],
        "left_wrist_pitch_link":[0,1,0],
        "right_shoulder_pitch_link":[0,1,0],
        "right_shoulder_roll_link":[1,0,0],
        "right_elbow_link": [0,1,0],
        "right_wrist_roll_link":[1,0,0],
        "right_wrist_pitch_link":[0,1,0],
    }
    
    extra_bodies=[]
    for mm in body_links:
        if mm not in body_to_model_map.values():
            extra_bodies.append(mm)
    
    # scales = np.array(
    # [
    #     1.0,
    #     1.0,
    #     1.0,
    #     1.0,
    #     1.0,
    #     1.0,
    #     1.0,
    #     1.0,
    #     1.0,
    #     1.0,
    #     1.0,
    #     1.0,
    #     1.0,
    #     1.0,
    #     1.0,
    #     1.0,
    #     1.0,
    #     1.0,
    #     1.0,
    #     1.0,
    #     1.0,
    #     1.0,
    #     1.0,
    # ],
    # )
