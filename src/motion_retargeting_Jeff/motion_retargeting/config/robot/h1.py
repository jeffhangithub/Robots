from motion_retargeting.config.bvh_retarget_config import BVHRetargetConfig

import numpy as np
# from robot_descriptions.h1_mj_description import MJCF_PATH
from motion_retargeting.utils.mujoco.model_editor import MJCFModelEditor
import os
from ament_index_python import get_package_share_directory

# MJCF_PATH = os.path.join(os.path.dirname(__file__), "../../../robots/h1_3/urdf/h1.xml")
MJCF_PATH = os.path.join(get_package_share_directory('motion_retargeting'), '../../../../src/motion_retargeting/robots/h1_3/urdf/h1.xml')
# Add a new body to the torso link to serve as the center of the torso
editor = MJCFModelEditor.from_path(MJCF_PATH)
editor.add_body("torso_center", "torso_link", np.array([0.0, 0, 0.4]), np.array([1, 0, 0, 0]))
editor.add_body("torso_middle", "torso_link", np.array([0.0, 0, 0.2]), np.array([1, 0, 0, 0]))
editor.add_body("spine", "torso_link", np.array([0.0, 0, 0.17]), np.array([1, 0, 0, 0]))
# Moving the elbow slightly backward to avoid singular configurations
editor.add_body(
    "right_elbow_center",
    "right_shoulder_yaw_link",
    np.array([-0.05, 0, -0.2]),
    np.array([1, 0, 0, 0]),
)
editor.add_body(
    "left_elbow_center",
    "left_shoulder_yaw_link",
    np.array([-0.05, 0, -0.2]),
    np.array([1, 0, 0, 0]),
)

editor.add_body(
    "right_hand",
    "right_elbow_link",
    np.array([0.26, 0, -0.025]),
    np.array([1, 0, 0, 0]),
)
editor.add_body("left_hand", "left_elbow_link", np.array([0.26, 0, -0.025]), np.array([1, 0, 0, 0]))

editor.compile()

save_path = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(save_path, f"{os.path.basename(MJCF_PATH).split('.')[0]}_edited.xml")

if not os.path.exists(save_path):
    os.makedirs(save_path)
editor.save(MODEL_PATH)


class H1_BVH_CONFIG(BVHRetargetConfig):
    mjcf_path = MODEL_PATH
    joint_limit_scale = 0.9
    output_fps = 60
    body_to_model_map = {
        "root": "spine",
        "left_hip": "left_hip_pitch_link",
        "right_hip": "right_hip_pitch_link",
        "left_knee": "left_knee_link",
        "right_knee": "right_knee_link",
        "left_foot": "left_ankle_link",
        "right_foot": "right_ankle_link",
        "left_hand": "left_hand",
        "right_hand": "right_hand",
        "left_elbow": "left_elbow_center",
        "right_elbow": "right_elbow_center",
        "left_shoulder": "left_shoulder_roll_link",
        "right_shoulder": "right_shoulder_roll_link",
    }
    scales = np.array(
    [
        1.0,
        0.9886474013328552,
        0.9873085618019104,
        0.9887232780456543,
        0.9888057708740234,
        1.0,
        1.0,
        1.0130506753921509,
        0.9883060455322266,
        0.9415378570556641,
        0.9727240800857544,
        1.0130506753921509,
        0.9883060455322266,
        0.9415378570556641,
        0.9727240800857544,
        1.0031288862228394,
        0.959779679775238,
        0.9843694567680359,
        1.0,
        1.0031288862228394,
        0.959779679775238,
        0.9843694567680359,
        1.0,

    ],
    )