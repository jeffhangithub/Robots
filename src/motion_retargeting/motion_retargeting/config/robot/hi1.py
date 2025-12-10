from motion_retargeting.config.bvh_retarget_config import BVHRetargetConfig

# from robot_descriptions.g1_mj_description import MJCF_PATH
from motion_retargeting.utils.mujoco.model_editor import MJCFModelEditor
import numpy as np
import os
from ament_index_python import get_package_share_directory

# MJCF_PATH = os.path.join(os.path.dirname(__file__), "../../../robots/hi_1/urdf/hi_1.xml")
MJCF_PATH = os.path.join(get_package_share_directory('motion_retargeting'), '../../../../src/motion_retargeting/robots/hi_1/urdf/hi_1.xml')
# Add a new body to the torso link to serve as the center of the torso
editor = MJCFModelEditor.from_path(MJCF_PATH)
editor.add_body("torso_center", "TORSO", np.array([0, 0, 0.25]), np.array([1, 0, 0, 0]))
editor.compile()

save_path = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(save_path, f"{os.path.basename(MJCF_PATH).split('.')[0]}_edited.xml")

if not os.path.exists(save_path):
    os.makedirs(save_path)
editor.save(MODEL_PATH)


class HI1_BVH_CONFIG(BVHRetargetConfig):
    mjcf_path = MODEL_PATH
    joint_limit_scale: float = 0.9
    output_fps = 60
    step_dt = 1.0 / 120
    body_to_model_map = {
        "root": "torso_center",
        "left_hip": "ILIUM_L",
        "right_hip": "ILIUM_R",
        "left_knee": "SHANK_L",
        "right_knee": "SHANK_R",
        "left_foot": "ASTRAGALUS_L",
        "right_foot": "ASTRAGALUS_R",
        "left_hand": "WRIST_UPDOWN_L",
        "right_hand": "WRIST_UPDOWN_R",
        "left_elbow": "WRIST_REVOLUTE_L",
        "right_elbow": "WRIST_REVOLUTE_R",
        "left_shoulder": "SHOULDER_L",
        "right_shoulder": "SHOULDER_R",
    }
    scales = np.array([
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ]
    )
    # 第一个表示根节点
    body_links=[
        "TORSO",
        "SACRUM",
        "ILIUM_R",
        "ISCHIUM_R",
        "THIGH_R",
        "SHANK_R",
        "ASTRAGALUS_R",
        "FOOT_R",
        "ILIUM_L",
        "ISCHIUM_L",
        "THIGH_L",
        "SHANK_L",
        "ASTRAGALUS_L",
        "FOOT_L",
        "SCAPULA_R",
        "SHOULDER_R",
        "UPPERARM_R",
        "FOREARM_R",
        "WRIST_REVOLUTE_R",
        "WRIST_UPDOWN_R",
        "HAND_R",
        "SCAPULA_L",
        "SHOULDER_L",
        "UPPERARM_L",
        "FOREARM_L",
        "WRIST_REVOLUTE_L",
        "WRIST_UPDOWN_L",
        "HAND_L",
    ]

    body_parent_links=[
        "TORSO",
        "TORSO",
        "SACRUM",
        "ILIUM_R",
        "ISCHIUM_R",
        "THIGH_R",
        "SHANK_R",
        "ASTRAGALUS_R",
        "SACRUM",
        "ILIUM_L",
        "ISCHIUM_L",
        "THIGH_L",
        "SHANK_L",
        "ASTRAGALUS_L",
        "TORSO",
        "SCAPULA_R",
        "SHOULDER_R",
        "UPPERARM_R",
        "FOREARM_R",
        "WRIST_REVOLUTE_R",
        "WRIST_UPDOWN_R",
        "TORSO",
        "SCAPULA_L",
        "SHOULDER_L",
        "UPPERARM_L",
        "FOREARM_L",
        "WRIST_REVOLUTE_L",
        "WRIST_UPDOWN_L",
    ]

    bvh_axis= {
        "SCAPULA_R":[0,1,0],
        "SHOULDER_R":[1,0,0],
        "FOREARM_R":[0,1,0],
        "WRIST_UPDOWN_R": [0,1,0],
        "HAND_R": [1,0,0],
        "SCAPULA_L":[0,1,0],
        "SHOULDER_L":[1,0,0],
        "FOREARM_L":[0,1,0],
        "WRIST_UPDOWN_L":[0,1,0],
        "HAND_L":[1,0,0],
        "ISCHIUM_R":[1,0,0],
        "THIGH_R":[0,1,0],
        "SHANK_R":[0,1,0],
        "ASTRAGALUS_R":[0,1,0],
        "FOOT_R":[-1,0,0],
        "ISCHIUM_L":[1,0,0],
        "THIGH_L":[0,1,0],
        "SHANK_L":[0,1,0],
        "ASTRAGALUS_L":[0,1,0],
        "FOOT_L": [1,0,0]
    }
    
    extra_bodies=[]
    for mm in body_links:
        if mm not in body_to_model_map.values():
            extra_bodies.append(mm)
    # scales = np.array(
    # [        
    #     1.0,
    #     0.9837591052055359,
    #     0.9810693860054016,
    #     0.9834277629852295,
    #     0.9834263920783997,
    #     1.0,
    #     1.0,
    #     1.0245699882507324,
    #     1.0191670656204224,
    #     0.9352324604988098,
    #     0.9556044340133667,
    #     1.0245699882507324,
    #     1.0191670656204224,
    #     0.9352324604988098,
    #     0.9556044340133667,
    #     1.0082836151123047,
    #     0.9423437714576721,
    #     0.9636573791503906,
    #     1.0,
    #     1.0082836151123047,
    #     0.9423437714576721,
    #     0.9636573791503906,
    #     1.0,

    # ],
    # )
    task_weights = {
        "joint_velocity": 2.0,
        "max_joint_acceleration": np.inf,
        "max_root_lin_acceleration": np.inf,
        "max_root_ang_acceleration": np.inf,
        "position": {
            "root": 15.0,  # 增加根节点位置权重
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


