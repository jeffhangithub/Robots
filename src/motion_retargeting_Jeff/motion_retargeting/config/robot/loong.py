from motion_retargeting.config.bvh_retarget_config import BVHRetargetConfig

import numpy as np
from motion_retargeting.utils.mujoco.model_editor import MJCFModelEditor
import os
from ament_index_python import get_package_share_directory

# 青龙机器人关节说明
# "Link_head_yaw"   头部绕Y轴旋转（偏航，Yaw）
# "Link_head_pitch" 头部绕X轴俯仰（俯仰，Pitch）

# 可能对应从肩部到末端执行器（如手部或工具）的7个关节，常见自由度包括：
# "Link_arm_r_01"       肩部旋转（Roll）
# "Link_arm_r_02"       肩部旋转（Pitch）
# "Link_arm_r_03"       肩部旋转（Yaw）
# "Link_arm_r_04"       肘部弯曲（Pitch）
# "Link_arm_r_05"       前臂旋转（Roll）
# "Link_arm_r_06"       腕部弯曲（Pitch）
# "Link_arm_r_07"       腕部弯曲（Roll）
# "Link_arm_l_01"       肩部旋转（Roll）
# "Link_arm_l_02"       肩部旋转（Pitch）
# "Link_arm_l_03"       肩部旋转（Yaw）
# "Link_arm_l_04"       肘部弯曲（Pitch）
# "Link_arm_l_05"       前臂旋转（Roll）
# "Link_arm_l_06"       腕部弯曲（Pitch）
# "Link_arm_l_07"       腕部弯曲（Roll）
# "Link_waist_pitch"    腰部俯仰（Pitch）
# "Link_waist_roll"     腰部侧倾（Roll）
# "Link_waist_yaw"      腰部偏航（Yaw）
# "Link_hip_r_roll"     髋部侧倾（Roll）
# "Link_hip_r_yaw"      髋部偏航（Yaw）
# "Link_hip_r_pitch"    髋部俯仰（Pitch）
# "Link_knee_r_pitch"   膝关节弯曲（Pitch）
# "Link_ankle_r_pitch"  踝关节俯仰（Pitch）
# "Link_ankle_r_roll"   踝关节侧倾（Roll）
# "Link_hip_l_roll"     髋部侧倾（Roll）
# "Link_hip_l_yaw"      髋部偏航（Yaw）
# "Link_hip_l_pitch"    髋部俯仰（Pitch）
# "Link_knee_l_pitch"   膝关节弯曲（Pitch）
# "Link_ankle_l_pitch"  踝关节俯仰（Pitch）
# "Link_ankle_l_roll"   踝关节侧倾（Roll）
# 典型的人体结构
# Head
# ├── Yaw (Link_head_yaw)
# └── Pitch (Link_head_pitch)

# Torso (Waist)
# ├── Yaw (Link_waist_yaw)
# ├── Pitch (Link_waist_pitch)
# └── Roll (Link_waist_roll)

# Right Arm
# ├── Shoulder Roll (Link_arm_r_01)
# ├── Shoulder Pitch (Link_arm_r_02)
# ├── Elbow Pitch (Link_arm_r_03)
# ├── Forearm Roll (Link_arm_r_04)
# ├── Wrist Pitch (Link_arm_r_05)
# └── Wrist Roll (Link_arm_r_06)

# Left Leg
# ├── Hip Roll (Link_hip_l_roll)
# ├── Hip Pitch (Link_hip_l_pitch)
# ├── Hip Yaw (Link_hip_l_yaw)
# ├── Knee Pitch (Link_knee_l_pitch)
# ├── Ankle Pitch (Link_ankle_l_pitch)
# └── Ankle Roll (Link_ankle_l_roll)

# MJCF_PATH = os.path.join(os.path.dirname(__file__), "../../../robots/openloong/AzureLoong.xml")
MJCF_PATH = os.path.join(get_package_share_directory('motion_retargeting'), '../../../../src/motion_retargeting/robots/openloong/AzureLoong.xml')
# Add a new body to the torso link to serve as the center of the torso
editor = MJCFModelEditor.from_path(MJCF_PATH)
editor.add_body("torso_center", "base_link", np.array([0, 0, 0.25]), np.array([1, 0, 0, 0]))

editor.compile()

save_path = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(save_path, f"{os.path.basename(MJCF_PATH).split('.')[0]}_edited.xml")

if not os.path.exists(save_path):
    os.makedirs(save_path)
editor.save(MODEL_PATH)


class LOONG_BVH_CONFIG(BVHRetargetConfig):
    template_scale = 0.8013491034507751
    mjcf_path = MODEL_PATH
    joint_limit_scale: float = 0.9
    output_fps = 60
    body_to_model_map = {
        "root": "torso_center",                             #"躯干中心",                   // 机器人运动学基准原点
        "left_hip": "Link_hip_l_roll",                      #"左髋滚动连杆",                // 左侧髋关节滚转自由度
        "right_hip": "Link_hip_r_roll",                     #"右髋滚动连杆",                // 右侧髋关节滚转自由度
        "left_knee": "Link_knee_l_pitch",                   #"左膝关节",                    // 左侧膝关节俯仰自由度
        "right_knee": "Link_knee_r_pitch",                  #"右膝关节",                    // 右侧膝关节俯仰自由度
        "left_foot": "Link_ankle_l_roll",                   #"左踝滚动连杆",                // 左侧踝关节滚转自由度
        "right_foot": "Link_ankle_r_roll",                  #"右踝滚动连杆",                // 右侧踝关节滚转自由度
        "left_hand": "Link_arm_l_05",                       #"左手零位连杆",                // 左手部基准坐标系
        "right_hand": "Link_arm_r_05",                      #"右手零位连杆",                // 右手部基准坐标系
        "left_elbow": "Link_arm_l_03",                      #"左肘俯仰连杆",                // 左侧肘关节俯仰自由度
        "right_elbow": "Link_arm_r_03",                     #"右肘俯仰连杆",                // 右侧肘关节俯仰自由度
        "left_shoulder": "Link_arm_l_01",                   #"左肩滚动连杆",                 // 左侧肩关节滚转自由度
        "right_shoulder": "Link_arm_r_01",                  #"右肩滚动连杆"                 // 右侧肩关节滚转自由度
    }
    scales = np.array(
    [
        1.0,
        0.9821683168411255,
        0.9798935651779175,
        0.9816139936447144,
        0.9815268516540527,
        1.0,
        1.0,
        1.0514346361160278,
        1.0018924474716187,
        0.906292200088501,
        0.9576401114463806,
        1.0514346361160278,
        1.0018924474716187,
        0.906292200088501,
        0.9576401114463806,
        1.006534457206726,
        0.9179723858833313,
        0.959964394569397,
        1.0,
        1.006534457206726,
        0.9179723858833313,
        0.959964394569397,
        1.0,

    ],
    )
