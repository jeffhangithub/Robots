from motion_retargeting.config.mapped_ik_config import MappedIKConfig


class BVHRetargetConfig(MappedIKConfig):    
    body_to_data_map: dict = {
        "root": "Chest4",
        "left_knee": "LeftKnee",
        "right_knee": "RightKnee",
        "left_foot": "LeftAnkle",
        "right_foot": "RightAnkle",
        "left_elbow": "LeftElbow",
        "right_elbow": "RightElbow",
        "left_hand": "LeftWrist",
        "right_hand": "RightWrist",
    }


    # body_to_data_map: dict = {
    #     "root": "Chest4",
    #     "left_knee": "LeftKnee",
    #     "right_knee": "RightKnee",
    #     "left_foot": "LeftAnkle",
    #     "right_foot": "RightAnkle",
    #     "left_elbow": "LeftElbow",
    #     "right_elbow": "RightElbow",
    #     # "left_hand": "LeftWrist",
    #     # "right_hand": "RightWrist",
    #     "left_shoulder": "LeftShoulder",
    #     "right_shoulder": "RightShoulder",
    # }