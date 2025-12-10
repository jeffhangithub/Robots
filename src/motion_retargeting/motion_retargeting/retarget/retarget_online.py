import numpy as np
import sys, os, re
import re
import glob
from scipy.spatial.transform import Rotation
from motion_retargeting.utils.quat_utils import yaw_matrix
from motion_retargeting.utils.mapped_ik import MappedIK

class BVHRetargetOnline(MappedIK):
    """专门用于在线处理的BVH重定向类"""
    def __init__(
        self,
        bvh_dataset_fps,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.frame_idx = 0
        self.current_frame = None  # 改为存储当前帧而不是帧列表
        self.skeleton = {}
        self.bvh_dataset_fps = bvh_dataset_fps
        self.scales = self.wbik_params.scales

    def rescale_skeleton(self):
        """重新缩放骨架"""
        for joint, scale in zip(self.skeleton.values(), self.scales):
            joint.offset *= scale
  
    def reset_scales(self, scales):
        """重置缩放系数"""
        self.scales = scales
        
    def set_skeleton(self, skeleton):
        """设置骨架结构（在线处理只需设置一次）"""
        self.skeleton = skeleton
        self.rescale_skeleton()
        self.reset()

    def set_current_frame(self, frame_data):
        """设置当前帧数据（在线处理）"""
        if frame_data is None:
            return False
            
        self.current_frame = frame_data
        
        # 检查Hips关节是否存在
        if 'Hips' not in self.current_frame:
            return False
            
        # 设置当前帧运动
        self.skeleton["Hips"].set_motion(self.current_frame)
        return True

    def get_height_offset(self):
        """获取高度偏移（在线处理简化版）"""
        if self.current_frame is None:
            return 0.0
            
        min_height = np.inf
        self.skeleton["Hips"].set_motion(self.current_frame)
        
        min_height = min(
            min_height,
            min(
                self.get_dataset_position(self.wbik_params.body_to_data_map["left_foot"])[2],
                self.get_dataset_position(self.wbik_params.body_to_data_map["right_foot"])[2],
            ),
        )
        self.skeleton["Hips"].set_zero()
        return min_height

    def get_dataset_position(self, body_name: str):
        """获取数据集位置"""
        return self.skeleton[body_name].position

    def get_dataset_rotation(self, body_name: str):
        """获取数据集旋转"""
        return self.skeleton[body_name].R

    def __len__(self):
        return 1  # 在线处理只有当前帧

    def step_frame(self):
        """在线处理不需要步进帧"""
        pass

    def process_frame(self, frame_data):
        """处理单帧数据（在线处理的核心方法）"""
        if not self.set_current_frame(frame_data):
            return None
            
        # 获取重定向结果
        retargeted_positions = {}
        retargeted_rotations = {}
        
        for joint_name in self.skeleton.keys():
            try:
                pos = self.get_dataset_position(joint_name)
                rot = self.get_dataset_rotation(joint_name)
                retargeted_positions[joint_name] = pos
                retargeted_rotations[joint_name] = rot
            except Exception as e:
                print(f"获取关节 {joint_name} 的重定向数据时出错: {e}")
                
        return retargeted_positions, retargeted_rotations