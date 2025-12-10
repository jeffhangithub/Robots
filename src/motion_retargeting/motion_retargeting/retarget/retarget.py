"""
算法分析 (Algorithm Analysis):

本文件实现了基于BVH数据的运动重定向核心类 `BVHRetarget`，其主要算法逻辑如下：

1. **骨骼结构建模 (Skeleton Modeling)**:
   - 使用 `Joint` 类构建分层骨骼树结构，维护关节间的父子关系、偏移量(Offset)和自由度(DOF)。
   - 支持解析BVH标准格式的欧拉角旋转顺序 (YXZ)。

2. **正向运动学 (Forward Kinematics)**:
   - `set_motion` 方法根据输入的运动帧数据，递归计算每个关节的全局位置和旋转矩阵。
   - 变换公式: Global_transform = Parent_transform * Local_transform。

3. **坐标系变换 (Coordinate Transformation)**:
   - 内置了从动画软件坐标系（通常Y-up）到机器人仿真坐标系（通常Z-up）的转换矩阵 `R_conversion`。
   - 位置映射逻辑: (x, y, z)_BVH -> (z, x, y)_Robot。

4. **运动数据处理 (Motion Processing)**:
   - 支持帧率重采样 (Rescaling)，根据目标FPS对原始BVH数据进行跳帧处理。
   - 提供 `get_dataset_position` 和 `get_dataset_rotation` 接口，为下游的逆向运动学(IK)求解器提供重定向的目标位姿（Target Pose）。

5. **可视化调试 (Visualization)**:
   - 集成 Matplotlib 3D 绘图功能，支持按身体部位着色绘制骨架，用于验证运动数据的解析正确性。

该类通常作为重定向流水线中的"Teacher"端，负责提供标准的人体动作参考数据。
"""
import numpy as np
import sys, os, re
import re
import glob
from scipy.spatial.transform import Rotation
from motion_retargeting.utils.quat_utils import yaw_matrix
from motion_retargeting.utils.mapped_ik import MappedIK

ASF_TO_METERS = 0.01  # 用于单位转换的系数

class BVHRetarget(MappedIK):
    def __init__(
        self,
        bvh_dataset_fps,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.frame_idx = 0
        self.frames = []
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
      
        
    def visualize_skeleton(self, frame_idx=0, save_path=None, title="BVH Skeleton Visualization"):
        """
        可视化BVH骨架模型
        
        参数:
            frame_idx: 要可视化的帧索引
            save_path: 保存路径，None表示显示图像
            title: 图像标题
        """
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        from mpl_toolkits.mplot3d.art3d import Line3DCollection

        
        colormap = list(mcolors.TABLEAU_COLORS.values())

        # 检查帧索引是否有效
        if frame_idx < 0 or frame_idx >= len(self.frames):
            print(f"错误: 帧索引{frame_idx}超出范围(0-{len(self.frames)-1})")
            return
        
        # 设置当前帧运动
        self.skeleton["Hips"].set_motion(self.frames[frame_idx])
        
        # 收集所有关节的位置和父子关系
        positions = []
        joint_names = []
        parent_indices = []
        
        # 构建关节位置列表和父子关系
        for i, (name, joint) in enumerate(self.skeleton.items()):
            positions.append(joint.position)
            joint_names.append(name)
            
            # 查找父关节索引
            parent_index = -1
            if joint.parent:
                parent_name = joint.parent.name
                if parent_name in self.skeleton:
                    parent_index = list(self.skeleton.keys()).index(parent_name)
            parent_indices.append(parent_index)
        
        positions = np.array(positions)
        
        # 创建3D图形
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制关节点
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                   c='blue', s=80, alpha=0.8, zorder=10)
        
        # 绘制骨骼连接
        lines = []
        line_colors = []
        
        # 定义身体部位到颜色的映射
        part_colors = {
            "head": colormap[0],
            "neck": colormap[1],
            "back": colormap[2],
            "shoulder": colormap[3],
            "arm": colormap[4],
            "elbow": colormap[5],
            "hand": colormap[6],
            "hip": colormap[7],
            "leg": colormap[8],
            "knee": colormap[9],
            "foot": colormap[0]  # 循环使用颜色
        }
        
        # 构建骨骼连接线
        for i, (pos, parent_idx) in enumerate(zip(positions, parent_indices)):
            if parent_idx != -1:
                parent_pos = positions[parent_idx]
                lines.append([parent_pos, pos])
                
                # 根据关节名称确定颜色
                name = joint_names[i].lower()
                color = colormap[0]  # 默认颜色
                
                # 根据关节名称分配颜色
                if "head" in name:
                    color = part_colors["head"]
                elif "neck" in name:
                    color = part_colors["neck"]
                elif "back" in name or "thorax" in name or "chest" in name:
                    color = part_colors["back"]
                elif "shoulder" in name or "collar" in name:
                    color = part_colors["shoulder"]
                elif "elbow" in name:
                    color = part_colors["elbow"]
                elif "hand" in name or "wrist" in name:
                    color = part_colors["hand"]
                elif "hip" in name:
                    color = part_colors["hip"]
                elif "knee" in name:
                    color = part_colors["knee"]
                elif "foot" in name or "ankle" in name:
                    color = part_colors["foot"]
                elif "arm" in name or "radius" in name or "humerus" in name:
                    color = part_colors["arm"]
                elif "leg" in name or "femur" in name or "tibia" in name:
                    color = part_colors["leg"]
                
                line_colors.append(color)
        
        # 绘制骨骼
        if lines:
            lc = Line3DCollection(lines, colors=line_colors, linewidths=3.0, alpha=0.9)
            ax.add_collection3d(lc)
        
        # 添加关节标签
        for i, (pos, name) in enumerate(zip(positions, joint_names)):
            # 只标记主要关节以减少混乱
            if "Hips" in name or "Hip" in name or "Shoulder" in name or "Head" in name or "Hand" in name or "Foot" in name:
                ax.text(pos[0], pos[1], pos[2], name, 
                        color='black', fontsize=9, zorder=11, 
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))
        
        # 设置坐标轴
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(title)
        
        # 自动调整视图范围
        min_val = positions.min(axis=0)
        max_val = positions.max(axis=0)
        range_val = max_val - min_val
        max_range = max(range_val) * 1.2
        
        # 计算中心点
        center = (min_val + max_val) / 2
        
        # 设置等比例坐标轴
        ax.set_xlim(center[0] - max_range/2, center[0] + max_range/2)
        ax.set_ylim(center[1] - max_range/2, center[1] + max_range/2)
        ax.set_zlim(center[2] - max_range/2, center[2] + max_range/2)
        
        # 设置网格和背景
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')
        
        # 设置视角
        ax.view_init(elev=20, azim=45)
        
        # 添加图例说明
        legend_elements = [
            plt.Line2D([0], [0], color=part_colors["head"], lw=3, label='头部'),
            plt.Line2D([0], [0], color=part_colors["back"], lw=3, label='躯干'),
            plt.Line2D([0], [0], color=part_colors["arm"], lw=3, label='手臂'),
            plt.Line2D([0], [0], color=part_colors["leg"], lw=3, label='腿部'),
            plt.Line2D([0], [0], color=part_colors["hand"], lw=3, label='手部'),
            plt.Line2D([0], [0], color=part_colors["foot"], lw=3, label='脚部')
        ]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1))
        
        plt.tight_layout()
        
        # 显示或保存图像
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"骨架图像已保存至: {save_path}")
        else:
            plt.show()
        
        plt.close(fig)

    def set_motion(self, skeleton, motions):
        """设置BVH运动数据"""
        self.frame_idx = 0
        self.skeleton = skeleton
        self.rescale_skeleton()
        
        # 应用帧率调整
        frame_skip = np.clip(
            int(self.bvh_dataset_fps / self.wbik_params.output_fps),
            a_min=1,
            a_max=len(motions)
        )
        self.frames = motions[::frame_skip]
        # 获取初始帧的根节点位置
        # initial_root_pos = np.array(self.frames[0]["Hips"][:3])
                  
        # for i in range(len(self.frames)):
        #     self.frames[i]["Hips"][0] -= initial_root_pos[0]  # X轴
        #     self.frames[i]["Hips"][2] -= initial_root_pos[2]  # Z轴
            # self.frames[i]["Hips"][1] = self.frames[i]["Hips"][1] - height_offset + self.wbik_params.height_offset
            # self.frames[i] = self.frames[0]
        self.skeleton["Hips"].set_motion(self.frames[self.frame_idx])
        # self.apply_foot_correction()
        self.reset()

        # for i in range(len(self.frames)):
        #   self.visualize_skeleton(i, "/ssd/code/recap/data/test/"+str(i)+'.png')
                
    def apply_foot_correction(self):
        """应用脚部校正"""
        self.skeleton["Hips"].set_motion(self.frames[0])
        root_orientation = self.skeleton["Hips"].R
        
        foot_joints = [self.wbik_params.body_to_data_map["left_foot"],self.wbik_params.body_to_data_map["right_foot"]]
  
        self.foot_correction = {}
        for foot_joint in foot_joints:
            foot_orientation = self.skeleton[foot_joint].R
            
            rel_rotation = foot_orientation @ yaw_matrix(root_orientation.T)
            self.foot_correction[foot_joint] = rel_rotation
            self.skeleton[foot_joint].update_rel_rotation(rel_rotation)
        
    def get_height_offset(self, frames):
        """获取高度偏移"""
        min_height = np.inf
        for motion in frames:
            self.skeleton["Hips"].set_motion(motion)
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
        return len(self.frames)

    def step_frame(self):
        """前进到下一帧"""
        self.frame_idx += 1

        if self.frame_idx < len(self):
        # 设置当前帧运动
          self.skeleton["Hips"].set_motion(self.frames[self.frame_idx])
          # for key,value in self.foot_correction.items():
          #   self.skeleton[key].update_rel_rotation(value)


class Joint:
    """表示BVH中的一个关节"""
    def __init__(self, name, offset, dof, limits):
        self.name = name
        self.offset = offset  # BVH中的偏移量
        self.dof = dof  # 自由度/通道顺序
        self.limits = limits  # 旋转限制
        self.parent = None
        self.children = []
        self.coordinate = None
        self.matrix = np.eye(3)  # 初始化为单位矩阵
        self.rel_rotation = np.eye(3)
        # 坐标系转换矩阵 (Y-up to Z-up)
        self.R_conversion = np.array([
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0] 
        ], dtype=np.float32)
        self.R_conversion_inv = np.linalg.inv(self.R_conversion)
        self.height_offset = 0
        
    def update_rel_rotation(self, rel_rotation):
        self.rel_rotation = rel_rotation
    
    def set_motion(self, motion=None):
        """设置关节的运动数据"""
        if motion is None or self.name not in motion:
            motion_value = np.zeros(6)
        else:
            motion_value = motion[self.name]
        
        # 根节点处理
        if self.name == "Hips" and len(motion_value) >= 6:
            # 位置直接取自运动数据
            self.coordinate = np.array(motion_value[:3]).reshape(3, 1)
          
            angles = motion_value[3:6]
            # 构建旋转矩阵 (YXZ顺序)
            self.matrix = self.euler2mat_yxz(angles)
            
        # 非根节点处理
        elif self.parent is not None:
            # 提取旋转角度
            angles = []
            for channel in self.dof:
                if channel.endswith('rotation'):
                    idx = self.dof.index(channel)
                    angles.append(motion_value[idx])
            
            # 确保有3个旋转角度
            if len(angles) < 3:
                angles.extend([0] * (3 - len(angles)))
            
            # 构建旋转矩阵 (YXZ顺序)
            local_rot = self.euler2mat_yxz(angles)
            
            # 全局旋转矩阵 = 父节点旋转矩阵 × 局部旋转矩阵
            self.matrix = self.parent.matrix @ local_rot
            
            # 位置 = 父节点位置 + (父节点旋转矩阵 × 偏移量)
            self.coordinate = self.parent.coordinate + self.parent.matrix @ self.offset.reshape(3, 1)
        
        # 递归设置子节点
        for child in self.children:
            child.set_motion(motion)
      
    def euler2mat_yxz(self, angles):
        """YXZ顺序的欧拉角转旋转矩阵 (BVH标准)"""
        # 注意顺序：先Y，再X，最后Z
        y, x, z = angles
        
        # 计算各轴旋转矩阵
        R_y = np.array([
            [np.cos(y), 0, np.sin(y)],
            [0, 1, 0],
            [-np.sin(y), 0, np.cos(y)]
        ])
        
        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(x), -np.sin(x)],
            [0, np.sin(x), np.cos(x)]
        ])
        
        R_z = np.array([
            [np.cos(z), -np.sin(z), 0],
            [np.sin(z), np.cos(z), 0],
            [0, 0, 1]
        ])
        
        return R_y @ R_x @ R_z

    def set_zero(self):
        """重置关节到初始位置"""
        self.set_motion()
    
    @property
    def position(self):
        """获取关节位置（转换到机器人坐标系）"""
        if self.coordinate is None:
            return np.zeros(3)
        return self.coordinate[[2,0,1],:].squeeze()

    @property
    def R(self):
        """获取旋转矩阵（转换到机器人坐标系）"""
        # 应用坐标系转换
        return self.R_conversion @ self.matrix @ self.R_conversion_inv
        # @self.rel_rotation