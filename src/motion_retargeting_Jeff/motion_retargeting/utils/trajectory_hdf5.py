import os
import h5py
import numpy as np
import quaternion
from dataclasses import dataclass
from typing import List, Dict, Any
from motion_retargeting.utils.trajectory import Trajectory
from motion_retargeting.utils.mujoco.renderer import MujocoRenderer
import time

# version: 版本信息
# env: bvh_retarget
# robot: 机器人名称
# total_episodes: 总运动轨迹数
# total_steps 所有运动轨迹的总帧数
# /data/demo_<index>/num_samples：帧数
# /data/demo_<index>/motion_name：运动名称
# /data/demo_<index>/fps：运动的频率
# /data/demo_<index>/obs/joint_positions: 关节位置 (n_frames × n_joints)
# /data/demo_<index>/obs/joint_velocities: 关节速度(n_frames × n_joints)
# /data/demo_<index>/obs/joint_order:关节顺序(n_joints+2) 'universe', 'float_base' ...
# /data/demo_<index>/obs/ee_pos: 末端(脚)执行器位置 (n_frames × 2 * 3)
# /data/demo_<index>/obs/ee_quat: 末端(脚)执行器方向 (n_frames × 2 * 4)
# /data/demo_<index>/obs/root_pos: 根节点位置 (n_frames × 3)
# /data/demo_<index>/obs/root_quat: 根节点方向 (n_frames × 4)
# /data/demo_<index>/obs/contacts: 脚部接触状态 (n_frames × 2)
# /data/demo_<index>/actions: 动作数据 (零填充)
# /data/demo_<index>/rewards: 奖励数据 (零填充)
# /data/demo_<index>/dones: 片段结束标志
# /data/demo_<index>/timestamps: 时间戳
# /data/demo_<index>/body_aliases: 机器人的mjcf映射名称

@dataclass
class HDF5Recorder:
    """用于将机器人运动数据保存为RoboMimic兼容的HDF5格式"""
    file_path: str
    robot_name: Any
    fps: float = 240.0
    
    def __post_init__(self):
        # 初始化 HDF5 文件和数据组，原本在 __enter__ 中的内容
        self.hdf5_file = h5py.File(self.file_path, "w")
        # 添加文件级元数据
        self.hdf5_file.attrs["version"] = "1.0.0"
        self.hdf5_file.attrs["env"] = "bvh_retarget"
        self.hdf5_file.attrs["robot"] = self.robot_name

        # 创建 data 组
        self.data_grp = self.hdf5_file.create_group("data")
        self.episode_idx = 0

    # def __enter__(self):
    #     self.hdf5_file = h5py.File(self.file_path, "w")
    #     # 添加文件级元数据
    #     self.hdf5_file.attrs["version"] = "1.0.0"
    #     self.hdf5_file.attrs["env"] = "bvh_retarget"
    #     self.hdf5_file.attrs["robot"] = self.robot_name
        
    #     # 创建data组
    #     self.data_grp = self.hdf5_file.create_group("data")
    #     self.episode_idx = 0
    #     return self
    
    # def __exit__(self, exc_type, exc_val, exc_tb):
    #     # 添加全局元数据
    #     self.hdf5_file.attrs["total_episodes"] = self.episode_idx
    #     total_steps = sum(grp.attrs["num_samples"] for grp in self.data_grp.values())
    #     self.hdf5_file.attrs["total_steps"] = total_steps
    #     self.hdf5_file.close()
    
    def add_episode(self, motion_name: str, trajectory: Trajectory):
        """添加一个运动片段到HDF5文件"""
        
        # 转换轨迹为字典格式
        traj_dict = trajectory.to_dict(out_dt=1/self.fps)
        
        # 提取关键数据
        joint_positions = traj_dict["joint_positions"]
        joint_velocities = traj_dict["joint_velocities"]
        joint_order = traj_dict["joint_order"]
        contacts = traj_dict["contacts"]
        body_aliases = traj_dict["body_aliases"]
        T = len(joint_positions)  # 轨迹长度
        # 提取根节点和末端执行器数据
        root_name = traj_dict["model_root"]
        transforms = traj_dict["transforms"]
        
        # 提取根节点位置和方向
        root_positions = np.array([transforms[root_name]["position"][i] for i in range(T)])
        root_orientations = np.array([transforms[root_name]["quaternion"][i] for i in range(T)])
        # 提取末端执行器数据
        ee_names = [name for name in transforms if "foot" in name]
        ee_positions = []
        ee_orientations = []
        
        for i in range(T):
            ee_pos = []
            ee_quat = []
            for name in ee_names:
                ee_pos.extend(transforms[name]["position"][i])
                ee_quat.extend(transforms[name]["quaternion"][i])
            ee_positions.append(ee_pos)
            ee_orientations.append(ee_quat)
        
        ee_positions = np.array(ee_positions)
        ee_orientations = np.array(ee_orientations)
        
        # 创建episode组
        episode_grp = self.data_grp.create_group(f"demo_{self.episode_idx}")
        episode_grp.attrs["num_samples"] = T
        episode_grp.attrs["motion_name"] = motion_name
        episode_grp.attrs["fps"] = self.fps
        
        body_grp = episode_grp.create_group("body_aliases")
        
        for key,value in body_aliases.items():
            body_grp.attrs[key] = value
        
        # 创建观测组
        obs_grp = episode_grp.create_group("obs")        
        
        # 保存关节位置
        obs_grp.create_dataset("joint_positions", data=joint_positions, dtype="float32")
        
        # 保存关节速度
        obs_grp.create_dataset("joint_velocities", data=joint_velocities, dtype="float32")
        
        obs_grp.create_dataset("joint_order", data= np.bytes_(joint_order))
        
        # 保存末端执行器位置
        obs_grp.create_dataset("ee_pos", data=ee_positions, dtype="float32")
        
        # 保存末端执行器方向
        obs_grp.create_dataset("ee_quat", data=ee_orientations, dtype="float32")
        
        # 保存根节点位置
        obs_grp.create_dataset("root_pos", data=root_positions, dtype="float32")
        
        # 保存根节点方向
        obs_grp.create_dataset("root_quat", data=root_orientations, dtype="float32")
        
        # 保存接触状态
        obs_grp.create_dataset("contacts", data=contacts, dtype="bool")
        
        # 创建动作数据集（零填充）
        actions = np.zeros((T, joint_positions.shape[1]), dtype="float32")
        episode_grp.create_dataset("actions", data=actions, dtype="float32")
        
        # 创建奖励数据集（零填充）
        rewards = np.zeros(T, dtype="float32")
        episode_grp.create_dataset("rewards", data=rewards, dtype="float32")
        
        # 创建完成标志数据集
        dones = np.zeros(T, dtype="bool")
        dones[-1] = True
        episode_grp.create_dataset("dones", data=dones, dtype="bool")
        
        # 创建时间戳数据集
        timestamps = np.arange(T) / self.fps
        episode_grp.create_dataset("timestamps", data=timestamps, dtype="float32")
        
        self.episode_idx += 1
        print(f"saving: {motion_name}, Frames: {T}")

    def close(self):
        """手动关闭 HDF5 文件"""
        self.hdf5_file.attrs["total_episodes"] = self.episode_idx
        total_steps = sum(grp.attrs["num_samples"] for grp in self.data_grp.values())
        self.hdf5_file.attrs["total_steps"] = total_steps
        self.hdf5_file.close()
        print(f"✅ HDF5 文件已保存: {self.file_path}, 总片段: {self.episode_idx}, 总步数: {total_steps}")

        # 可选：析构时自动关闭（但不保证一定执行，建议显式调用 close()）
    def __del__(self):
        if hasattr(self, 'hdf5_file') and not self.hdf5_file.id.closed:
            self.close()

@dataclass
class HDF5Player:
    """从HDF5文件读取并播放机器人运动数据"""
    file_path: str
    robot_name: Any
    
    def __enter__(self):
        self.hdf5_file = h5py.File(self.file_path, "r")
        self.data_grp = self.hdf5_file["data"]
        self.episode_names = list(self.data_grp.keys())
        self.current_episode = 0
        self.current_frame = 0
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.hdf5_file.close()
    
    def get_episode_count(self):
        """获取运动片段数量"""
        return len(self.episode_names)
    
    def load_episode(self, episode_idx: int):
        """加载指定运动片段"""
        if episode_idx < 0 or episode_idx >= len(self.episode_names):
            raise ValueError(f"无效的运动片段索引: {episode_idx}")
        
        self.current_episode = episode_idx
        self.current_frame = 0
        episode_name = self.episode_names[episode_idx]
        self.episode_grp = self.data_grp[episode_name]
        
        # 加载元数据
        self.motion_name = self.episode_grp.attrs["motion_name"]
        self.fps = self.episode_grp.attrs["fps"]
        self.num_frames = self.episode_grp.attrs["num_samples"]
        
        # 加载观测数据
        obs_grp = self.episode_grp["obs"]
        self.joint_positions = obs_grp["joint_positions"][:]
        self.root_positions = obs_grp["root_pos"][:]
        self.root_orientations = obs_grp["root_quat"][:]
        self.ee_positions = obs_grp["ee_pos"][:]
        self.ee_orientations = obs_grp["ee_quat"][:]
        self.contacts = obs_grp["contacts"][:]
        
        # 加载时间戳
        self.timestamps = self.episode_grp["timestamps"][:]
        
        print(f"loading {episode_idx}: {self.motion_name}, frames: {self.num_frames}")
    
    def get_current_frame(self):
        """获取当前帧的机器人配置"""
        if self.current_frame >= self.num_frames:
            return None
        
        # 构建完整的q向量 (根节点位置+方向+关节角度)
        q = np.zeros(7 + self.joint_positions.shape[1])
        q[:3] = self.root_positions[self.current_frame]  # 位置
        q[3:7] = self.root_orientations[self.current_frame]  # 方向 (四元数)
        q[7:] = self.joint_positions[self.current_frame]  # 关节角度
        
        return q
    
    def advance_frame(self):
        """前进到下一帧"""
        self.current_frame += 1
        if self.current_frame >= self.num_frames:
            return False
        return True
    
    def reset(self):
        """重置到当前运动片段的开始"""
        self.current_frame = 0