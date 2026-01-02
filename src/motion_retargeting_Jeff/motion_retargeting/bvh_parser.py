import rclpy
from rclpy.node import Node
from typing import List, Dict, Optional, Tuple
import numpy as np
from geometry_msgs.msg import Point, TransformStamped, Quaternion, Pose
from sensor_msgs.msg import JointState
from xsens_mvn_ros_msgs.msg import LinkStateArray, LinkState
from std_msgs.msg import Header
import tf2_ros
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import re
from scipy.spatial.transform import Rotation as R

# 导入重定向模块
from motion_retargeting.retarget.retarget import BVHRetarget, Joint
from motion_retargeting.utils.trajectory_hdf5 import HDF5Recorder
from motion_retargeting.utils.trajectory import Trajectory
from motion_retargeting.utils.mujoco.renderer import MujocoRenderer
from motion_retargeting.config.robot.g1 import G1_BVH_CONFIG
from motion_retargeting.config.robot.h1 import H1_BVH_CONFIG
from motion_retargeting.config.robot.loong import LOONG_BVH_CONFIG
from motion_retargeting.config.robot.hi1 import HI1_BVH_CONFIG

import os
import glob
from pathlib import Path
import h5py
import pickle

# 单位转换常量
ASF_TO_METERS = 0.01

# 支持的机器人类型与对应配置
ROBOT_CONFIGS = {
    "g1": G1_BVH_CONFIG,
    "h1": H1_BVH_CONFIG,
    "loong": LOONG_BVH_CONFIG,
    "hi_1": HI1_BVH_CONFIG  #浙江人形
}

class BVHParser(Node):
    """优化的BVH文件解析器，仅获取根关节位置和旋转，其他关节仅获取相对旋转"""
    
    def __init__(self):
        super().__init__('bvh_parser')
        
        # 参数声明
        # self.declare_parameter('bvh_file_path', '/home/uneedrobot/workshops/source/xsens/ros2/data/bvh/dataset1/data/XSENS_LINK-walk.bvh')
        self.declare_parameter('bvh_data_root', '/home/uneedrobot/workshops/source/xsens/ros2/data/bvh/dataset_zjh') #注意，最后不要带/
        self.declare_parameter('publish_rate', 40.0)  # Hz
        self.declare_parameter('reference_frame', 'world')
        self.declare_parameter('loop_playback', True)
        self.declare_parameter('robot_name', 'hi_1')
        self.declare_parameter('enable_retargeting', True)
        self.declare_parameter('bvh_dataset_fps', 40)

        self.declare_parameter('output_dir', '/home/uneedrobot/workshops/source/xsens/ros2/results')  # HDF5 输出目录
        self.declare_parameter('record_pickle', True)  # 是否生成pickle文件
        self.declare_parameter('record_hdf5', True)  # 是否记录 HDF5
        self.declare_parameter('skip_motions', '')  # 跳过指定动作，多个用空格隔开
        self.declare_parameter('enable_render', True)  # 是否启用渲染（需MJCF+MuJoCo）


        # ======== 参数获取 ========
        # self.bvh_file_path = self.get_parameter('bvh_file_path').value
        self.bvh_data_root = self.get_parameter('bvh_data_root').value
        self.publish_rate = self.get_parameter('publish_rate').value
        self.reference_frame = self.get_parameter('reference_frame').value
        self.loop_playback = self.get_parameter('loop_playback').value
        self.robot_name = self.get_parameter('robot_name').value
        self.enable_retargeting = self.get_parameter('enable_retargeting').value
        self.bvh_dataset_fps = self.get_parameter('bvh_dataset_fps').value

        self.output_dir = self.get_parameter('output_dir').value
        self.record_hdf5 = self.get_parameter('record_hdf5').value
        self.skip_motions_str = self.get_parameter('skip_motions').value
        self.enable_render = self.get_parameter('enable_render').value
        self.record_pickle = self.get_parameter('record_pickle').value
        self.skip_motions = self.skip_motions_str.split() if self.skip_motions_str else []
        # ======== 机器人配置与 Retarget 初始化 ========
        # self.robot_config = ROBOT_CONFIGS.get(self.robot_name, G1_BVH_CONFIG)
        self.robot_config = ROBOT_CONFIGS.get(self.robot_name, {})
        self.bvh_axis = {}
        if hasattr(self.robot_config, 'bvh_axis'):
            self.bvh_axis = self.robot_config.bvh_axis
        
        self.retargeter: Optional[BVHRetarget] = None
        self.retargeter_flag = False

        if self.enable_retargeting:
            self.retargeter = BVHRetarget(
                bvh_dataset_fps=self.bvh_dataset_fps,
                wbik_params=self.robot_config
            )
            self.retargeter_flag = True
            self.get_logger().info(f"BVHRetarget initialized for robot: {self.robot_name}")
        
        
        # 发布器 - 模拟XSens的数据流
        qos_profile = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )
        
        # 统一的重定向数据发布器
        self.retargeted_joint_pub = self.create_publisher(
            JointState, 
            '/retargeted_joint_states', 
            qos_profile
        )
        
        self.retargeted_link_pub = self.create_publisher(
            LinkStateArray, 
            '/retargeted_link_states', 
            qos_profile
        )

        self.hdf5_path = ""
        self.recorder = None
        self.trajectory = None
        self.current_motion_name = ""
        self.pick_out = ""

        # ======== 创建输出目录 ========
        if self.record_hdf5:
            self.hdf5_path = os.path.join(self.output_dir, f"{self.robot_name}_motions.hdf5")

            os.makedirs(os.path.dirname(self.hdf5_path), exist_ok=True)
            self.recorder = HDF5Recorder(self.hdf5_path, self.robot_name, self.bvh_dataset_fps)
            self.get_logger().info(f"📝 HDF5 将记录到: {self.hdf5_path}")
        if self.record_pickle:
            data_menu = self.bvh_data_root.split('/')[-1]
            self.pick_out = os.path.join(self.output_dir, os.path.join(self.robot_name,data_menu))
            os.makedirs(self.pick_out, exist_ok=True)

        # ========== 加载 BVH 文件列表 ==========
        self.bvh_files = self._scan_bvh_files()
        if not self.bvh_files:
            self.get_logger().error("❌ 未找到任何 .bvh 文件，请检查 bvh_data_root 参数")
            self.destroy_node()
            return

        self.get_logger().info(f"📂 找到 {len(self.bvh_files)} 个 BVH 文件，将从第一个开始实时播放。")

        self.current_bvh_idx = 0
        self.current_frame = 0
        self.skeleton = None
        self.motion_data = None
        self.renderer = None 
        
        # ======== 定时器发布数据 ========
        self.timer = self.create_timer(1.0 / self.publish_rate, self.publish_frame_data)

    def _scan_bvh_files(self) -> List[str]:
        pattern = os.path.join(self.bvh_data_root, "**", "*.bvh")
        files = glob.glob(pattern, recursive=True)
        # 可选：按 skip_motions 过滤
        if self.skip_motions:
            files = [f for f in files if not any(skip in os.path.basename(f) for skip in self.skip_motions)]
        return files


    def load_bvh_file(self, file_path: str) -> Tuple[bool, Optional[object], Optional[list], int, Optional[float]]:
        """读取BVH文件并解析骨骼结构和运动数据"""
        try:
            with open(file_path, 'r') as f:
                content = f.readlines()
            
            # 解析层次结构
            hierarchy_start = content.index("HIERARCHY\n")
            hierarchy_end = content.index("MOTION\n")
            hierarchy = content[hierarchy_start:hierarchy_end]

            # 解析运动数据
            motion = content[hierarchy_end:]
            frames_line = next(line for line in motion if line.startswith("Frames:"))
            frame_count = int(frames_line.split()[1])
            frame_time_line = next(line for line in motion if line.startswith("Frame Time:"))
            frame_time = float(frame_time_line.split()[2])
            
            # 解析关节偏移量和父子关系
            joints = {}
            parents = {}
            stack = []
            current_joint = None
            
            for line in hierarchy:
                if "ROOT" in line or "JOINT" in line:
                    name = line.split()[1]
                    joints[name] = {"offset": None, "channels": []}
                    if current_joint:
                        parents[name] = current_joint
                        stack.append(current_joint)
                    current_joint = name
                elif "End Site" in line:
                    name = f"{current_joint}_end"
                    joints[name] = {"offset": None, "channels": []}
                    parents[name] = current_joint
                    current_joint = name
                    stack.append(current_joint)
                elif "OFFSET" in line:
                    offset = list(map(float, line.split()[1:4]))
                    joints[current_joint]["offset"] = np.array(offset)
                elif "CHANNELS" in line:
                    channels = line.split()[2:]
                    joints[current_joint]["channels"] = channels
                elif "}" in line:
                    if stack:
                        current_joint = stack.pop()

            # 解析运动数据
            motion_data = []
            data_lines = [line.split() for line in motion if line.strip() and not line.startswith(("Frames:", "Frame Time:"))]
            
            for i,line in enumerate(data_lines):
                if i == 1 or len(line) < 6:  # 确保有足够的数据
                    continue
                    
                frame_data = {}
                idx = 0
                
                for joint in joints:
                    if joint.endswith("_end"):
                        continue
                        
                    channels = joints[joint]["channels"]
                    if not channels:
                        continue
                        
                    data = []
                    for channel in channels:
                        if channel == "Xposition":
                            data.append(float(line[idx]) * ASF_TO_METERS)
                        elif channel == "Yposition":
                            data.append(float(line[idx]) * ASF_TO_METERS)
                        elif channel == "Zposition":
                            data.append(float(line[idx]) * ASF_TO_METERS)
                        else:  # 旋转通道
                            data.append(np.deg2rad(float(line[idx])))
                        idx += 1
                    
                    frame_data[joint] = data
                # print("    frame_data:      ", frame_data)
                motion_data.append(frame_data)

            # 创建骨架结构
            skeleton = {}
            for name, data in joints.items():
                if name.endswith("_end"):
                    continue
                    
                offset = data["offset"] * ASF_TO_METERS if data["offset"] is not None else np.zeros(3)
                dof = data["channels"]
                # self.get_logger().info(f"offset: {offset}")
                # self.get_logger().info(f"dof: {dof}")
                limits = [(-180, 180)] * 3  # 默认旋转限制
                
                # 创建关节实例
                joint = Joint(name, offset, dof, limits)
                
                if name in parents:
                    parent_name = parents[name]
                    if parent_name in skeleton:
                        skeleton[parent_name].children.append(joint)
                        joint.parent = skeleton[parent_name]
                
                skeleton[name] = joint
            
            # 设置根节点
            root_name = next(name for name in joints if "Hips" in name)
            skeleton[root_name].parent = None
            return skeleton, motion_data, frame_count, frame_time, parents
            
            
        except Exception as e:
            self.get_logger().error(f"解析BVH文件时出错: {e}")
            return None, None, None, None, None   

    def publish_frame_data(self):
        """发布当前帧数据"""
        if not self.skeleton or not self.motion_data:
            if self.current_bvh_idx >= len(self.bvh_files):
                self.get_logger().info("🔁 所有 BVH 文件播放完毕。")
                self.destroy_node()
                return
            
            bvh_path = self.bvh_files[self.current_bvh_idx]
            motion_name = os.path.basename(bvh_path).replace(".bvh", "")

            self.get_logger().info(f"🎬 正在加载 BVH: {bvh_path} （动作: {motion_name}）")

            if self.record_hdf5 or self.record_pickle:
                self.current_motion_name = motion_name
                self.trajectory = Trajectory(sample_dt=1.0 / self.bvh_dataset_fps)

            try:
                self.skeleton, self.motion_data, frame_count, frame_time, parents = self.load_bvh_file(bvh_path)
                if self.skeleton is None:
                    raise ValueError("BVH 解析失败")
                    
                self.get_logger().info(f"skeleton: {self.skeleton}")

                self.get_logger().info(f"✅ BVH 解析成功，总帧数: {frame_count}")

                if self.enable_retargeting:
                    self.retargeter.set_motion(self.skeleton, self.motion_data)
                    self.get_logger().info("🔁 BVH 数据已设置到 Retargeter.")
                
                self.current_frame = 0

            except Exception as e:
                self.get_logger().error(f"解析 BVH {bvh_path} 时出错: {e}")
                self.current_bvh_idx += 1
                return

        # 如果当前帧超出范围
        if self.current_frame >= len(self.motion_data):

            if self.loop_playback:
                self.get_logger().info("🔂 当前动作播放完毕，切换至下一个 BVH.")
                self.current_bvh_idx += 1
                self.current_frame = 0
                self.skeleton = None
                self.motion_data = None
                self.trajectory = None
                return
            else:
                self.get_logger().info("✅ 全部 BVH 播放完毕.")
                self.destroy_node()
                return
            

        # 如果启用 retargeting，则进行重定向处理
        if self.enable_retargeting and self.retargeter_flag:
            # try:               
                # 直接获取当前帧的重定向结果
                pose_data = next(iter(self.retargeter))  # 获取第一项
                
                # 发布重定向数据
                self.publish_retargted_data()

                # 记录到轨迹（用于HDF5）
                if (self.record_hdf5 or self.record_pickle) and self.trajectory is not None:
                    self.trajectory.add_sample(pose_data)
                    
                    # 文件结束时保存一次，避免频繁IO
                    if self.current_frame == len(self.motion_data) - 1:
                        if self.record_hdf5:
                            self.recorder.add_episode(self.current_motion_name, self.trajectory)
                            self.get_logger().info(f"💾💾 已保存动作 {self.current_motion_name} 帧 {self.current_frame} 到 HDF5.")
                        if self.record_pickle:
                            self.save_pickle_data()
                            

                # 渲染处理（如果启用）
                if self.enable_render:
                    if self.renderer is None:
                        mjcf_path = self.robot_config.mjcf_path
                        if mjcf_path and os.path.exists(mjcf_path):
                            output_video_path = os.path.join(self.output_dir, self.robot_name, "bvh_render.mp4")
                            os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
                            self.renderer = MujocoRenderer(mjcf_path, output_video_path)
                            self.get_logger().info(f"🎮🎮 渲染器已启动")
                    
                    if self.renderer:
                        try:
                            self.retargeter.render_solution(self.renderer)
                            self.renderer.step()
                        except Exception as e:
                            self.get_logger().warn(f"渲染时出错: {e}")
                            
            # except KeyError:
            #     self.get_logger().warn("未找到 Hips 关节，检查 BVH 结构.")
            # except StopIteration:
            #     self.get_logger().warn("重定向器迭代结束")
            # except Exception as e:
            #     self.get_logger().warn(f"重定向处理时出错: {e}")

        else:
            self.get_logger().info(f"⚠️ 未启用 retargeting")

        self.current_frame += 1
    
    def _extract_rotation_about_axis(self, rotation_matrix: np.ndarray, axis: np.ndarray) -> float:
        """
        正确地从旋转矩阵中提取绕特定轴的旋转角度
        
        Args:
            rotation_matrix: 3x3旋转矩阵
            axis: 旋转轴（单位向量）
            
        Returns:
            angle: 旋转角度（弧度）
        """
        axis = np.asarray(axis, dtype=np.float64).flatten()
        axis = axis / np.linalg.norm(axis)
        
        rotation_matrix = np.asarray(rotation_matrix, dtype=np.float64)
        if rotation_matrix.shape != (3, 3):
            rotation_matrix = rotation_matrix.reshape(3, 3)
        
        # 方法1：使用罗德里格斯公式的逆
        # 从旋转矩阵中提取旋转向量
        try:
            # 计算旋转角度（使用矩阵的迹）
            cos_angle = (np.trace(rotation_matrix) - 1) / 2
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            
            # 计算旋转轴
            if abs(angle) < 1e-10:
                # 无旋转
                return 0.0
            
            # 计算旋转轴（非单位向量）
            rx = rotation_matrix[2, 1] - rotation_matrix[1, 2]
            ry = rotation_matrix[0, 2] - rotation_matrix[2, 0]
            rz = rotation_matrix[1, 0] - rotation_matrix[0, 1]
            
            rotation_axis = np.array([rx, ry, rz])
            rotation_axis_norm = np.linalg.norm(rotation_axis)
            
            if rotation_axis_norm < 1e-10:
                # 可能是180度旋转
                # 需要特殊处理
                return angle if np.dot(axis, rotation_axis) >= 0 else -angle
            
            # 单位化旋转轴
            rotation_axis = rotation_axis / rotation_axis_norm
            
            # 计算旋转轴与给定轴的点积
            axis_dot = np.dot(rotation_axis, axis)
            
            # 如果旋转轴与给定轴方向相反，角度为负
            if axis_dot < 0:
                angle = -angle
                axis_dot = -axis_dot
            
            # 投影系数：旋转轴在给定轴上的投影长度
            # 实际绕给定轴的旋转角度 = 总旋转角度 × 投影系数
            projected_angle = angle * axis_dot
            
            return projected_angle
            
        except Exception as e:
            self.get_logger().warn(f"提取旋转角度失败: {e}")
            return 0.0
    
    def save_pickle_data(self):
        motion_data = {}
        root_pos = []
        root_rot = []
        local_body_pos = []
        link_body_list = []
        dof_pos = []
        # 转换轨迹为字典格式
        traj_dict = self.trajectory.to_dict(out_dt=1/self.bvh_dataset_fps)
        
        # 提取关键数据
        joint_positions = traj_dict["joint_positions"]
        T = len(joint_positions)  # 轨迹长度
        
        # 提取根节点
        root_name = self.robot_config.body_links[0]
        transforms = traj_dict["transforms"]

        # 提取根节点位置和方向
        root_positions = np.array([transforms[root_name]["position"][i] for i in range(T)])
        root_orientations = np.array([transforms[root_name]["quaternion"][i] for i in range(T)])
        # 坐标系转换：X前,Y左,Z上 → X后,Y右,Z上
        # 位置转换：X反向，Y反向，Z不变
        root_positions = root_positions * np.array([-1, -1, 1])
        for i in range(T):
            q = root_orientations[i]
            # 四元数格式假设为 [x, y, z, w]
            root_orientations[i] = np.array([-q[0], -q[1], q[2], q[3]])
        motion_data["root_pos"] = root_positions
        motion_data["root_rot"] = root_orientations
        local_body_pos = np.zeros((T, len(self.robot_config.body_links), 3), dtype=np.float32)
        for j, key in enumerate(self.robot_config.body_links):
            if key in transforms:
                positions = np.array([transforms[key]["position"][i] for i in range(T)])
                # 同样应用坐标系转换到每个身体部位的位置
                positions = positions * np.array([-1, -1, 1])
                local_body_pos[:, j, :] = positions

        motion_data["local_body_pos"] = local_body_pos
        
        dof_pos = np.zeros((T, len(self.robot_config.body_links)), dtype=np.float32)
        for i in range(T):
            for j, key in enumerate(self.robot_config.body_links):
                if key not in transforms:
                    continue
                rot = R.from_quat(transforms[key]["quaternion"][i])
                if key == traj_dict["model_root"]:
                    continue
                    
                rot_parent = R.from_quat(transforms[self.robot_config.body_parent_links[j]]["quaternion"][i])
                if key in self.bvh_axis:
                    bvh_axis = np.array(self.bvh_axis[key])
                else:
                    bvh_axis = np.array([0,0,1])
                rotation_child_parent = rot_parent * rot.inv()
                angle = self._extract_rotation_about_axis(rotation_child_parent.as_matrix(), bvh_axis)
                dof_pos[i][j] = angle
        motion_data["dof_pos"] = dof_pos[:,1:]
        motion_data["fps"] = self.bvh_dataset_fps
        motion_data["link_body_list"] = self.robot_config.body_links
        with open(os.path.join(self.pick_out,f"{self.current_motion_name}.pkl"), 'wb') as f:
            pickle.dump(motion_data, f)
 
    
    
    def publish_retargted_data(self):
        """发布重定向后的数据"""

        # 发布 JointState
        self.publish_joint_state()

        # 发布 LinkState
        self.publish_link_state()
    

    def publish_joint_state(self):
        joint_state = JointState()
        joint_state.header.stamp = self.get_clock().now().to_msg()
        joint_state.header.frame_id = f"{self.robot_name}_retargeted"

        for joint_name in self.retargeter.skeleton.keys():
            pos = self.retargeter.get_dataset_position(joint_name)
            self.get_logger().info(f"{joint_name} 当前帧 Pos: {pos}")
            if isinstance(pos, np.ndarray):
                pos_list = pos.tolist()
            else:
                pos_list = [pos.x, pos.y, pos.z] if hasattr(pos, 'x') else [0.0, 0.0, 0.0]

            pos_list = pos_list[:3]  # 只取前3个
            while len(pos_list) < 3:
                pos_list.append(0.0)

            joint_state.name.append(joint_name)
            joint_state.position.extend(pos_list)


        self.retargeted_joint_pub.publish(joint_state)

    def publish_link_state(self):
        link_array = LinkStateArray()
        for joint_name in self.retargeter.skeleton.keys():
            pos = self.retargeter.get_dataset_position(joint_name)
            rot = self.retargeter.get_dataset_rotation(joint_name)

            link = LinkState()
            link.header.stamp = self.get_clock().now().to_msg()
            link.header.frame_id = f"{joint_name}_retargeted"

            # Position
            if isinstance(pos, np.ndarray):
                p = pos.tolist()
            else:
                p = [pos.x, pos.y, pos.z] if hasattr(pos, 'x') else [0.0, 0.0, 0.0]
            p = p[:3]
            while len(p) < 3:
                p.append(0.0)

            link.pose.position.x = float(p[0])
            link.pose.position.y = float(p[1])
            link.pose.position.z = float(p[2])

            # Orientation
            if rot is not None:
                if hasattr(rot, 'as_quat'):  # 是 Rotation 对象
                    q = rot.as_quat()  # x,y,z,w
                elif hasattr(rot, 'flatten') and len(rot) == 3:  # 是旋转矩阵
                    q = R.from_matrix(rot).as_quat()
                else:
                    q = [0.0, 0.0, 0.0, 1.0]  # 默认无旋转
            else:
                q = [0.0, 0.0, 0.0, 1.0]

            link.pose.orientation.x = float(q[0])
            link.pose.orientation.y = float(q[1])
            link.pose.orientation.z = float(q[2])
            link.pose.orientation.w = float(q[3])

            link_array.states.append(link)

        self.retargeted_link_pub.publish(link_array)


def main(args=None):
    rclpy.init(args=args)
    node = BVHParser()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()