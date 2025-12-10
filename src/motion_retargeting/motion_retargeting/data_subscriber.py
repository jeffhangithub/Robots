#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from tf2_ros import TransformListener, Buffer
from geometry_msgs.msg import Point, TransformStamped, Pose, Quaternion
from sensor_msgs.msg import JointState
from xsens_mvn_ros_msgs.msg import LinkStateArray, LinkState
import tf2_geometry_msgs
import numpy as np
import os
import json
from typing import Dict, List, Tuple, Optional
from scipy.spatial.transform import Rotation as R
import time
import pickle

# 导入在线处理模块
from motion_retargeting.config.robot.g1 import G1_BVH_CONFIG
from motion_retargeting.config.robot.h1 import H1_BVH_CONFIG
from motion_retargeting.config.robot.loong import LOONG_BVH_CONFIG
from motion_retargeting.config.robot.hi1 import HI1_BVH_CONFIG
from motion_retargeting.utils.mujoco.renderer import MujocoRenderer
from motion_retargeting.utils.trajectory import Trajectory

# 导入重定向模块
from motion_retargeting.retarget.retarget import Joint
from motion_retargeting.retarget.retarget_online import BVHRetargetOnline

robots = {
    "g1": G1_BVH_CONFIG,
    "h1": H1_BVH_CONFIG,
    "loong": LOONG_BVH_CONFIG,
    "hi_1": HI1_BVH_CONFIG
}

class XSensSubscriber(Node):
    def __init__(self):
        super().__init__('motion_retargeting_node')
        
        # Parameters
        self.declare_parameter('model_name', 'skeleton')
        self.declare_parameter('reference_frame', 'world')
        self.declare_parameter('target_frame', 'base_link')
        self.declare_parameter('bvh_dataset_fps', 120)
        self.declare_parameter('robot_name', 'g1')
        self.declare_parameter('publish_rate', 30.0)
        self.declare_parameter('enable_render', True)
        self.declare_parameter('output_dir', '/home/uneedrobot/workshops/source/xsens/ros2/results')
        
        self.model_name = self.get_parameter('model_name').value
        self.reference_frame = self.get_parameter('reference_frame').value
        self.target_frame = self.get_parameter('target_frame').value
        self.bvh_dataset_fps = self.get_parameter('bvh_dataset_fps').value
        self.robot_name = self.get_parameter('robot_name').value
        self.publish_rate = self.get_parameter('publish_rate').value
        self.enable_render = self.get_parameter('enable_render').value
        self.output_dir = self.get_parameter('output_dir').value

        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)

        # 关节处理顺序
        self.joint_order = [
            'Hips', 'Chest', 'Chest2', 'Chest3', 'Chest4', 'Neck', 'Head',
            'RightCollar', 'RightShoulder', 'RightElbow', 'RightWrist',
            'LeftCollar', 'LeftShoulder', 'LeftElbow', 'LeftWrist',
            'RightHip', 'RightKnee', 'RightAnkle', 'RightToe',
            'LeftHip', 'LeftKnee', 'LeftAnkle', 'LeftToe'
        ]

        # 获取机器人配置
        robot_config_class = robots.get(self.robot_name, {})
        if robot_config_class is None:
            self.get_logger().error(f"未找到机器人配置: {self.robot_name}")
            return
        else:
            self.robot_config = robot_config_class()

        self.bvh_axis = {}
        if hasattr(self.robot_config, 'bvh_axis'):
            self.bvh_axis = self.robot_config.bvh_axis
        
        self.get_logger().info(f"使用机器人配置: {self.robot_name}")

        # Subscribers
        self.joint_state_sub = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_state_callback,
            10
        )
        
        self.link_state_sub = self.create_subscription(
            LinkStateArray,
            'link_states',
            self.link_state_callback,
            10
        )
        
        self.com_sub = self.create_subscription(
            Point,
            'com',
            self.com_callback,
            10
        )

        # 数据存储
        self.joint_positions: Dict[str, List[float]] = {}
        self.link_states: Dict[str, Dict] = {}  # 存储完整的link信息
        self.center_of_mass: Point = Point()
        self.last_timestamp = None
        self.skeleton: Optional[Dict[str, Joint]] = None
        self.current_frame_data = None  # 改为存储当前帧数据
        self.current_frame = 0

        # 初始化重定向器 - 在线处理版本
        self.retargeter = BVHRetargetOnline(
            bvh_dataset_fps=self.bvh_dataset_fps,
            wbik_params=self.robot_config
        )
        self.get_logger().info(f"BVHRetargetOnline initialized for robot: {self.robot_name}")

        # 初始化渲染器
        self.renderer = None
        if self.enable_render:
            try:
                mjcf_path = self.robot_config.mjcf_path
                if mjcf_path and os.path.exists(mjcf_path):
                    output_video_path = os.path.join(self.output_dir, f"{self.robot_name}_online_render.mp4")
                    self.renderer = MujocoRenderer(mjcf_path, output_video_path)
                    self.get_logger().info(f"🎮 渲染器已启动")
                else:
                    self.get_logger().warning(f"MJCF文件不存在: {mjcf_path}")
            except ImportError:
                self.get_logger().warning("MuJoCo渲染器不可用")
            except Exception as e:
                self.get_logger().warning(f"渲染器初始化失败: {e}")

        # 发布器
        qos_profile = rclpy.qos.QoSProfile(
            depth=10,
            reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
            durability=rclpy.qos.DurabilityPolicy.VOLATILE
        )
        
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
 
        # 添加XSens到BVH映射键的字典
        self.xsens_key_to_bvh_key = {
            'pelvis_pelvis_NA': 'Hips',           # 骨盆 -> 骨盆
            'l5_s1': 'Chest',                       # 腰椎5/骶骨1 -> 脊柱（胸部）
            'l4_l3': 'Chest2',                      # 腰椎4/3 -> 脊柱1（胸部2）
            'l1_t12': 'Chest3',                     # 腰椎1/胸椎12 -> 脊柱2（胸部3）
            't9_t8': 'Chest4',                      # 胸椎9/8 -> 脊柱3（胸部4）
            't1_c7': 'Neck',                        # 胸椎1/颈椎7 -> 颈部
            'c1_head': 'Head',                      # 颈椎1/头部 -> 头部
            'right_c7_shoulder': 'RightCollar',  # 右C7肩部 -> 右肩（右锁骨）
            'right_shoulder': 'RightShoulder',    # 右肩 -> 右上臂（右肩部）
            'right_elbow': 'RightElbow',         # 右肘 -> 右前臂（右肘部）
            'right_wrist': 'RightWrist',            # 右腕 -> 右手（右腕部）
            'left_c7_shoulder': 'LeftCollar',    # 左C7肩部 -> 左肩（左锁骨）
            'left_shoulder': 'LeftShoulder',      # 左肩 -> 左上臂（左肩部）
            'left_elbow': 'LeftElbow',           # 左肘 -> 左前臂（左肘部）
            'left_wrist': 'LeftWrist',              # 左腕 -> 左手（左腕部）
            'right_hip': 'RightHip',               # 右髋 -> 右髋
            'right_knee': 'RightKnee',            # 右膝 -> 右大腿（右膝部）
            'right_ankle': 'RightAnkle',            # 右踝 -> 右小腿（右踝部）
            'right_ballfoot': 'RightToe',         # 右球脚 -> 右脚（右趾部）
            'left_hip': 'LeftHip',                 # 左髋 -> 左髋
            'left_knee': 'LeftKnee',              # 左膝 -> 左大腿（左膝部）
            'left_ankle': 'LeftAnkle',              # 左踝 -> 左小腿（左踝部）
            'left_ballfoot': 'LeftToe',           # 左球脚 -> 左脚（左趾部）
        }
        
        # 添加link名称到BVH关节的映射
        self.link_to_bvh_map = {
            'pelvis': 'Hips',
            'l5': 'Chest',
            'l3': 'Chest2',
            't12': 'Chest3',
            't8': 'Chest4',
            'neck': 'Neck',
            'head': 'Head',
            
            #右臂链
            'right_shoulder': 'RightCollar',
            'right_upper_arm': 'RightShoulder',
            'right_forearm': 'RightElbow',
            'right_hand': 'RightWrist',

            #左臂链
            'left_shoulder': 'LeftCollar',
            'left_upper_arm': 'LeftShoulder',
            'left_forearm': 'LeftElbow',
            'left_hand': 'LeftWrist',

            #右腿链
            'right_upper_leg': 'RightHip',
            'right_lower_leg': 'RightKnee',
            'right_foot': 'RightAnkle',
            'right_toe': 'RightToe',

            #左腿链
            'left_upper_leg': 'LeftHip',
            'left_lower_leg': 'LeftKnee',
            'left_foot': 'LeftAnkle',
            'left_toe': 'LeftToe',
        }
        
        self.parent_map = {
            'Hips': None,
            'Chest': 'Hips',
            'Chest2': 'Chest',
            'Chest3': 'Chest2', 
            'Chest4': 'Chest3',
            'Neck': 'Chest4',
            'Head': 'Neck',
            'RightCollar': 'Chest4',
            'RightShoulder': 'RightCollar',
            'RightElbow': 'RightShoulder', 
            'RightWrist': 'RightElbow',
            'LeftCollar': 'Chest4',
            'LeftShoulder': 'LeftCollar',
            'LeftElbow': 'LeftShoulder',
            'LeftWrist': 'LeftElbow',
            'RightHip': 'Hips',
            'RightKnee': 'RightHip',
            'RightAnkle': 'RightKnee',
            'RightToe': 'RightAnkle',
            'LeftHip': 'Hips',
            'LeftKnee': 'LeftHip',
            'LeftAnkle': 'LeftKnee',
            'LeftToe': 'LeftAnkle'
        }

        # # 初始化关节偏移量字典（不准确）
        self.joint_offsets = {
            "Hips": (0.0, 0.0, 0.0),
            "Chest": (0.0, 6.906505, -5.157238),
            "Chest2": (0.0, 7.062918, 0.000204),
            "Chest3": (0.0, 10.097865, -0.000105),
            "Chest4": (0.0, 10.413862, -0.000116),
            "Neck": (0.0, 17.108243, 0.000098),
            "Head": (0.0, 10.150588, 0.000000),
            "RightCollar": (-2.881770, 10.693114, 0.000049),
            "RightShoulder": (-16.064314, 0.000000, 0.000000),
            "RightElbow": (-24.028326, 0.000000, 0.000000),
            "RightWrist": (-24.135334, 0.000000, 0.000000),
            "LeftCollar": (2.881770, 10.693114, 0.000049),
            "LeftShoulder": (16.064314, 0.000000, 0.000000),
            "LeftElbow": (24.028326, 0.000000, 0.000000),
            "LeftWrist": (24.135334, 0.000000, 0.000000),
            "RightHip": (-7.459725, 0.001394, -0.000221),
            "RightKnee": (0.0, -39.829222, -0.000031),
            "RightAnkle": (0.0, -39.950013, -0.000052),
            "RightToe": (0.0, -6.363212, 16.389543),
            "LeftHip": (7.459725, -0.000747, 0.000118),
            "LeftKnee": (0.0, -39.829222, -0.000031),
            "LeftAnkle": (0.0, -39.950013, -0.000052),
            "LeftToe": (0.0, -6.363212, 16.389543),
        }

        # 初始化骨架
        self.init_skeleton()

        # 设置骨架到重定向器（在线处理只需设置一次）
        self.retargeter.set_skeleton(self.skeleton)

        # 处理定时器
        self.process_timer = self.create_timer(1.0/self.publish_rate, self.process_data)
        
        # 数据保存设置
        self.data_file = os.path.join(self.output_dir, "online_retargeted_data.jsonl")
        self.save_counter = 0
        self.save_every_n_frames = 1
        
        self.get_logger().info(f"在线运动重定向节点已初始化，机器人: {self.robot_name}")

    def init_skeleton(self):
        """初始化BVH骨架结构"""
        self.skeleton = {}
        
        # 创建关节实例
        for joint_name in self.joint_order:
            # 设置默认偏移量（后续会根据实际数据更新）
            offset = np.zeros(3)
            dof = []
            limits = [(-180, 180)] * 3
            
            # 根据关节类型设置自由度
            if joint_name == 'pelvis' or joint_name == 'Hips':
                dof = ['Xposition', 'Yposition', 'Zposition', 'Yrotation', 'Xrotation', 'Zrotation']
                bvh_joint_name = 'Hips'
            else:
                dof = ['Yrotation', 'Xrotation', 'Zrotation']
                bvh_joint_name = joint_name
            offset = np.array(self.joint_offsets[bvh_joint_name]) * 0.01
                        
            joint = Joint(bvh_joint_name, offset, dof, limits)
            self.skeleton[bvh_joint_name] = joint
        
        # 设置父子关系
        for joint_name, parent_name in self.parent_map.items():
            # 映射到BVHRetarget期望的关节名称
            bvh_joint_name = 'Hips' if joint_name == 'pelvis' else joint_name
            bvh_parent_name = 'Hips' if parent_name == 'pelvis' else parent_name if parent_name else None
        
            if bvh_parent_name and bvh_parent_name in self.skeleton and bvh_joint_name in self.skeleton:
                joint = self.skeleton[bvh_joint_name]
                parent_joint = self.skeleton[bvh_parent_name]
                joint.parent = parent_joint
                parent_joint.children.append(joint)

    def convert_xsens_to_bvh_frame(self) -> Optional[Dict]:
        """
        将当前XSens数据转换为BVH格式的一帧数据
        
        Returns:
            BVH格式的帧数据字典
        """
        if not self.link_states or len(self.link_states) < len(self.joint_order) // 2:
            return None
        
        try:
            frame_data = {}
            
            # 计算全局位置和旋转
            global_positions = {}
            global_rotations = {}
        
            for joint_name in self.joint_order:
                bvh_joint_name = 'Hips' if joint_name == 'pelvis' else joint_name

                if joint_name in self.link_states:
                    link_data = self.link_states[joint_name]
                    pos = link_data['position']
                    quat = link_data['orientation']
                    
                    # 坐标系转换: XSens (Z-up) -> BVH (Y-up)
                    # 位置转换: X->Z, Z->Y, Y->X
                    bvh_pos = np.array([pos[1], pos[2], pos[0]])
                    
                    # 旋转转换
                    bvh_rot = R.from_quat(quat)
                    
                    global_positions[bvh_joint_name] = bvh_pos
                    global_rotations[bvh_joint_name] = bvh_rot
                else:
                    # 使用默认值
                    global_positions[bvh_joint_name] = np.zeros(3)
                    global_rotations[bvh_joint_name] = np.eye(3)
            
            # 计算局部旋转（欧拉角）
            for joint_name in self.joint_order:
                bvh_joint_name = 'Hips' if joint_name == 'pelvis' else joint_name
                parent_name = self.parent_map.get(joint_name)
                bvh_parent_name = 'Hips' if parent_name == 'pelvis' else parent_name if parent_name else None

                
                if bvh_parent_name is None:  # 根节点
                    # 根节点包含位置和旋转
                    global_rot = global_rotations[bvh_joint_name]
                    
                    # # 根节点数据: [位置X, 位置Y, 位置Z, 旋转Z, 旋转X, 旋转Y]
                    euler_angles = global_rot.as_euler('ZYX', degrees=True)
                    
                    # 根节点数据: [位置X, 位置Y, 位置Z, 旋转Z, 旋转X, 旋转Y]
                    frame_data[bvh_joint_name] = [
                        float(global_positions[bvh_joint_name][0]),  # X
                        float(global_positions[bvh_joint_name][1]),  # Y  
                        float(global_positions[bvh_joint_name][2]),  # Z
                        np.deg2rad(float(euler_angles[0])),  # Z rotation
                        np.deg2rad(float(euler_angles[1])),  # Y rotation
                        np.deg2rad(float(euler_angles[2]))   # X rotation
                    ]
                    
                else:
                    # 计算局部旋转
                    parent_rot = global_rotations[bvh_parent_name]
                    joint_rot = global_rotations[bvh_joint_name]
                    local_rot = parent_rot.inv() * joint_rot  # 相对旋转
                    
                    euler_angles = local_rot.as_euler('ZYX', degrees=True)
                    
                    # 非根节点只有旋转
                    if bvh_joint_name == 'RightShoulder':
                        frame_data[bvh_joint_name] = [
                            np.deg2rad(float(euler_angles[0])),  # Z rotation
                            np.deg2rad(float(euler_angles[1])),  # Y rotation
                            np.deg2rad(float(euler_angles[2]) + 90)   # X rotation
                        ]
                    elif bvh_joint_name == 'LeftShoulder':
                        frame_data[bvh_joint_name] = [
                            np.deg2rad(float(euler_angles[0])),  # Z rotation
                            np.deg2rad(float(euler_angles[1])),  # Y rotation
                            np.deg2rad(float(euler_angles[2]) - 90)    # X rotation
                        ]
                    elif bvh_joint_name in ["RightElbow", "RightWrist"]:
                        frame_data[bvh_joint_name] = [
                            np.deg2rad(-float(euler_angles[1])),  # Z rotation
                            np.deg2rad(float(euler_angles[0])),  # Y rotation
                            np.deg2rad(float(euler_angles[2]))   # X rotation
                        ]
                    elif bvh_joint_name in ["LeftElbow", "LeftWrist"]:
                        frame_data[bvh_joint_name] = [
                            np.deg2rad(float(euler_angles[1])),  # Z rotation
                            np.deg2rad(-float(euler_angles[0])),  # Y rotation
                            np.deg2rad(float(euler_angles[2]))   # X rotation
                        ]
                    else:
                        frame_data[bvh_joint_name] = [
                            np.deg2rad(float(euler_angles[0])),  # Z rotation
                            np.deg2rad(float(euler_angles[1])),  # Y rotation
                            np.deg2rad(float(euler_angles[2]))   # X rotation
                        ]
                    
            
            return frame_data
            
        except Exception as e:
            self.get_logger().error(f"转换XSens数据到BVH格式时出错: {e}")
            return None

    def joint_state_callback(self, msg: JointState):
        """关节状态回调"""
        try:
            # 清空之前的关节数据
            self.joint_positions.clear()

            # 解析关节名称和位置
            for i, name in enumerate(msg.name):
                # 清理关节名称
                clean_name = name.replace(f"{self.model_name}_", "").replace("_x", "").replace("_y", "").replace("_z", "")
                
                # 使用映射字典得到BVH映射键
                if clean_name in self.xsens_key_to_bvh_key:
                    bvh_key = self.xsens_key_to_bvh_key[clean_name]
                    if bvh_key not in self.joint_positions:
                        self.joint_positions[bvh_key] = [0.0, 0.0, 0.0]
                    
                    # 存储关节角度
                    if name.endswith('_x'):
                        self.joint_positions[bvh_key][0] = round(float(msg.position[i]),6)
                    elif name.endswith('_y'):
                        self.joint_positions[bvh_key][1] = round(float(msg.position[i]),6)
                    elif name.endswith('_z'):
                        self.joint_positions[bvh_key][2] = round(float(msg.position[i]),6)
            
            self.get_logger().debug(f"Received joint states for {len(self.joint_positions)} joints")
            
        except Exception as e:
            self.get_logger().error(f"Error processing joint states: {e}")
    
    def link_state_callback(self, msg: LinkStateArray):
        """Callback for link state messages"""
        try:
            for link_state in msg.states:
                frame_id = link_state.header.frame_id
                
                # 清理link名称
                clean_name = frame_id.replace(f"{self.model_name}_", "")
                
                # 映射到BVH关节名称
                bvh_name = self.link_to_bvh_map.get(clean_name, clean_name)
                
                # 提取完整的link信息
                self.link_states[bvh_name] = {
                    'position': np.array([
                        link_state.pose.position.x, 
                        link_state.pose.position.y, 
                        link_state.pose.position.z
                    ]),
                    'orientation': np.array([
                        link_state.pose.orientation.x,
                        link_state.pose.orientation.y,
                        link_state.pose.orientation.z,
                        link_state.pose.orientation.w
                    ]),
                    'timestamp': link_state.header.stamp
                }
            
            self.last_timestamp = msg.states[0].header.stamp if msg.states else None
            
        except Exception as e:
            self.get_logger().error(f"Error processing link states: {e}")
    
    def com_callback(self, msg: Point):
        """Callback for center of mass messages"""
        try:
            self.center_of_mass = msg
            self.get_logger().debug(f"Received COM: x={msg.x:.3f}, y={msg.y:.3f}, z={msg.z:.3f}")
            
        except Exception as e:
            self.get_logger().error(f"Error processing COM: {e}")
    
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

    def process_data(self):
        """Main processing function called by timer"""
        try:
            if not self.link_states:
                self.get_logger().debug("等待链接状态数据...")
                return

            # 转换XSens数据到BVH格式
            frame_data = self.convert_xsens_to_bvh_frame()
            
            if frame_data is None:
                self.get_logger().debug("无法转换XSens数据到BVH格式")
                return

            # 检查Hips关节是否存在
            if 'Hips' not in frame_data:
                self.get_logger().warning("BVH数据中缺少Hips关节")
                return
            
            # 使用在线处理方式处理当前帧
            result = self.retargeter.process_frame(frame_data)
            
            if result is None:
                self.get_logger().warning("处理帧数据失败")
                return
                
            retargeted_positions, retargeted_rotations = result
                        
            # 发布重定向数据
            # self.publish_retargeted_data(retargeted_positions, retargeted_rotations)
            
            # 保存数据
            # self.save_retargeted_data(retargeted_positions, retargeted_rotations)     

            # 创建迭代器并获取当前帧数据
            pose_data = next(iter(self.retargeter))

            cmd_motion_data = self.process_robot_cmd(pose_data)

            # 发布cmd_motion_data
            self.publish_cmd_motion_data(cmd_motion_data)

            # self.get_logger().info(f"cmd_motion_data is: {cmd_motion_data}")
            
            # 渲染
            if self.enable_render and self.renderer:
                try:
                    self.retargeter.render_solution(self.renderer)
                    self.renderer.step()
                except Exception as e:
                    self.get_logger().warning(f"渲染时出错: {e}")
            
            self.current_frame += 1
            
            # 每100帧打印一次状态
            if self.current_frame % 100 == 0:
                self.get_logger().info(f"已处理 {self.current_frame} 帧数据")
                    
                
        except Exception as e:
            self.get_logger().error(f"数据处理过程中出错: {e}")

    def process_robot_cmd(self, pose_data):
        cmd_motion_data = {}
        cmd_motion_data["fps"] = self.bvh_dataset_fps
        cmd_motion_data["link_body_list"] = self.robot_config.body_links
        root_name = self.robot_config.body_links[0]

        transforms = pose_data.transforms
        # self.get_logger().info(f"transforms {transforms}")
        transforms_dict = {item.name: {'position': item.position, 'quaternion': item.quaternion} for item in transforms}

        local_body_pos = np.zeros((len(self.robot_config.body_links), 3), dtype=np.float32)
        dof_pos = np.zeros((len(self.robot_config.body_links)), dtype=np.float32)

        # 定义坐标系转换矩阵
        # coordinate_transform = np.array([
        #     [0, 0, 1],  # X -> Z
        #     [-1, 0, 0], # Y -> -X  
        #     [0, 1, 0]   # Z -> Y
        # ])
        coordinate_transform = np.array([
            [0, -1, 0],  # X -> Z
            [1, 0, 0], # Y -> X 
            [0, 0, 1]   # Z -> Y
        ])

        # 用于四元数转换的旋转矩阵
        quat_transform = R.from_matrix(coordinate_transform)
        
        # 提取根节点位置和方向（应用坐标系转换）
        if root_name in transforms_dict:
            # 位置转换
            original_root_pos = transforms_dict[root_name]["position"]
            transformed_root_pos = coordinate_transform @ original_root_pos
            cmd_motion_data["root_pos"] = transformed_root_pos
            
            # 四元数转换
            original_root_quat = transforms_dict[root_name]["quaternion"]
            # 注意：可能需要调整四元数顺序
            # original_rot = R.from_quat([original_root_quat.x, original_root_quat.y, original_root_quat.z, original_root_quat.w])
            original_rot = R.from_quat([original_root_quat.w, original_root_quat.x, original_root_quat.y, original_root_quat.z])
            
            # 应用坐标系旋转
            transformed_rot = quat_transform * original_rot
            transformed_quat = transformed_rot.as_quat()  # [x, y, z, w]
            
            # cmd_motion_data["root_rot"] = np.array([transformed_quat[3], transformed_quat[0], transformed_quat[1], transformed_quat[2]])
            cmd_motion_data["root_rot"] = transformed_quat

        for j, key in enumerate(self.robot_config.body_links):
            if key in transforms_dict:
                # 位置转换
                original_pos = transforms_dict[key]["position"]
                transformed_pos = coordinate_transform @ original_pos
                local_body_pos[j, :] = transformed_pos
                
                # 四元数转换
                quat = transforms_dict[key]["quaternion"]
                # 注意四元数顺序：通常w在前
                # original_rot = R.from_quat([quat.w, quat.x, quat.y, quat.z])
                original_rot = R.from_quat([quat.x, quat.y, quat.z, quat.w])
                transformed_rot = quat_transform * original_rot
                transformed_quat = transformed_rot.as_quat()
                
                if key == pose_data.model_root:
                    continue
                
                # 获取父节点的旋转（同样需要转换）
                parent_key = self.robot_config.body_parent_links[j]
                if parent_key in transforms_dict:
                    parent_quat = transforms_dict[parent_key]["quaternion"]
                    # parent_rot = R.from_quat([parent_quat.w, parent_quat.x, parent_quat.y, parent_quat.z])
                    parent_rot = R.from_quat([parent_quat.x, parent_quat.y, parent_quat.z, parent_quat.w])
                    transformed_parent_rot = quat_transform * parent_rot
                    
                    # 计算相对旋转
                    rotation_child_parent = transformed_parent_rot * transformed_rot.inv()
                    
                    if key in self.bvh_axis:
                        bvh_axis = np.array(self.bvh_axis[key])
                        # 注意：bvh_axis可能也需要转换
                        transformed_bvh_axis = bvh_axis
                    else:
                        transformed_bvh_axis = np.array([0, 0, 1])  # 默认Z轴
                    
                    dof_pos[j] = self._extract_rotation_about_axis(
                        rotation_child_parent.as_matrix(), 
                        transformed_bvh_axis
                    )
        cmd_motion_data["dof_pos"] = dof_pos
        cmd_motion_data["local_body_pos"] = local_body_pos

        # self.get_logger().info(f"cmd_motion_data {cmd_motion_data}")

        return cmd_motion_data

    def publish_cmd_motion_data(self, cmd_data: Dict):
        """发布运动数据"""
        try:
            # 使用JointState发布关键数据
            joint_msg = JointState()
            joint_msg.header.stamp = self.get_clock().now().to_msg()
            joint_msg.header.frame_id = "motion_data"
            
            # 发布fps作为第一个"关节"的值
            fps = float(cmd_data.get("fps", self.bvh_dataset_fps))
            joint_msg.name.append("fps")
            joint_msg.position.append(fps)

            # 发布根位置和旋转
            if "root_pos" in cmd_data:
                root_pos = cmd_data["root_pos"]
                for i, pos in enumerate(root_pos[:3]):
                    joint_msg.name.append(f"root_pos_{i}")
                    joint_msg.position.append(float(pos))
            
            if "root_rot" in cmd_data:
                root_rot = cmd_data["root_rot"]
                for i, rot in enumerate(root_rot[:4]):
                    joint_msg.name.append(f"root_rot_{i}")
                    joint_msg.position.append(float(rot))

            # 发布自由度位置
            if "dof_pos" in cmd_data and "link_body_list" in cmd_data and "local_body_pos" in cmd_data:
                dof_pos = cmd_data["dof_pos"]
                link_body_list = cmd_data["link_body_list"]
                local_body_pos = cmd_data["local_body_pos"]

                for link_body, pos, local_pos in zip(link_body_list, dof_pos, local_body_pos):
                    joint_msg.name.append(f"{link_body}_dof_pos")
                    joint_msg.position.append(float(pos))
                    joint_msg.name.append(f"{link_body}_local_body_pos_x")
                    joint_msg.position.append(float(local_pos[0]))
                    joint_msg.name.append(f"{link_body}_local_body_pos_y")
                    joint_msg.position.append(float(local_pos[1]))
                    joint_msg.name.append(f"{link_body}_local_body_pos_z")
                    joint_msg.position.append(float(local_pos[2]))
                    
            # self.get_logger().info(f"joint_msg {joint_msg}")
            self.retargeted_joint_pub.publish(joint_msg)
                    
            self.get_logger().debug(f"发布运动数据: {len(joint_msg.name)}个数据点")
            
        except Exception as e:
            self.get_logger().error(f"发布运动数据时出错: {e}")

    # def publish_retargeted_data(self, positions: Dict[str, np.ndarray], rotations: Dict[str, np.ndarray]):
    #     """发布重定向后的数据"""
    #     try:
    #         # 发布关节状态
    #         joint_msg = JointState()
    #         joint_msg.header.stamp = self.get_clock().now().to_msg()
    #         joint_msg.header.frame_id = f"{self.target_frame}_retargeted"
            
    #         for joint_name in self.joint_order:
    #             if joint_name in positions:
    #                 joint_msg.name.append(joint_name)
    #                 pos = positions[joint_name]
    #                 pos_list = pos.tolist() if hasattr(pos, 'tolist') else list(pos)
    #                 while len(pos_list) < 3:
    #                     pos_list.append(0.0)

    #                 # 格式化位置数据，限制小数点后6位
    #                 formatted_pos = [round(float(x), 6) for x in pos_list[:3]]
    #                 joint_msg.position.extend(formatted_pos)
            
    #         self.retargeted_joint_pub.publish(joint_msg)
            
    #         # 发布链接状态
    #         link_msg = LinkStateArray()
            
    #         for joint_name, position in positions.items():
    #             link_state = LinkState()
    #             link_state.header.stamp = self.get_clock().now().to_msg()
    #             link_state.header.frame_id = f"{joint_name}_retargeted"
                
    #             pos_list = position.tolist() if hasattr(position, 'tolist') else list(position)
    #             while len(pos_list) < 3:
    #                 pos_list.append(0.0)
                    

    #             # 格式化位置数据，限制小数点后6位
    #             formatted_pos = [round(float(x), 6) for x in pos_list[:3]]

    #             link_state.pose.position.x = formatted_pos[0]
    #             link_state.pose.position.y = formatted_pos[1]
    #             link_state.pose.position.z = formatted_pos[2]
                
    #             if joint_name in rotations:
    #                 rotation = rotations[joint_name]
    #                 if hasattr(rotation, 'as_quat'):
    #                     quat = rotation.as_quat()
    #                 else:
    #                     quat = R.from_matrix(rotation).as_quat()
                    
    #                 # 格式化四元数，限制小数点后6位
    #                 formatted_quat = [round(float(q), 6) for q in quat]

    #                 link_state.pose.orientation.x = formatted_quat[0]
    #                 link_state.pose.orientation.y = formatted_quat[1]
    #                 link_state.pose.orientation.z = formatted_quat[2]
    #                 link_state.pose.orientation.w = formatted_quat[3]
    #             else:
    #                 link_state.pose.orientation.w = 1.0
                
    #             link_msg.states.append(link_state)
            
    #         self.retargeted_link_pub.publish(link_msg)
            
    #     except Exception as e:
    #         self.get_logger().error(f"发布重定向数据时出错: {e}")

    def save_retargeted_data(self, positions: Dict[str, np.ndarray], rotations: Dict[str, np.ndarray]):
        """保存重定向数据"""
        try:
            self.save_counter += 1
            if self.save_counter < self.save_every_n_frames:
                return
                
            self.save_counter = 0
            
            data = {
                'timestamp': time.time(),
                'frame': self.current_frame,
                'positions': {},
                'rotations': {}
            }
            
            for joint, pos in positions.items():
                if hasattr(pos, 'tolist'):
                    pos_list = pos.tolist()
                else:
                    pos_list = list(pos)
                
                # 格式化每个数值，限制小数点后6位
                formatted_pos = [round(float(x), 6) for x in pos_list]
                data['positions'][joint] = formatted_pos
            
            # 转换旋转数据
            for joint, rot in rotations.items():
                if hasattr(rot, 'as_quat'):
                    quat = rot.as_quat().tolist()
                else:
                    quat = R.from_matrix(rot).as_quat().tolist()

                # 格式化四元数，限制小数点后6位
                formatted_quat = [round(float(q), 6) for q in quat]
                data['rotations'][joint] = formatted_quat
            
            # 追加到文件
            with open(self.data_file, 'a') as f:
                f.write(json.dumps(data) + '\n')
                
        except Exception as e:
            self.get_logger().error(f"Error saving retargeted data: {e}")

def main(args=None):
    rclpy.init(args=args)
    subscriber = None
    try:
        subscriber = XSensSubscriber()
        rclpy.spin(subscriber)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error in motion retargeting node: {e}")
    finally:
        if subscriber is not None:
            subscriber.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()