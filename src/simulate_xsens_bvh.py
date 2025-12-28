#!/usr/bin/env python3
"""
模拟 Xsens 客户端节点 (Simulate Xsens Client Node)
数据源: BVH 文件
功能: 
1. 解析 BVH 文件
2. 计算正向运动学 (FK) 获取每个骨骼的全局位置和姿态 (模拟 Xsens 硬件解算)
3. 发布 TF 变换 (模拟 xsens_client_node 的行为)
4. 发布 MarkerArray 用于可视化 (适配 bvh_rviz_config.rviz)
"""

import rclpy
from rclpy.node import Node
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import JointState
import numpy as np
from scipy.spatial.transform import Rotation as R
import time
import os

# 默认 BVH 文件路径
DEFAULT_BVH_PATH = "/home/jeff/Codes/Robots/data/Geely test-001.bvh"
START_FRAME = 1700  # 跳过前 1700 帧，直接播放行走动作

class BVHParser:
    def __init__(self, file_path):
        self.file_path = file_path
        self.joints = []
        self.parents = {}
        self.offsets = {}
        self.channels = {}
        self.motion_data = []
        self.frame_time = 0.033
        self._parse()

    def _parse(self):
        print(f"Parsing BVH file: {self.file_path}")
        with open(self.file_path, 'r') as f:
            lines = [line.strip() for line in f.readlines()]

        stack = []
        i = 0
        current_joint = None

        # 解析 Hierarchy
        while i < len(lines):
            line = lines[i]
            if line.startswith('ROOT') or line.startswith('JOINT'):
                name = line.split()[1]
                self.joints.append(name)
                parent = stack[-1] if stack else None
                self.parents[name] = parent
                stack.append(name)
                current_joint = name
            elif line.startswith('End Site'):
                name = current_joint + '_End'
                self.joints.append(name)
                self.parents[name] = stack[-1]
                stack.append(name)
                current_joint = name
            elif line.startswith('OFFSET'):
                parts = line.split()
                self.offsets[current_joint] = np.array([float(x) for x in parts[1:4]])
            elif line.startswith('CHANNELS'):
                parts = line.split()
                self.channels[current_joint] = parts[2:]
            elif line.startswith('}'):
                if stack:
                    stack.pop()
            elif line.startswith('MOTION'):
                i += 1
                break
            i += 1

        # 解析 Motion
        while i < len(lines):
            line = lines[i]
            if line.startswith('Frame Time:'):
                self.frame_time = float(line.split()[2])
            elif line and not line.startswith('Frames:'):
                try:
                    self.motion_data.append([float(x) for x in line.split()])
                except ValueError:
                    pass
            i += 1
        
        self.motion_data = np.array(self.motion_data)
        print(f"Parsed {len(self.joints)} joints and {len(self.motion_data)} frames.")

class XsensSimulator(Node):
    def __init__(self, bvh_path):
        super().__init__('xsens_simulator_node')
        
        # 1. 加载数据
        if not os.path.exists(bvh_path):
            self.get_logger().error(f"File not found: {bvh_path}")
            return
            
        self.bvh = BVHParser(bvh_path)
        
        # 坐标系修正: BVH (Y-up) -> ROS (Z-up)
        # 绕 X 轴旋转 90 度
        self.R_fix = R.from_euler('x', 90, degrees=True)
        
        # 初始位置校准 (用于将人体移动到原点)
        self.initial_root_pos = None
        
        # 2. 初始化发布器
        self.tf_broadcaster = TransformBroadcaster(self)
        self.marker_pub = self.create_publisher(MarkerArray, '/bvh_skeleton', 10)
        self.joint_state_pub = self.create_publisher(JointState, '/joint_states', 10)
        
        # 3. 播放控制
        self.current_frame = START_FRAME
        self.fps = 1.0 / self.bvh.frame_time
        self.timer = self.create_timer(self.bvh.frame_time, self.update_frame)
        
        self.get_logger().info(f"Simulation started. FPS: {self.fps:.2f}")

    def compute_fk(self, frame_idx):
        """
        计算正向运动学 (Forward Kinematics)
        这相当于 Xsens 硬件内部的解算过程，或者是 XSensClient.cpp 接收到的数据来源
        """
        frame_data = self.bvh.motion_data[frame_idx]
        global_transforms = {} # {joint_name: (pos, rot_quat)}
        channel_offset = 0

        for joint in self.bvh.joints:
            if joint.endswith('_End'):
                parent = self.bvh.parents[joint]
                p_pos, p_rot = global_transforms[parent]
                offset = self.bvh.offsets[joint] * 0.01 # cm -> m
                
                # Apply parent rotation to offset
                r = R.from_quat(p_rot)
                pos = p_pos + r.apply(offset)
                global_transforms[joint] = (pos, p_rot) # End site inherits rotation
                continue

            channels = self.bvh.channels.get(joint, [])
            
            # 提取局部变换
            local_pos = np.zeros(3)
            local_rot_euler = []
            rot_order = ""
            
            for ch in channels:
                val = frame_data[channel_offset]
                channel_offset += 1
                
                if 'position' in ch.lower():
                    idx = ['Xposition', 'Yposition', 'Zposition'].index(ch)
                    local_pos[idx] = val
                elif 'rotation' in ch.lower():
                    axis = ch[0].lower()
                    rot_order += axis
                    local_rot_euler.append(val)

            # 处理根节点位置 (cm -> m)
            if self.bvh.parents[joint] is None:
                local_pos *= 0.01
            else:
                local_pos = self.bvh.offsets[joint] * 0.01

            # 计算局部旋转四元数
            if local_rot_euler:
                local_rot = R.from_euler(rot_order, local_rot_euler, degrees=True)
            else:
                local_rot = R.from_quat([0, 0, 0, 1])

            # 计算全局变换
            parent = self.bvh.parents[joint]
            if parent is None:
                # --- 根节点特殊处理 ---
                
                # 1. 原始计算
                global_pos = local_pos
                global_rot = local_rot
                
                # 2. 记录初始位置 (用于归零)
                if self.initial_root_pos is None:
                    self.initial_root_pos = global_pos.copy()
                
                # 3. 减去初始位置 (归零)
                global_pos -= self.initial_root_pos
                
                # 4. 应用坐标系修正 (Y-up -> Z-up)
                # 位置旋转
                global_pos = self.R_fix.apply(global_pos)
                # 姿态旋转
                global_rot = self.R_fix * global_rot
                
            else:
                p_pos, p_rot = global_transforms[parent]
                p_rot_obj = R.from_quat(p_rot)
                
                global_pos = p_pos + p_rot_obj.apply(local_pos)
                global_rot = p_rot_obj * local_rot

            global_transforms[joint] = (global_pos, global_rot.as_quat())

        return global_transforms

    def update_frame(self):
        if self.current_frame >= len(self.bvh.motion_data):
            self.current_frame = 0 # Loop
            
        # 1. 计算当前帧的所有骨骼姿态
        transforms = self.compute_fk(self.current_frame)
        
        timestamp = self.get_clock().now().to_msg()
        
        # 2. 发布 TF (模拟 xsens_client_node)
        for joint, (pos, quat) in transforms.items():
            if joint.endswith('_End'): continue
            
            t = TransformStamped()
            t.header.stamp = timestamp
            t.header.frame_id = "world"
            t.child_frame_id = joint # 使用 BVH 关节名作为 Frame ID
            
            t.transform.translation.x = pos[0]
            t.transform.translation.y = pos[1]
            t.transform.translation.z = pos[2]
            t.transform.rotation.x = quat[0]
            t.transform.rotation.y = quat[1]
            t.transform.rotation.z = quat[2]
            t.transform.rotation.w = quat[3]
            
            self.tf_broadcaster.sendTransform(t)

        # 3. 发布 MarkerArray (用于 RViz 可视化)
        self.publish_markers(transforms, timestamp)
        
        self.current_frame += 1

    def publish_markers(self, transforms, timestamp):
        marker_array = MarkerArray()
        id_counter = 0
        
        # 绘制关节 (Sphere)
        for joint, (pos, _) in transforms.items():
            marker = Marker()
            marker.header.frame_id = "world"
            marker.header.stamp = timestamp
            marker.ns = "joints"
            marker.id = id_counter
            id_counter += 1
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.scale.x = 0.03; marker.scale.y = 0.03; marker.scale.z = 0.03
            marker.color.r = 1.0; marker.color.a = 1.0
            marker.pose.position.x = pos[0]
            marker.pose.position.y = pos[1]
            marker.pose.position.z = pos[2]
            marker_array.markers.append(marker)
            
            # 绘制骨骼 (Line)
            parent = self.bvh.parents.get(joint)
            if parent:
                p_pos, _ = transforms[parent]
                line = Marker()
                line.header.frame_id = "world"
                line.header.stamp = timestamp
                line.ns = "bones"
                line.id = id_counter
                id_counter += 1
                line.type = Marker.LINE_STRIP
                line.action = Marker.ADD
                line.scale.x = 0.01
                line.color.g = 1.0; line.color.a = 1.0
                
                from geometry_msgs.msg import Point
                p1 = Point(x=p_pos[0], y=p_pos[1], z=p_pos[2])
                p2 = Point(x=pos[0], y=pos[1], z=pos[2])
                line.points = [p1, p2]
                marker_array.markers.append(line)

        self.marker_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    
    # 使用指定的 BVH 文件
    node = XsensSimulator(DEFAULT_BVH_PATH)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
