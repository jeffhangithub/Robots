#!/usr/bin/env python3
"""
BVH to RViz 可视化节点
将 BVH 文件中的骨骼动作数据发布为 TF 变换，在 RViz 中实时播放
"""

import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.node import Node
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from visualization_msgs.msg import Marker, MarkerArray


def parse_bvh(file_path):
    """解析 BVH 文件，提取骨骼结构和运动数据"""
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    
    # 解析层次结构
    joints = []  # 关节名称列表（按BVH中出现顺序）
    parents = {}  # {子关节: 父关节}
    offsets = {}  # {关节: [x,y,z] 偏移}
    channels = {}  # {关节: ['Xrotation', 'Yrotation', ...]}
    stack = []  # 父关节栈
    
    i = 0
    current_joint = None
    
    # 解析 HIERARCHY 部分
    while i < len(lines):
        line = lines[i]
        
        if line.startswith('ROOT'):
            name = line.split()[1]
            joints.append(name)
            parents[name] = None
            stack.append(name)
            current_joint = name
            i += 1
            
        elif line.startswith('JOINT'):
            name = line.split()[1]
            joints.append(name)
            parents[name] = stack[-1]
            stack.append(name)
            current_joint = name
            i += 1
            
        elif line.startswith('End Site'):
            name = current_joint + '_End'
            joints.append(name)
            parents[name] = stack[-1]
            stack.append(name)
            current_joint = name
            i += 1
            
        elif line.startswith('OFFSET'):
            parts = line.split()
            offsets[current_joint] = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
            i += 1
            
        elif line.startswith('CHANNELS'):
            parts = line.split()
            num_channels = int(parts[1])
            channel_names = parts[2:2+num_channels]
            channels[current_joint] = channel_names
            i += 1
            
        elif line.startswith('}'):
            if stack:
                stack.pop()
            i += 1
            
        elif line.startswith('MOTION'):
            i += 1
            break
            
        else:
            i += 1
    
    # 解析运动数据
    frame_count = 0
    frame_time = 0.033
    
    while i < len(lines):
        line = lines[i]
        if line.startswith('Frames:'):
            frame_count = int(line.split()[1])
        elif line.startswith('Frame Time:'):
            frame_time = float(line.split()[2])
        elif line and not line.startswith('Frame'):
            break
        i += 1
    
    # 读取所有帧数据
    motion_data = []
    while i < len(lines):
        line = lines[i].strip()
        if line:
            motion_data.append([float(x) for x in line.split()])
        i += 1
    
    return joints, parents, offsets, channels, np.array(motion_data), frame_time


def compute_joint_transforms(joints, parents, offsets, channels, frame_data):
    """
    计算每个关节的全局位置和旋转
    返回: {关节名: (位置[x,y,z], 旋转四元数[x,y,z,w])}
    """
    global_transforms = {}
    channel_idx = 0
    
    for joint_name in joints:
        if joint_name.endswith('_End'):
            # End Site 没有通道数据，只有位置
            parent_name = parents[joint_name]
            if parent_name in global_transforms:
                parent_pos, parent_quat = global_transforms[parent_name]
                parent_rot = R.from_quat(parent_quat)
                offset = offsets.get(joint_name, np.zeros(3))
                
                # 位置 = 父位置 + 父旋转 * 偏移
                pos = parent_pos + parent_rot.apply(offset)
                global_transforms[joint_name] = (pos, parent_quat)
            continue
        
        joint_channels = channels.get(joint_name, [])
        
        if parents[joint_name] is None:
            # ROOT 关节：通常有 6 个通道 (x,y,z 位置 + 旋转)
            pos = np.zeros(3)
            rot_angles = []
            rot_order = ''
            
            for ch in joint_channels:
                if 'position' in ch.lower() or ch in ['Xposition', 'Yposition', 'Zposition']:
                    axis = ch[0].lower()
                    if axis == 'x':
                        pos[0] = frame_data[channel_idx]
                    elif axis == 'y':
                        pos[1] = frame_data[channel_idx]
                    elif axis == 'z':
                        pos[2] = frame_data[channel_idx]
                    channel_idx += 1
                elif 'rotation' in ch.lower() or ch in ['Xrotation', 'Yrotation', 'Zrotation']:
                    rot_order += ch[0].lower()
                    rot_angles.append(frame_data[channel_idx])
                    channel_idx += 1
            
            # 将位置从厘米转为米（BVH 通常使用厘米）
            pos *= 0.01
            
            # 计算旋转四元数
            if rot_angles:
                rot = R.from_euler(rot_order, rot_angles, degrees=True)
                quat = rot.as_quat()  # [x, y, z, w]
            else:
                quat = np.array([0, 0, 0, 1])
            
            global_transforms[joint_name] = (pos, quat)
            
        else:
            # 子关节：只有旋转通道
            rot_angles = []
            rot_order = ''
            
            for ch in joint_channels:
                if 'rotation' in ch.lower() or ch in ['Xrotation', 'Yrotation', 'Zrotation']:
                    rot_order += ch[0].lower()
                    rot_angles.append(frame_data[channel_idx])
                    channel_idx += 1
            
            # 局部旋转
            if rot_angles:
                local_rot = R.from_euler(rot_order, rot_angles, degrees=True)
            else:
                local_rot = R.from_quat([0, 0, 0, 1])
            
            # 获取父关节的全局变换
            parent_name = parents[joint_name]
            parent_pos, parent_quat = global_transforms[parent_name]
            parent_rot = R.from_quat(parent_quat)
            
            # 偏移（从厘米转为米）
            offset = offsets.get(joint_name, np.zeros(3)) * 0.01
            
            # 全局位置 = 父位置 + 父旋转 * 偏移
            pos = parent_pos + parent_rot.apply(offset)
            
            # 全局旋转 = 父旋转 * 局部旋转
            global_rot = parent_rot * local_rot
            quat = global_rot.as_quat()
            
            global_transforms[joint_name] = (pos, quat)
    
    return global_transforms


class BVHPlayer(Node):
    """ROS 2 节点：播放 BVH 文件并发布 TF 变换"""
    
    def __init__(self, bvh_file, fps=30.0, loop=True, scale=1.0):
        super().__init__('bvh_player')
        
        self.get_logger().info(f'Loading BVH file: {bvh_file}')
        
        # 解析 BVH 文件
        self.joints, self.parents, self.offsets, self.channels, self.motion_data, frame_time = parse_bvh(bvh_file)
        
        # 应用缩放
        for joint in self.offsets:
            self.offsets[joint] *= scale
        
        self.get_logger().info(f'Loaded {len(self.motion_data)} frames, {len(self.joints)} joints')
        self.get_logger().info(f'Frame time: {frame_time}s, FPS: {1.0/frame_time:.1f}')
        
        # 参数
        self.fps = fps
        self.loop = loop
        self.current_frame = 0
        self.total_frames = len(self.motion_data)
        
        # TF 广播器
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # Marker 发布器（用于可视化骨骼连接）
        self.marker_pub = self.create_publisher(MarkerArray, 'bvh_skeleton', 10)
        
        # 定时器：按指定 FPS 播放
        timer_period = 1.0 / self.fps
        self.timer = self.create_timer(timer_period, self.publish_frame)
        
        self.get_logger().info(f'Playing at {self.fps} FPS, loop={self.loop}')
    
    def publish_frame(self):
        """发布当前帧的 TF 变换和可视化标记"""
        if self.current_frame >= self.total_frames:
            if self.loop:
                self.current_frame = 0
                self.get_logger().info('Looping back to frame 0')
            else:
                self.get_logger().info('Playback finished')
                self.timer.cancel()
                return
        
        # 计算当前帧的关节变换
        frame_data = self.motion_data[self.current_frame]
        transforms = compute_joint_transforms(
            self.joints, self.parents, self.offsets, self.channels, frame_data
        )
        
        # 当前时间戳
        now = self.get_clock().now().to_msg()
        
        # 发布 TF 变换
        for joint_name, (pos, quat) in transforms.items():
            if joint_name.endswith('_End'):
                continue  # End Site 不发布 TF
            
            t = TransformStamped()
            t.header.stamp = now
            t.header.frame_id = 'world'
            t.child_frame_id = joint_name
            
            t.transform.translation.x = float(pos[0])
            t.transform.translation.y = float(pos[1])
            t.transform.translation.z = float(pos[2])
            
            t.transform.rotation.x = float(quat[0])
            t.transform.rotation.y = float(quat[1])
            t.transform.rotation.z = float(quat[2])
            t.transform.rotation.w = float(quat[3])
            
            self.tf_broadcaster.sendTransform(t)
        
        # 发布骨骼连接的可视化标记
        self.publish_skeleton_markers(transforms, now)
        
        self.current_frame += 1
    
    def publish_skeleton_markers(self, transforms, timestamp):
        """发布 Marker 来可视化骨骼连接"""
        marker_array = MarkerArray()
        marker_id = 0
        
        # 发布关节球体
        for joint_name, (pos, _) in transforms.items():
            marker = Marker()
            marker.header.frame_id = 'world'
            marker.header.stamp = timestamp
            marker.ns = 'joints'
            marker.id = marker_id
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            
            marker.pose.position.x = float(pos[0])
            marker.pose.position.y = float(pos[1])
            marker.pose.position.z = float(pos[2])
            marker.pose.orientation.w = 1.0
            
            marker.scale.x = 0.02
            marker.scale.y = 0.02
            marker.scale.z = 0.02
            
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            
            marker_array.markers.append(marker)
            marker_id += 1
        
        # 发布骨骼连接线
        for joint_name, (pos, _) in transforms.items():
            parent_name = self.parents.get(joint_name)
            if parent_name and parent_name in transforms:
                parent_pos, _ = transforms[parent_name]
                
                marker = Marker()
                marker.header.frame_id = 'world'
                marker.header.stamp = timestamp
                marker.ns = 'bones'
                marker.id = marker_id
                marker.type = Marker.LINE_STRIP
                marker.action = Marker.ADD
                
                marker.scale.x = 0.01  # 线宽
                
                marker.color.r = 0.0
                marker.color.g = 0.5
                marker.color.b = 1.0
                marker.color.a = 1.0
                
                # 添加两个点：父关节和子关节
                from geometry_msgs.msg import Point
                p1 = Point()
                p1.x, p1.y, p1.z = float(parent_pos[0]), float(parent_pos[1]), float(parent_pos[2])
                
                p2 = Point()
                p2.x, p2.y, p2.z = float(pos[0]), float(pos[1]), float(pos[2])
                
                marker.points = [p1, p2]
                
                marker_array.markers.append(marker)
                marker_id += 1
        
        self.marker_pub.publish(marker_array)


def main():
    parser = argparse.ArgumentParser(description='播放 BVH 文件并在 RViz 中可视化')
    parser.add_argument('bvh_file', help='BVH 文件路径')
    parser.add_argument('--fps', type=float, default=30.0, help='播放帧率 (默认: 30)')
    parser.add_argument('--no-loop', action='store_true', help='播放一次后停止')
    parser.add_argument('--scale', type=float, default=1.0, help='缩放因子 (默认: 1.0)')
    
    args = parser.parse_args()
    
    rclpy.init()
    
    node = BVHPlayer(
        bvh_file=args.bvh_file,
        fps=args.fps,
        loop=not args.no_loop,
        scale=args.scale
    )
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
