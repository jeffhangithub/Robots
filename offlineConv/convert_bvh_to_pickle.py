#!/usr/bin/env python3
"""
独立的 BVH 到 pickle 转换脚本
用于 G1 机器人动作数据的转换，无需 ROS2
"""

import numpy as np
import pickle
import os
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import sys

# 添加项目路径到 Python 搜索路径
# 指向 src/motion_retargeting 目录，以便能导入内部的 motion_retargeting 包
sys.path.insert(0, '/home/jeff/Codes/Robots/src/motion_retargeting')

try:
    from motion_retargeting.retarget.retarget import BVHRetarget, Joint
    from motion_retargeting.config.robot.g1 import G1_BVH_CONFIG
    from motion_retargeting.utils.trajectory import Trajectory
    RETARGET_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  无法导入 motion_retargeting: {e}")
    RETARGET_AVAILABLE = False


def parse_bvh_simple(bvh_file):
    """
    简单的 BVH 解析器，提取骨架结构和动作数据
    返回 (skeleton, motions)
    """
    with open(bvh_file, 'r') as f:
        lines = f.readlines()
    
    # 跳过空行
    lines = [l.strip() for l in lines if l.strip()]
    
    skeleton = {}
    parent_map = {}
    joint_order = []
    
    # 解析 HIERARCHY
    i = 0
    current_parent = None
    
    while i < len(lines):
        line = lines[i]
        
        if line.startswith('ROOT'):
            parts = line.split()
            joint_name = parts[1]
            joint_order.append(joint_name)
            skeleton[joint_name] = {'name': joint_name, 'parent': None, 'offset': [0, 0, 0]}
            current_parent = joint_name
            i += 1
        
        elif line.startswith('JOINT'):
            parts = line.split()
            joint_name = parts[1]
            joint_order.append(joint_name)
            skeleton[joint_name] = {'name': joint_name, 'parent': current_parent, 'offset': [0, 0, 0]}
            parent_map[joint_name] = current_parent
            i += 1
        
        elif line.startswith('OFFSET'):
            parts = line.split()
            offset = [float(parts[1]), float(parts[2]), float(parts[3])]
            # 指向最后添加的关节
            if joint_order:
                skeleton[joint_order[-1]]['offset'] = offset
            i += 1
        
        elif line.startswith('CHANNELS'):
            i += 1
        
        elif line == '}':
            # 回退到父级
            if current_parent:
                if current_parent in parent_map:
                    current_parent = parent_map[current_parent]
                else:
                    current_parent = None
            i += 1
        
        elif line.startswith('MOTION'):
            # 开始解析动作数据
            break
        
        else:
            i += 1
    
    # 解析动作数据
    motions = []
    if 'MOTION' in lines[i]:
        i += 1
        # 跳过 Frames
        while i < len(lines) and not lines[i].startswith('Frame'):
            i += 1
        i += 1  # 跳过 "Frame Time:" 行
        
        # 解析每一帧
        while i < len(lines):
            line = lines[i].strip()
            if not line or line.startswith('#'):
                i += 1
                continue
            
            try:
                values = [float(x) for x in line.split()]
                # 创建帧数据字典（mapping 关节顺序 -> 值）
                frame_data = {}
                for j, joint in enumerate(joint_order):
                    if joint == joint_order[0]:  # ROOT 有 6 个通道 (x, y, z, rx, ry, rz)
                        frame_data[joint] = values[j*6:(j+1)*6]
                    else:  # 其他关节 3 个通道 (rx, ry, rz)
                        frame_data[joint] = values[j*3:(j+1)*3]
                motions.append(frame_data)
            except (ValueError, IndexError):
                pass
            
            i += 1
    
    return skeleton, motions, joint_order


def load_bvh_for_retarget(bvh_file):
    """
    基于 motion_retargeting 的 Joint 结构解析 BVH，便于后续 IK 重定向。
    
    该函数执行以下操作：
    1. 读取 BVH 文件内容，分离 HIERARCHY（骨架结构）和 MOTION（动作数据）部分。
    2. 解析 MOTION 部分的元数据，提取帧数和帧时间，计算帧率 (FPS)。
    3. 解析 HIERARCHY 部分，构建关节字典和父子关系映射，处理嵌套结构。
    4. 解析 MOTION 部分的每一帧数据，根据关节通道定义提取位置和旋转信息。
       - 位置数据转换为米 (乘以 0.01)。
       - 旋转数据转换为弧度。
    5. 构建 Joint 对象树，根据解析出的关节信息和父子关系创建完整的骨架结构。
    6. 返回构建好的骨架字典 (skeleton)、动作数据列表 (motion_data) 和帧率 (bvh_fps)。
    """
    with open(bvh_file, 'r') as f:
        content = f.readlines()

    hierarchy_start = content.index("HIERARCHY\n")
    hierarchy_end = content.index("MOTION\n")
    hierarchy = content[hierarchy_start:hierarchy_end]

    motion = content[hierarchy_end:]
    frames_line = next(line for line in motion if line.startswith("Frames:"))
    frame_count = int(frames_line.split()[1])
    frame_time_line = next(line for line in motion if line.startswith("Frame Time:"))
    frame_time = float(frame_time_line.split()[2])
    bvh_fps = int(round(1.0 / frame_time)) if frame_time > 0 else 60

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

    ASF_TO_METERS = 0.01
    motion_data = []
    data_lines = [l.split() for l in motion if l.strip() and not l.startswith(("Frames:", "Frame Time:"))]
    for i, line in enumerate(data_lines):
        if i == 1 or len(line) < 6:
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
                else:
                    data.append(np.deg2rad(float(line[idx])))
                idx += 1
            frame_data[joint] = data
        motion_data.append(frame_data)

    skeleton = {}
    for name, data in joints.items():
        if name.endswith("_end"):
            continue
        offset = data["offset"] * ASF_TO_METERS if data["offset"] is not None else np.zeros(3)
        dof = data["channels"]
        limits = [(-180, 180)] * 3
        joint = Joint(name, offset, dof, limits)
        if name in parents:
            parent_name = parents[name]
            if parent_name in skeleton:
                skeleton[parent_name].children.append(joint)
                joint.parent = skeleton[parent_name]
        skeleton[name] = joint

    root_name = next(name for name in joints if "Hips" in name)
    skeleton[root_name].parent = None
    return skeleton, motion_data, bvh_fps


def save_retargeted_pickle(trajectory, robot_config, output_pkl, bvh_fps, bvh_axis=None):
    """将 retarget 结果保存为与可视化兼容的 pkl。"""
    # 如果 bvh_axis 为 None，则初始化为空字典
    bvh_axis = bvh_axis or {}
    # 初始化 motion_data 字典，用于存储最终保存的数据
    motion_data = {}
    # 将轨迹对象转换为字典格式，指定输出的时间间隔为 1/FPS
    traj_dict = trajectory.to_dict(out_dt=1 / bvh_fps)
    # 获取关节位置数据
    joint_positions = traj_dict["joint_positions"]
    # 获取总帧数 T
    T = len(joint_positions)
    # 获取所有关节的变换矩阵（位置和旋转）
    transforms = traj_dict["transforms"]
    # 获取机器人根关节的名称（通常是 body_links 列表的第一个元素）
    root_name = robot_config.body_links[0]

    # 提取根关节在每一帧的位置
    root_positions = np.array([transforms[root_name]["position"][i] for i in range(T)])
    # 提取根关节在每一帧的旋转（四元数）
    root_orientations = np.array([transforms[root_name]["quaternion"][i] for i in range(T)])
    # 对根关节位置进行坐标系转换（例如从 Y-up 到 Z-up，具体取决于坐标系定义），这里是 x, y 取反
    root_positions = root_positions * np.array([-1, -1, 1])
    # 遍历每一帧，调整根关节四元数的符号（x, y 取反），以匹配坐标系变换
    for i in range(T):
        q = root_orientations[i]
        root_orientations[i] = np.array([-q[0], -q[1], q[2], q[3]])

    # 初始化局部身体位置数组，形状为 (T, 身体连杆数, 3)
    local_body_pos = np.zeros((T, len(robot_config.body_links), 3), dtype=np.float32)
    # 遍历每个身体连杆
    for j, key in enumerate(robot_config.body_links):
        # 如果该连杆在变换数据中存在
        if key in transforms:
            # 提取该连杆在每一帧的位置
            positions = np.array([transforms[key]["position"][i] for i in range(T)])
            # 同样进行坐标系转换（x, y 取反）
            positions = positions * np.array([-1, -1, 1])
            # 存入 local_body_pos 数组
            local_body_pos[:, j, :] = positions

    # 初始化自由度位置（关节角度）数组，形状为 (T, 身体连杆数)
    dof_pos = np.zeros((T, len(robot_config.body_links)), dtype=np.float32)
    # 遍历每一帧
    for i in range(T):
        # 遍历每个身体连杆
        for j, key in enumerate(robot_config.body_links):
            # 如果该连杆不在变换数据中，跳过
            if key not in transforms:
                continue
            # 获取当前连杆的旋转（四元数转旋转对象）
            rot = R.from_quat(transforms[key]["quaternion"][i])
            # 如果是模型根节点，跳过（根节点通常由 root_pos/root_rot 处理）
            if key == traj_dict["model_root"]:
                continue
            # 获取父连杆的旋转
            rot_parent = R.from_quat(transforms[robot_config.body_parent_links[j]]["quaternion"][i])
            # 获取该关节的旋转轴，默认为 Z 轴 [0, 0, 1]
            axis = np.array(bvh_axis.get(key, [0, 0, 1]))
            # 计算子连杆相对于父连杆的旋转：R_child_local = R_parent_inv * R_child
            # 注意：这里代码写的是 rot_parent * rot.inv()，这通常计算的是 R_child 到 R_parent 的相对旋转
            rotation_child_parent = rot_parent * rot.inv()
            # 提取绕指定轴的旋转角度
            angle = _extract_rotation_about_axis(rotation_child_parent.as_matrix(), axis)
            # 存入 dof_pos 数组
            dof_pos[i][j] = angle

    # 将处理好的数据存入 motion_data 字典
    motion_data["root_pos"] = root_positions
    motion_data["root_rot"] = root_orientations
    motion_data["local_body_pos"] = local_body_pos
    # dof_pos 去掉第一列（通常对应根节点，不需要作为关节角度存储）
    motion_data["dof_pos"] = dof_pos[:, 1:]
    motion_data["fps"] = bvh_fps
    motion_data["link_body_list"] = robot_config.body_links

    # 创建输出目录（如果不存在）
    os.makedirs(os.path.dirname(output_pkl), exist_ok=True)
    # 将 motion_data 字典保存为 pickle 文件
    with open(output_pkl, 'wb') as f:
        pickle.dump(motion_data, f)


def _extract_rotation_about_axis(rotation_matrix: np.ndarray, axis: np.ndarray) -> float:
    axis = np.asarray(axis, dtype=np.float64).flatten()
    axis = axis / (np.linalg.norm(axis) + 1e-8)
    rotation_matrix = np.asarray(rotation_matrix, dtype=np.float64).reshape(3, 3)
    cos_angle = (np.trace(rotation_matrix) - 1) / 2
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    if abs(angle) < 1e-10:
        return 0.0
    rx = rotation_matrix[2, 1] - rotation_matrix[1, 2]
    ry = rotation_matrix[0, 2] - rotation_matrix[2, 0]
    rz = rotation_matrix[1, 0] - rotation_matrix[0, 1]
    rotation_axis = np.array([rx, ry, rz])
    rotation_axis_norm = np.linalg.norm(rotation_axis)
    if rotation_axis_norm < 1e-10:
        return angle
    rotation_axis = rotation_axis / rotation_axis_norm
    axis_dot = np.dot(rotation_axis, axis)
    if axis_dot < 0:
        angle = -angle
        axis_dot = -axis_dot
    return angle * axis_dot


def extract_motion_data_from_bvh(bvh_file, output_pkl, robot_name='g1'):
    """
    从 BVH 文件提取动作数据并保存为 pickle
    """
    print(f"📖 读取 BVH 文件: {bvh_file}")

    # 优先尝试 motion_retargeting 完整管道
    if RETARGET_AVAILABLE:
        try:
            # 使用 load_bvh_for_retarget 函数加载 BVH 文件，获取骨架、动作数据和帧率
            skeleton, motions, bvh_fps = load_bvh_for_retarget(bvh_file)
            # 如果没有找到动作数据，打印错误并返回 False
            if not motions:
                print("❌ 未找到动作数据")
                return False
            # 打印解析成功的相关信息
            print(f"✅ 解析完成（retarget 模式），找到 {len(motions)} 帧，{len(skeleton)} 个关节，FPS={bvh_fps}")

            # 初始化 BVHRetarget 对象，传入帧率和机器人配置参数
            retargeter = BVHRetarget(bvh_dataset_fps=bvh_fps, wbik_params=G1_BVH_CONFIG)
            # 设置重定向器的动作数据
            retargeter.set_motion(skeleton, motions)

            # 初始化轨迹记录对象，设置采样时间间隔
            trajectory = Trajectory(sample_dt=1.0 / bvh_fps)
            # 遍历重定向器生成的每一帧姿态数据，添加到轨迹中
            for pose_data in retargeter:
                trajectory.add_sample(pose_data)

            # 获取机器人配置中的 bvh_axis 参数，如果不存在则为空字典
            bvh_axis = getattr(G1_BVH_CONFIG, 'bvh_axis', {})
            # 将生成的轨迹数据保存为 pickle 文件
            save_retargeted_pickle(trajectory, G1_BVH_CONFIG, output_pkl, bvh_fps, bvh_axis=bvh_axis)
            # 打印保存成功的消息
            print(f"✅ 已使用 motion_retargeting 生成并保存: {output_pkl}")
            # 返回 True 表示成功
            return True
        except Exception as e:  # 回退到简易解析
            # 如果发生异常，打印警告信息并继续执行后续的简易解析逻辑
            print(f"⚠️  motion_retargeting 处理失败，改用简易解析: {e}")

    # 简单解析 BVH（回退路径）
    skeleton, motions, joint_order = parse_bvh_simple(bvh_file)
    if not motions:
        print("❌ 未找到动作数据")
        return False
    print(f"✅ 解析完成（简易模式），找到 {len(motions)} 帧，{len(skeleton)} 个关节")

    T = len(motions)
    root_pos = np.zeros((T, 3), dtype=np.float32)
    root_rot = np.zeros((T, 4), dtype=np.float32)
    dof_pos = np.zeros((T, len(joint_order) - 1), dtype=np.float32)
    local_body_pos = np.zeros((T, len(joint_order), 3), dtype=np.float32)

    for frame_idx, frame_data in enumerate(motions):
        if joint_order[0] in frame_data:
            root_values = frame_data[joint_order[0]]
            root_pos[frame_idx] = [root_values[0] * 0.01, root_values[1] * 0.01, root_values[2] * 0.01]
            euler_angles = np.array([root_values[3], root_values[4], root_values[5]])
            rot = R.from_euler('xyz', euler_angles, degrees=True)
            quat = rot.as_quat()
            root_rot[frame_idx] = [quat[3], quat[0], quat[1], quat[2]]
        for joint_idx, joint in enumerate(joint_order[1:], 1):
            if joint in frame_data:
                joint_values = frame_data[joint]
                dof_pos[frame_idx, joint_idx - 1] = joint_values[0]

    motion_data = {
        'fps': 60,
        'root_pos': root_pos,
        'root_rot': root_rot,
        'dof_pos': dof_pos,
        'local_body_pos': local_body_pos,
        'link_body_list': joint_order,
    }

    os.makedirs(os.path.dirname(output_pkl), exist_ok=True)
    with open(output_pkl, 'wb') as f:
        pickle.dump(motion_data, f)

    print(f"✅ 已保存到: {output_pkl}")
    print(f"   - 帧数: {T}")
    print(f"   - 关节数: {len(joint_order)}")
    print(f"   - DOF: {dof_pos.shape[1]}")
    return True


if __name__ == '__main__':
    # 设置输入 BVH 文件的路径
    bvh_file = '/home/jeff/Codes/Robots/data/Geely test-001.bvh'
    # 设置输出目录
    output_dir = '/home/jeff/Codes/Robots/output/g1'
    # 拼接输出 pickle 文件的完整路径
    output_pkl = os.path.join(output_dir, 'Geely test-001.pkl')
    
    # 检查输入文件是否存在
    if not os.path.exists(bvh_file):
        # 如果文件不存在，打印错误信息并退出
        print(f"❌ BVH 文件不存在: {bvh_file}")
        sys.exit(1)
    
    # 调用核心函数进行转换，传入输入文件、输出路径和机器人名称
    success = extract_motion_data_from_bvh(bvh_file, output_pkl, robot_name='g1')
    
    # 根据转换结果输出相应信息
    if success:
        # 转换成功，打印提示信息和后续可视化命令
        print("\n🎯 转换成功！现在可以运行可视化:")
        print(f"/usr/bin/python /home/jeff/Codes/Robots/src/vis_robot_motion.py \\")
        print(f"  --xml_path /home/jeff/Codes/Robots/src/motion_retargeting/robots/g1/urdf/g1.xml \\")
        print(f"  --robot_motion_path '{output_pkl}'")
    else:
        # 转换失败，打印错误信息并退出
        print("\n❌ 转换失败")
        sys.exit(1)
