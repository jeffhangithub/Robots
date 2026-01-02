"""
轨迹数据处理与存储模块 (Trajectory Data Processing and Storage Module)

文件作用 (Purpose):
    本模块提供了机器人运动轨迹的数据结构和处理功能，主要用于存储、处理和导出机器人的
    时间序列运动数据，包括关节角度、刚体变换、接触状态等信息。支持从BVH文件或实时传感器
    数据中构建轨迹，并进行插值、滤波和导出。

数据流 (Data Flow):
    输入数据流: BVH文件/Xsens实时数据 → PoseData → Trajectory.add_sample() 
    处理流程: 原始姿态数据 → 样条插值/Savitzky-Golay滤波 → 速度计算 
    输出数据流: Trajectory.to_dict()/save() → pickle文件/HDF5文件

输入输出 (Input/Output):
    输入:
        - PoseData对象: 包含单帧的刚体变换、关节角度、接触状态等
        - sample_dt: 采样时间间隔(秒)
        - BodyTransform: 单个刚体的位置和四元数姿态
    
    输出:
        - Trajectory对象: 完整的运动轨迹时间序列
        - pickle文件(.pkl): 包含关节轨迹、刚体变换、接触信息的字典
        - 字典格式数据: 可用于HDF5存储或进一步处理

使用方法 (Usage):
    本文件不能直接命令行执行，需作为模块导入使用:
    
    ```python
    from motion_retargeting.utils.trajectory import Trajectory, PoseData, BodyTransform
    
    # 创建轨迹对象
    traj = Trajectory(sample_dt=0.01)
    
    # 添加姿态数据样本
    pose = PoseData(transforms=[...], q=joint_angles, ...)
    traj.add_sample(pose)
    
    # 导出为pickle文件(默认20Hz)
    traj.save("output_path", out_dt=0.02)
    ```

项目引用 (Referenced By):
    本模块被以下文件引用:
    - bvh_parser.py: BVH文件解析器，使用Trajectory类存储解析结果
    - data_subscriber.py: ROS2订阅器，实时接收Xsens数据并构建Trajectory
    - wbik_solver.py / wbik_solver2.py: 全身逆运动学求解器，使用PoseData和BodyTransform
    - trajectory_hdf5.py: HDF5格式存储工具，使用Trajectory数据结构

使用环境要求 (Environment Requirements):
    Python版本: >= 3.8
    依赖包:
        - numpy: 数组操作和数值计算
        - numpy-quaternion: 四元数运算库
        - scipy: 信号滤波(savgol_filter)和样条插值(make_interp_spline, RotationSpline)
        - pickle: 数据序列化(Python标准库)
    
    ROS2环境: 本模块作为motion_retargeting包的一部分，需在ROS2 Humble环境中使用

注释信息 (Documentation Info):
    注释时间: 2026年1月3日
    注释人: Jeff
"""

import numpy as np
import quaternion
from dataclasses import dataclass
import pickle
from scipy.signal import savgol_filter
from scipy.interpolate import make_interp_spline
from scipy.spatial.transform import Rotation, RotationSpline

from typing import List


def derivative(data, sample_dt):
    return (
        np.diff(
            data,
            prepend=[data[0]],
            axis=0,
        )
        / sample_dt
    )


def angular_velocity(quaternions, sample_dt):
    q_dot_savgol = savgol_filter(x=quaternions, polyorder=3, window_length=5, axis=0, deriv=1, delta=sample_dt)
    angular_velocities = np.zeros((len(q_dot_savgol), 3))
    for i in range(len(q_dot_savgol)):
        q_dot_quat = quaternion.quaternion(*q_dot_savgol[i])
        q_quat = quaternion.quaternion(*quaternions[i])
        q_conj_quat = q_quat.conjugate()
        result = 2 * q_dot_quat * q_conj_quat
        # Extract vector part (x, y, z) as angular velocity
        angular_velocities[i, 0] = result.x
        angular_velocities[i, 1] = result.y
        angular_velocities[i, 2] = result.z
    return angular_velocities


@dataclass
class BodyTransform:
    """
    Transform info for a single body
    """

    name: str
    position: np.ndarray
    quaternion: np.ndarray


@dataclass
class PoseData:
    """
    Frame data for a single time step
        transforms: list[BodyTransform] # list of body transforms
        q: np.ndarray # minimal coordinates of the robot model
        body_aliases: dict # map of body names to mjcf names
        model_root: str # name of the root body in mjcf model
        contacts: np.ndarray # contact states
        position_error: float # absolute position error of all links wrt to the reference
        dt: float # sample time
        joint_order: list # order of the joints in the q vector
    """

    # transforms: list[BodyTransform]
    transforms: List[BodyTransform]
    q: np.ndarray
    body_aliases: dict
    model_root: str
    contacts: np.ndarray
    position_error: float
    dt: float
    joint_order: list


class BodyTimeSeries:
    """
    Data structure for storing body transform timeseries
    """

    def __init__(self, name: str, sample_dt: float):
        self.name = name
        self.sample_dt = sample_dt
        self.positions = None
        self.quaternions = None
        self._linear_velocities = None
        self._angular_velocities = None

    def add_sample(self, position: np.ndarray, quat: np.ndarray):
        """
        Add a sample to the body data

        Args:
            position (np.ndarray): position of the body in the world frame
            quat (np.ndarray): quaternion of the body in the world frame [w, x, y, z] order
        """
        np_quat = quaternion.as_float_array(quat)
        if self.positions is None:
            self.positions = position.copy()
            self.quaternions = np_quat.copy()
        else:
            self.positions = np.vstack((self.positions, position.copy()))
            self.quaternions = np.vstack((self.quaternions, np_quat.copy()))

    def to_dict(self, scalar_first: bool = True, out_dt: float = None):
        if out_dt is None:
            out_dt = self.sample_dt

        quaternions = quaternion.as_float_array(quaternion.unflip_rotors(quaternion.from_float_array(self.quaternions)))

        old_t = np.arange(self.positions.shape[0]) * self.sample_dt
        new_t = np.arange(old_t[0], old_t[-1], out_dt)

        positions_spline = make_interp_spline(old_t, self.positions, k=1, bc_type="not-a-knot", axis=0)
        sci_quats = Rotation.from_quat(quaternions)
        quat_spline = RotationSpline(old_t, sci_quats)

        positions = positions_spline(new_t)
        quaternions = quat_spline(new_t).as_quat()
        return {
            "position": positions,
            "quaternion": (quaternions if scalar_first else quaternions[:, [1, 2, 3, 0]]),
            "linear_velocity": derivative(positions, out_dt),
            "angular_velocity": angular_velocity(quaternions, out_dt),
        }


class Trajectory:
    def __init__(self, sample_dt: float = 0.001):
        self.contacts = None
        self.bodies: dict[str, BodyTimeSeries] = {}
        self.qs = None
        self._joint_velocities = None
        self.sample_dt = sample_dt

    def add_sample(
        self,
        pose_data: PoseData,
    ):
        """
        Add a pose data to the trajectory.

        Args:
            pose_data: PoseData object
        """
        contacts = pose_data.contacts.copy()

        if self.contacts is None:
            self.contacts = contacts
            self.qs = pose_data.q
            self.body_aliases = pose_data.body_aliases
            self.model_root = pose_data.model_root
            self.joint_order = pose_data.joint_order
        else:
            self.contacts = np.vstack([self.contacts, contacts])
            self.qs = np.vstack([self.qs, pose_data.q])

        for transform in pose_data.transforms:
            if transform.name in ["world", "universe"]:
                continue
            if transform.name not in self.bodies.keys():
                self.bodies[transform.name] = BodyTimeSeries(transform.name, self.sample_dt)
            self.bodies[transform.name].add_sample(transform.position, transform.quaternion)

    def to_dict(self, out_dt: float = None):
        if out_dt is None:
            out_dt = self.sample_dt

        old_t = np.arange(self.qs.shape[0]) * self.sample_dt
        new_t = np.arange(old_t[0], old_t[-1], out_dt)

        joint_position_spline = make_interp_spline(old_t, self.qs[:, 7:], k=1, bc_type="not-a-knot", axis=0)
        joint_positions = joint_position_spline(new_t)
        joint_velocities = joint_position_spline(new_t, 1)

        out_dict = {
            "dt": out_dt,
            "joint_positions": joint_positions,
            "joint_velocities": joint_velocities,
            "joint_order": self.joint_order,
            "body_aliases": self.body_aliases,
            "model_root": self.model_root,
            "contacts": self.contacts[np.abs(old_t - new_t[:, None]).argmin(1)],
            "transforms": {},
        }

        for name, body in self.bodies.items():
            body_dict = body.to_dict(out_dt=out_dt)

            out_dict["transforms"][name] = {}
            for k, v in body_dict.items():
                out_dict["transforms"][name][k] = v
        return out_dict

    def convert_ndarrays(self, data_dict):
        for k, v in data_dict.items():
            if isinstance(v, np.ndarray):
                data_dict[k] = v.tolist()
            if isinstance(v, dict):
                data_dict[k] = self.convert_ndarrays(v)
        return data_dict

    def save(self, path, out_dt: float = 0.02):
        # Have to remove numpy arrays entirely
        # to preserve compatibility between versions
        trajectory_dict = self.convert_ndarrays(self.to_dict(out_dt=out_dt))
        with open(path + ".pkl", "wb") as f:
            pickle.dump(trajectory_dict, f)


def to_numpy(data_dict):
    for k, v in data_dict.items():
        if isinstance(v, list):
            data_dict[k] = np.array(v)
        if isinstance(v, dict):
            data_dict[k] = to_numpy(v)
    return data_dict
