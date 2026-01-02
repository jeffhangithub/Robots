"""
四元数工具函数库 (Quaternion Utility Functions Library)

文件作用 (Purpose):
    本模块提供了全面的四元数运算和旋转表示转换工具，支持PyTorch张量和NumPy数组两种数据格式。
    主要功能包括:
    - 四元数基本运算（归一化、乘法、求逆）
    - 旋转表示转换（四元数↔旋转矩阵↔轴角）
    - 向量旋转操作（使用四元数旋转3D向量）
    - 角速度计算（从连续四元数序列推导）
    - 偏航角提取（从旋转矩阵中提取绕Z轴旋转）
    
    本模块是机器人运动学计算的基础工具，广泛应用于IK求解、轨迹处理、可视化等场景。

数据流 (Data Flow):
    输入: 
        - 四元数 (PyTorch Tensor / NumPy array)
        - 旋转矩阵 (3×3 或 batch×3×3)
        - 轴角表示 (axis + angle 或 axis_angle向量)
        - 3D向量 (待旋转的向量)
    
    处理: 
        - 格式转换和标准化
        - 数学运算（乘法、求逆、插值）
        - 坐标变换和旋转应用
    
    输出: 转换后的旋转表示（四元数、旋转矩阵、偏航矩阵等）

输入输出 (Input/Output):
    核心函数:
    
    1. normalize(quaternions: torch.Tensor) -> torch.Tensor
       输入: 四元数张量 (batch, 4) [w, x, y, z]
       输出: 归一化后的四元数，确保w≥0（标准化方向）
    
    2. axis_angle_to_quat(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor
       输入: 旋转轴 (batch, 3), 旋转角度 (batch, 1) 弧度
       输出: 四元数 (batch, 4) [w, x, y, z]
    
    3. quat_multiply(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor
       输入: 两个四元数 u, v (batch, 4)
       输出: 四元数乘积 u*v (batch, 4)
    
    4. rotate(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor
       输入: 四元数 (batch, 4), 向量 (batch, 3)
       输出: 旋转后的向量 (batch, 3)
    
    5. quat_inv(q: torch.Tensor) -> torch.Tensor
       输入: 四元数 (batch, 4)
       输出: 四元数的逆 (batch, 4)
    
    6. angular_velocity(q1, q2, dt: float)
       输入: 初始四元数 q1, 终止四元数 q2, 时间间隔 dt
       输出: 角速度向量 (batch, 3) rad/s
    
    7. yaw_matrix(rotation_matrix: np.ndarray) -> np.ndarray
       输入: 旋转矩阵 (3, 3) 或 (batch, 3, 3)
       输出: 仅保留偏航(yaw)的旋转矩阵（忽略pitch和roll）
    
    8. quat2mat(q: np.ndarray, scalar_last=False) -> np.ndarray
       输入: 四元数 (batch, 4), scalar_last指定四元数格式
       输出: 旋转矩阵 (batch, 3, 3)
    
    9. axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor
       输入: 轴角向量 (batch, 3) - 方向为轴，模长为角度
       输出: 四元数 (batch, 4) [w, x, y, z]

使用方法 (Usage):
    本文件为工具库，不可直接执行。需要在其他Python模块中导入使用:
    
    ```python
    import torch
    from motion_retargeting.utils.quat_utils import normalize, quat_multiply, rotate
    
    # 示例1: 旋转向量
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])  # 单位四元数(无旋转)
    vec = torch.tensor([[1.0, 0.0, 0.0]])         # X轴方向
    rotated_vec = rotate(quat, vec)
    
    # 示例2: 四元数乘法（组合旋转）
    q1 = torch.tensor([[0.707, 0.707, 0.0, 0.0]])  # 绕X轴转90°
    q2 = torch.tensor([[0.707, 0.0, 0.707, 0.0]])  # 绕Y轴转90°
    q_combined = quat_multiply(q1, q2)              # 组合旋转
    
    # 示例3: 提取偏航角（仅保留绕Z轴的旋转）
    import numpy as np
    rotation_mat = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])  # 绕Z轴转90°
    yaw_mat = yaw_matrix(rotation_mat)  # 提取偏航分量
    ```

项目引用 (Referenced By):
    本模块被广泛引用，是运动重定向系统的基础工具:
    
    1. wbik_solver.py / wbik_solver2.py
       - 使用 yaw_matrix() 提取脚部偏航角，用于接触约束
    
    2. torch_fk.py
       - 使用 quat_multiply(), rotate(), axis_angle_to_quat()
       - 在批量正运动学计算中进行四元数旋转操作
    
    3. retarget.py / retarget_online.py
       - 使用 yaw_matrix() 处理人体骨架的偏航对齐
       - 在BVH数据解析和重定向映射中使用
    
    4. mujoco/renderer.py
       - 使用 quat2mat() 将四元数转换为旋转矩阵
       - 用于可视化渲染中的坐标系绘制
    
    几乎所有涉及旋转计算的模块都依赖本工具库。

使用环境要求 (Environment Requirements):
    Python版本: >= 3.8
    
    依赖包:
        - torch >= 1.10: PyTorch深度学习框架（支持CPU/GPU）
            - 用于GPU加速的四元数批量运算
            - 支持自动微分（可用于基于梯度的优化）
        - numpy: 数组操作和数值计算
            - 用于NumPy格式的四元数转换
        - numpy-quaternion: 扩展的四元数运算库
            - 提供高级四元数操作（如from_rotation_matrix, as_rotation_matrix）
        - torch.nn.functional: PyTorch函数库
            - 使用F.normalize进行张量归一化
    
    硬件支持:
        - CPU模式: 标准多核处理器即可
        - GPU模式: CUDA兼容显卡，可大幅加速批量四元数运算
    
    ROS2环境: 不强制依赖，但通常作为motion_retargeting包的一部分在ROS2环境中使用

注释信息 (Documentation Info):
    注释时间: 2026年1月3日
    注释人: Jeff
"""

import torch
import numpy as np
import quaternion
from torch.nn import functional as F


def normalize(quaternions: torch.Tensor):
    target_quat = None
    # Normalization is done in the backward pass if the input requires grad
    # Can't normalize if there is a backward pass because of the inplace operation
    if quaternions.requires_grad:
        target_quat = quaternions
    else:
        target_quat = F.normalize(quaternions)

    return torch.where(target_quat[:, 0].unsqueeze(1) < 0, -target_quat, target_quat)


def axis_angle_to_quat(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Provides a quaternion that describes rotating around axis by angle.

    Args:
      axis: (3,) axis (x,y,z)
      angle: () float angle to rotate by

    Returns:
      A quaternion that rotates around axis by angle
    """
    s, c = torch.sin(angle * 0.5), torch.cos(angle * 0.5)
    return normalize(torch.cat([c, axis * s], dim=1))


def quat_multiply(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Multiplies two quaternions.

    Args:
      u: (4,) quaternion (w,x,y,z)
      v: (4,) quaternion (w,x,y,z)

    Returns:
      A quaternion u * v.
    """
    quaternion = torch.stack(
        [
            u[:, 0] * v[:, 0] - u[:, 1] * v[:, 1] - u[:, 2] * v[:, 2] - u[:, 3] * v[:, 3],
            u[:, 0] * v[:, 1] + u[:, 1] * v[:, 0] + u[:, 2] * v[:, 3] - u[:, 3] * v[:, 2],
            u[:, 0] * v[:, 2] - u[:, 1] * v[:, 3] + u[:, 2] * v[:, 0] + u[:, 3] * v[:, 1],
            u[:, 0] * v[:, 3] + u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1] + u[:, 3] * v[:, 0],
        ]
    ).T
    return normalize(quaternion)


def rotate(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotates a vector vec by a unit quaternion quat.

    Args:
      vec: (3,) a vector
      quat: (4,) a quaternion

    Returns:
      ndarray(3) containing vec rotated by quat.
    """
    quat = normalize(quat)
    s, u = quat[:, 0].unsqueeze(1), quat[:, 1:]
    u_dot_vec = (u * vec).sum(1).unsqueeze(1)
    u_dot_u = (u * u).sum(1).unsqueeze(1)
    r = 2 * (u_dot_vec * u) + ((s * s) - u_dot_u) * vec
    r = r + 2 * s * torch.cross(u, vec, dim=1)
    return r


def quat_inv(q: torch.Tensor) -> torch.Tensor:
    """Calculates the inverse of quaternion q.

    Args:
      q: (4,) quaternion [w, x, y, z]

    Returns:
      The inverse of q, where qmult(q, inv_quat(q)) = [1, 0, 0, 0].
    """
    return normalize(q * torch.tensor([[1, -1, -1, -1]], device=q.device))


def angular_velocity(q1, q2, dt: float):
    need_flatten = len(q1.shape) > 2
    batch_size = q1.shape[0]
    if need_flatten:
        q1 = q1.view(-1, 4)
        q2 = q2.view(-1, 4)
    if isinstance(q1, torch.Tensor):
        stack_func = torch.column_stack
    else:
        stack_func = np.column_stack
    velocities = (2 / dt) * stack_func(
        [
            q1[:, 0] * q2[:, 1] - q1[:, 1] * q2[:, 0] - q1[:, 2] * q2[:, 3] + q1[:, 3] * q2[:, 2],
            q1[:, 0] * q2[:, 2] + q1[:, 1] * q2[:, 3] - q1[:, 2] * q2[:, 0] - q1[:, 3] * q2[:, 1],
            q1[:, 0] * q2[:, 3] - q1[:, 1] * q2[:, 2] + q1[:, 2] * q2[:, 1] - q1[:, 3] * q2[:, 0],
        ],
    )
    if need_flatten:
        return velocities.view((batch_size, -1, 3))
    return velocities


def yaw_matrix(rotation_matrix: np.ndarray):
    """
    Returns a quaternion that keeps only the yaw rotation of the given rotation matrix
    """

    quat_yaw = quaternion.from_rotation_matrix(rotation_matrix)
    if len(rotation_matrix.shape) == 2:
        quat_yaw.x = 0.0
        quat_yaw.y = 0.0
    else:
        np_quats = quaternion.as_float_array(quat_yaw)
        np_quats[:, 1:3] = 0
        quat_yaw = quaternion.from_float_array(np_quats)
    return quaternion.as_rotation_matrix(np.normalized(quat_yaw))


def quat2mat(q: np.ndarray, scalar_last=False, eps=np.finfo(np.float64).eps):
    """
    Convert quaternions to rotation matrices.

    Args:
        q: Quaternions to convert, shape (..., 4).
        scalar_last: If True, the scalar (w) is the last element of q.
    """
    if len(q.shape) == 1:
        q = q.reshape(1, -1)

    batch_size = q.shape[0]
    if scalar_last:
        x, y, z, w = np.split(q, 4, axis=-1)
    else:
        w, x, y, z = np.split(q, 4, axis=-1)

    Nq = w * w + x * x + y * y + z * z
    out = np.zeros((batch_size, 3, 3))
    out[Nq.reshape(batch_size) < eps] = np.eye(3)
    s = 2.0 / Nq
    X = x * s
    Y = y * s
    Z = z * s
    wX = w * X
    wY = w * Y
    wZ = w * Z
    xX = x * X
    xY = x * Y
    xZ = x * Z
    yY = y * Y
    yZ = y * Z
    zZ = z * Z
    out[:, 0, 0] = (1.0 - (yY + zZ)).reshape(-1)
    out[:, 0, 1] = (xY - wZ).reshape(-1)
    out[:, 0, 2] = (xZ + wY).reshape(-1)
    out[:, 1, 0] = (xY + wZ).reshape(-1)
    out[:, 1, 1] = (1.0 - (xX + zZ)).reshape(-1)
    out[:, 1, 2] = (yZ - wX).reshape(-1)
    out[:, 2, 0] = (xZ - wY).reshape(-1)
    out[:, 2, 1] = (yZ + wX).reshape(-1)
    out[:, 2, 2] = (1.0 - (xX + yY)).reshape(-1)
    return out


def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = 0.5 - (angles[small_angles] * angles[small_angles]) / 48
    quaternions = torch.cat([torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1)
    return quaternions
