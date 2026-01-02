"""
PyTorch批量正运动学计算模块 (PyTorch Batched Forward Kinematics Module)

文件作用 (Purpose):
    本模块基于PyTorch实现了针对MuJoCo机器人模型的批量正运动学(Forward Kinematics, FK)计算。
    核心功能包括:
    - 从MJCF文件加载机器人模型结构（刚体树、关节配置）
    - 高效批量计算给定关节配置下所有刚体的世界坐标位姿
    - 支持GPU加速的张量运算，适合大批量FK计算（如采样、优化）
    - 提供随机配置生成器（用于采样、初始化等场景）
    
    与Pinocchio/MuJoCo原生FK的区别:
    - 支持批量处理 (batch_size × nq → batch_size × nbodies × 7)
    - 基于PyTorch，支持GPU加速和自动微分
    - 完全可微分，可用于基于梯度的优化

数据流 (Data Flow):
    输入: 关节配置张量 q (batch_size, nq)
          ├─ [0:3]: 浮动基座位置 (x, y, z)
          ├─ [3:7]: 浮动基座姿态四元数 (w, x, y, z) scalar-first
          └─ [7:nq]: 各关节角度
    
    处理流程:
        1. 从MJCF文件解析刚体树结构（父子关系、局部变换）
        2. 从根节点开始递归计算刚体全局变换
        3. 对于带关节的刚体，应用关节旋转（轴角→四元数→位姿变换）
        4. 累积计算所有刚体的世界坐标位置和姿态
    
    输出: (body_names, body_positions, body_quaternions)
          ├─ body_names: 刚体名称列表 [str, ...]
          ├─ body_positions: 位置张量 (batch_size, nbodies, 3)
          └─ body_quaternions: 姿态四元数 (batch_size, nbodies, 4) scalar-first

输入输出 (Input/Output):
    类初始化:
        - 输入: 
            - mjcf_file: 机器人MJCF模型文件路径 (.xml)
            - device: PyTorch设备 (torch.device, 如 'cuda:0' 或 'cpu')
        - 输出: TorchForwardKinematics实例
    
    主要方法:
        - forward_kinematics(q, pin_notation=False):
            输入: 
                - q: 关节配置张量 (batch_size, nq) 或 (nq,)
                - pin_notation: 是否使用Pinocchio四元数顺序 (xyzw → wxyz)
            输出: (body_names, body_positions, body_quaternions)
                - body_names: list[str], 长度为nbodies
                - body_positions: torch.Tensor (batch_size, nbodies, 3)
                - body_quaternions: torch.Tensor (batch_size, nbodies, 4)
        
        - random_configuration(batch_size):
            输入: batch_size: 采样数量
            输出: torch.Tensor (batch_size, nq) - 满足关节限制的随机配置
        
        - nq: 属性，返回关节自由度总数 (包含浮动基座7维)
        - nbodies: 属性，返回刚体数量

使用方法 (Usage):
    本文件为工具库，不可直接执行。需作为模块导入使用:
    
    ```python
    import torch
    from motion_retargeting.utils.torch_fk import TorchForwardKinematics
    
    # 初始化FK求解器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fk = TorchForwardKinematics(
        mjcf_file="robot.xml",
        device=device
    )
    
    # 批量生成随机配置
    q_batch = fk.random_configuration(batch_size=1000)
    
    # 批量计算正运动学
    body_names, positions, quaternions = fk.forward_kinematics(q_batch)
    
    # 提取特定刚体（如左手）的位置
    left_hand_idx = body_names.index('left_hand')
    left_hand_pos = positions[:, left_hand_idx, :]  # (1000, 3)
    ```

项目引用 (Referenced By):
    当前未被项目中其他Python文件直接引用，但提供了可用于以下场景的基础设施:
    - 基于采样的IK求解（随机采样+FK验证）
    - 轨迹优化（需要可微FK进行梯度计算）
    - 批量配置验证（碰撞检测、可达性分析）
    - 机器学习训练（需要GPU加速的FK计算）
    
    潜在使用场景:
    - 替代Pinocchio的FK进行GPU加速批量计算
    - 基于梯度的IK优化（与wbik_solver的QP方法互补）
    - 强化学习中的策略网络正向传播

使用环境要求 (Environment Requirements):
    Python版本: >= 3.8
    
    依赖包:
        - torch >= 1.10: PyTorch深度学习框架（支持CPU/GPU）
        - numpy: 数组操作
        - mujoco >= 2.3: MuJoCo物理引擎Python绑定（用于加载MJCF）
        - motion_retargeting.utils.quat_utils: 四元数运算工具
            - quat_multiply: 四元数乘法
            - rotate: 四元数旋转向量
            - axis_angle_to_quat: 轴角转四元数
    
    硬件推荐:
        - CPU模式: 多核处理器，批量计算建议batch_size < 100
        - GPU模式: CUDA兼容显卡，可处理batch_size > 1000的大批量计算
    
    ROS2环境: 本模块可独立运行，不强制依赖ROS2环境

注释信息 (Documentation Info):
    注释时间: 2026年1月3日
    注释人: Jeff
"""

import torch
import numpy as np
import mujoco
from motion_retargeting.utils.quat_utils import quat_multiply, rotate, axis_angle_to_quat


class TorchForwardKinematics:
    """
    Torch based batched forward kinematics for MuJoCo robot model
    """

    def __init__(self, mjcf_file: str, device: torch.device):
        self.device = device
        self.mj_model = mujoco.MjModel.from_xml_path(mjcf_file)
        self.parents = {}
        self.local_position = {}
        self.local_rotation = {}
        self.joint_offsets = {}
        self.joint_axes = {}
        self.dof_ids = {}
        self.body_names = {}

        self.lower_limit = (
            torch.from_numpy(np.concatenate([[-100] * 3, [-1] * 4, self.mj_model.jnt_range[1:, 0]]))
            .float()
            .to(self.device)
        )
        self.upper_limit = (
            torch.from_numpy(np.concatenate([[100] * 3, [1] * 4, self.mj_model.jnt_range[1:, 1]]))
            .float()
            .to(self.device)
        )

        for body_id in range(self.mj_model.nbody):
            body = self.mj_model.body(body_id)
            body_name = body.name
            self.body_names[body_id] = body_name
            parent_id = int(body.parentid[0])
            self.parents[body_id] = parent_id
            self.local_position[body_id] = torch.from_numpy(np.array(body.pos, dtype=np.float32)).to(self.device)
            self.local_rotation[body_id] = torch.from_numpy(np.array(body.quat, dtype=np.float32)).to(self.device)

        for joint_id in range(self.mj_model.njnt):
            joint = self.mj_model.joint(joint_id)
            child_id = int(joint.bodyid[0])
            self.dof_ids[child_id] = int(joint.qposadr[0])
            self.joint_offsets[child_id] = torch.from_numpy(np.array(joint.pos, dtype=np.float32)).to(self.device)
            self.joint_axes[child_id] = torch.from_numpy(np.array(joint.axis, dtype=np.float32)).to(self.device)

    @property
    def nq(
        self,
    ):
        return self.mj_model.nq

    @property
    def nbodies(
        self,
    ):
        return self.mj_model.nbody

    def random_configuration(self, batch_size):
        """
        Returns a random configuration for the robot within the joint limits
        """
        samples = self.lower_limit + torch.rand(batch_size, self.nq, device=self.device) * (
            self.upper_limit - self.lower_limit
        )
        samples = torch.clip(samples, self.lower_limit, self.upper_limit)
        samples[:, 3:7] = samples[:, 3:7] / (torch.norm(samples[:, 3:7], dim=1).unsqueeze(1) + 1e-4).float()
        return samples

    def forward_kinematics(self, q: torch.Tensor, pin_notation: bool = False):
        """
        Computes the forward kinematics of the robot given a configuration in minimal coordinates q
            q tensor of shape (batch_size, nq)
        Returns:
            body_names: list of body names
            body_positions: tensor of shape (batch_size, nbodies, 3)
            body_quaternions: tensor of shape (batch_size, nbodies, 4) scalar first
        """
        if len(q.shape) == 1:
            q = q.unsqueeze(0)

        if pin_notation:
            q[:, 3:7] = q[:, [6, 3, 4, 5]]

        batch_size = q.shape[0]

        if q.shape[1] != self.nq:
            raise ValueError(f"Expected q to have shape (batch_size, {self.nq}), got {q.shape}")

        if (torch.norm(q[:, 3:7], dim=1) == 0).any():
            raise ValueError("Zero norm quaternion in forward_kinematics call")

        body_positions = torch.zeros(q.shape[0], self.nbodies, 3, device=self.device)
        body_quaternions = torch.zeros(q.shape[0], self.nbodies, 4, device=self.device)
        body_quaternions[:, :, 0] = 1

        root_id = 0
        for id, dofid in self.dof_ids.items():
            if dofid == 0:
                root_id = id

        body_positions[:, root_id, :] = q[:, :3]
        body_quaternions[:, root_id, :] = q[:, 3:7]

        body_names = []
        for body_id, parent_id in self.parents.items():
            body_names.append(self.body_names[body_id])

            parent_pos = body_positions[:, parent_id]
            parent_quat = body_quaternions[:, parent_id]

            local_quat = self.local_rotation[body_id].unsqueeze(0).repeat((batch_size, 1))
            local_pos = self.local_position[body_id].unsqueeze(0).repeat((batch_size, 1))

            pos = parent_pos + rotate(parent_quat, local_pos)
            quat = quat_multiply(parent_quat, local_quat)

            if body_id not in self.dof_ids:
                body_positions[:, body_id] = pos.clone()
                body_quaternions[:, body_id] = quat.clone()
                continue

            dof_adr = self.dof_ids[body_id]
            if dof_adr == 0:
                continue
            joint_position = self.joint_offsets[body_id].unsqueeze(0).repeat((batch_size, 1))
            joint_axes = self.joint_axes[body_id].unsqueeze(0).repeat((batch_size, 1))

            anchor = rotate(quat, joint_position) + pos

            joint_rotation = axis_angle_to_quat(joint_axes, q[:, dof_adr].unsqueeze(1))

            body_positions[:, body_id] = anchor - rotate(quat, joint_position)
            body_quaternions[:, body_id] = quat_multiply(quat, joint_rotation)

        return body_names, body_positions, body_quaternions
