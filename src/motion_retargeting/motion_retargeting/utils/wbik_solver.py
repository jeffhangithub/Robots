import os
from abc import abstractmethod
import numpy as np
import qpsolvers
import pinocchio as pin
import quaternion

from motion_retargeting.utils.velocity_estimator import VelocityEstimator
from motion_retargeting.utils.quat_utils import yaw_matrix
from motion_retargeting.utils.mujoco.renderer import MujocoRenderer
from motion_retargeting.utils.trajectory import PoseData, BodyTransform

from motion_retargeting.config.wbik_config import WBIKConfig

import pink
from pink import solve_ik
from pink.limits import ConfigurationLimit, AccelerationLimit
from pink.tasks import FrameTask, PostureTask, ComTask
from pink.barriers import PositionBarrier

# The cons of using manual limits.
# Pink would spam warnings if we violiate the limits
import logging

logging.disable("WARN")
EPS = 0.0


class NoSolutionException(Exception):
    pass

# wbik_params: 包含所有配置参数的对象，如：
# mjcf_path: 机器人模型文件路径
# step_dt: 求解步长
# task_weights: 各任务的权重（位置、旋转、关节速度等）
# contact_velocity: 判断脚部接触的速度阈值
# contact_target_lerp: 接触目标插值系数
# joint_limit_scale: 关节限制范围缩放因子
# termination_velocity: 终止求解的速度阈值
# max_iters: 最大迭代次数
# skip_iters: 首次求解的最大迭代次数
# yaw_only_feet: 是否只考虑脚部偏航角

# 速度估计器参数:
# in_shape=3: 输入维度（3D位置）
# window_size=5: 滑动窗口大小
# sample_dt: 采样时间步长
# 位置障碍参数:
# indices=[2]: 只考虑Z轴（高度）
# p_min=[0]: 最小高度（地面）
# p_max=[np.inf]: 最大高度（无限制）
# gain=[100.0]: 障碍增益
# safe_displacement_gain=1.0: 安全位移增益

# QP求解器: 优先使用quadprog，备选其他可用求解器
class WBIKSolver:
    """Wholebody inverse kinematics problem definition and solver for bipedal robots"""

    def __init__(
        self,
        wbik_params: WBIKConfig,
    ):
        """_summary_

        Args:
            wbik_params (WBIKConfig): Task configuration
            data_dt (float): Integration step of the solver

        """
        # 添加调试信息
        self.debug_info = {
            "iterations": [],
            "errors": [],
            "velocities": []
        }
        self.wbik_params = wbik_params
        self.body_to_model_map = self.wbik_params.body_to_model_map
        self.model = pin.buildModelFromMJCF(self.wbik_params.mjcf_path)
        # Mujoco and pinocchio handle root body offsets differently
        # if the floating base link has pos= attribute assigned in the MJCF file.
        # Mujoco ignores this attribute and pinocchio actually offsets the body.
        # So we preemptively set the root offset to 0.
        # 修正根关节偏移（解决MuJoCo和Pinocchio差异）
        self.model.jointPlacements[1].translation[:] = 0.0
        self.data = self.model.createData()

        self.step_dt = self.wbik_params.step_dt

        self.mjcf_to_alias_map = {v: k for k, v in self.body_to_model_map.items()}
        if len(self.mjcf_to_alias_map) != len(self.body_to_model_map):
            raise ValueError(f"Body map and contains duplicate definitions: {self.body_to_model_map}")

        # # --------------------------
        # # 新增：人体关节旋转限制（弧度）
        # # --------------------------
        # self.joint_rot_limits = {
        #     # 肘部：仅允许向内弯曲（-60°~60°）
        #     "left_elbow": (-np.pi/3, np.pi/3),
        #     "right_elbow": (-np.pi/3, np.pi/3),
        #     # 膝盖：仅允许向后弯曲（-90°~0°，防止前弯）
        #     "left_knee": (-np.pi/2, 0),
        #     "right_knee": (-np.pi/2, 0),
        #     # 肩部：限制前后摆动（-45°~45°）
        #     "left_shoulder": (-np.pi/4, np.pi/4),
        #     "right_shoulder": (-np.pi/4, np.pi/4),
        #     # 髋部：限制前后摆动（-30°~30°）
        #     "left_hip": (-np.pi/6, np.pi/6),
        #     "right_hip": (-np.pi/6, np.pi/6)
        # }
        
        # # 为每个关节添加旋转约束任务（提高惩罚权重，强制遵守限制）
        # for joint_name, (lower, upper) in self.joint_rot_limits.items():
        #     self.tasks[joint_name] = PostureTask(
        #         joint_name,
        #         lower=lower,
        #         upper=upper,
        #         cost=100.0  # 权重越高，约束越强
        #     )


        # 初始化身体部位到Pinocchio框架ID的映射
        self.body_to_pin_id = {}
        self.available_frames = []
        # 遍历所有框架，建立映射关系
        for i in range(self.model.nframes):
            frame_name = self.model.frames[i].name
            self.available_frames.append(frame_name)
            if frame_name in self.mjcf_to_alias_map.keys():
                self.body_to_pin_id[self.mjcf_to_alias_map[frame_name]] = i

        # 验证所有配置的身体部位都在模型中
        for alias, name in self.body_to_model_map.items():
            if alias not in self.body_to_pin_id.keys():
                raise ValueError(
                    f"Frame {alias}:{name} not found in the robot model{os.linesep}\
                                 Available frames: {self.available_frames}"
                )

        # 获取关节名称顺序
        self.joint_order = list(self.model.names)

        # 为左右脚创建速度估计器
        self.foot_vel_estimators = [
            VelocityEstimator(in_shape=3, window_size=5, sample_dt=self.wbik_params.step_dt) for i in range(2)
        ]

        # 初始化NIK（逆运动学）求解器
        self.__setup_nik()

    def reset(self):
        """
        Method to reset the solution to the initial state
        Needs to be called at the start of every consecutive motion clip
        """
        # 重置左右脚速度估计器
        for i in range(2):
            self.foot_vel_estimators[i].reset()

        # 重置接触状态（初始为未接触）
        self.contacts = [False, False]
        # 标记为首次求解
        self.first_solution = True

    def set_target_transform(
        self,
        body_name: str,
        position: np.ndarray = None,
        rotation_matrix: np.ndarray = None,
    ):
        """Sets the desried frame task position and translation in the world frame


        Args:
            body_name (str): body frame in the body_to_model_map dict
            position (np.ndarray, optional): desired frame position in world frame
            rotation_matrix (np.ndarray, optional):  desired frame rotation in world frame
        """
        # 处理位置参数
        if position is None:
            position = np.zeros(3)
        # 处理旋转参数
        if rotation_matrix is None:
            rotation_matrix = np.eye(3)
        else:
            # 如果是脚部且配置为只考虑偏航角
            if "foot" in body_name and self.wbik_params.yaw_only_feet:
                rotation_matrix = yaw_matrix(rotation_matrix)

        # 创建SE3变换对象并存储
        self.targets[body_name] = pin.SE3(rotation_matrix, position)

    @abstractmethod
    def set_targets(
        self,
    ):
        pass

    def __setup_nik(self):
        # 初始化任务字典
        self.tasks = {}
        # 初始化目标字典
        self.targets = {}
        # 初始化前一帧目标字典
        self.prev_targets = {}

        # 初始化关节位置（浮动基座）
        q_init = np.zeros(self.model.nq)
        q_init[6] = 1 # 四元数的w分量设为1（单位四元数）
        # 创建Pink配置对象
        self.configuration = pink.Configuration(self.model, self.data, q_init)

        # 选择QP求解器（优先使用quadprog）
        self.solver = qpsolvers.available_solvers[0]
        if "quadprog" in qpsolvers.available_solvers:
            self.solver = "quadprog"

        # 重置求解器状态
        self.reset()

        # 为每个身体部位创建框架任务
        # Create a frame task for each frame in the mapping
        for body_name in self.body_to_model_map.keys():
            subname = "root" if body_name == "root" else body_name.split("_")[1]
            # 获取位置和旋转权重
            pos_weight = self.wbik_params.task_weights["position"].get(subname, EPS)
            rot_weight = self.wbik_params.task_weights["rotation"].get(subname, EPS)
            # 如果权重都太小则跳过
            if pos_weight == EPS and rot_weight == EPS:
                continue

            # 创建框架任务
            self.tasks[body_name] = FrameTask(
                self.body_to_model_map[body_name],
                position_cost=pos_weight,
                orientation_cost=rot_weight,
            )
        # 添加关节位置任务（最小化关节速度）
        # Add joint velocity task
        self.joint_pos_task = PostureTask(cost=self.wbik_params.task_weights["joint_velocity"])

        # 添加位置障碍（防止脚穿透地面）
        # Add barriers and limits
        self.barriers = []
        ## Prevent the feet from penetrating the ground
        for side in ["left", "right"]:
            self.barriers.append(
                PositionBarrier(
                    self.body_to_model_map[f"{side}_foot"], # 脚部框架名
                    indices=[2],  # Z axis # Z轴（高度方向）
                    p_min=np.array([0]), # 最小高度（地面）
                    p_max=np.array([np.inf]), # 最大高度（无限制）
                    gain=np.array([100.0]), # 障碍增益
                    safe_displacement_gain=1.0,  # 安全位移增益
                )
            )
        # 调整关节位置限制范围
        half_limit = (
            np.abs(self.model.lowerPositionLimit - self.model.upperPositionLimit)
            * (1 - self.wbik_params.joint_limit_scale)
            * 0.5
        )
        self.model.lowerPositionLimit += half_limit
        self.model.upperPositionLimit -= half_limit
        # 创建关节位置限制对象
        self.dof_limits = ConfigurationLimit(self.model, config_limit_gain=1.0)
        # 设置加速度限制
        acceleration_limit = np.full(self.model.nv, 0.0)
        # 根关节线加速度限制
        acceleration_limit[:3] = self.wbik_params.task_weights["max_root_lin_acceleration"]
        # 根关节角加速度限制
        acceleration_limit[3:6] = self.wbik_params.task_weights["max_root_ang_acceleration"]
        # 关节加速度限制
        acceleration_limit[6:] = self.wbik_params.task_weights["max_joint_acceleration"]

        # 创建加速度限制对象
        self.acceleration_limit = AccelerationLimit(self.model, acceleration_limit)

    def solve(
        self,
        renderer=None,
    ) -> PoseData:
        """
        Solves IK using the desired positions of the limbs
        specified in the set_targets method

        Returns:
            tuple(np.ndarray):
                -position of the root link in the world frame
                -world frame root link quaternion
                -joint positions
                -sum of translation task errors
        """
        # 设置目标位姿（由子类实现）
        self.set_targets()
        
        # 调试：打印目标位置
        # print("目标位置:")
        # for name, target in self.targets.items():
        #     print(f"  {name}: pos={target.translation}, rot={target.rotation}")
        # print("body:")            
        # for frame_name in self.body_to_pin_id:
        #     if frame_name in self.targets.keys():
        #       print(f"  {frame_name}: pos={self.body(frame_name).translation}, rot={self.body(frame_name).rotation}")   

        # 如果是首次求解，初始化配置
        if self.first_solution:
            # Init configuration at target root pos
            # 使用目标根位置初始化
            q_init = np.zeros(self.model.nq)
            q_init[:3] = self.targets["root"].translation
            q_init[3:7] = quaternion.as_float_array(quaternion.from_rotation_matrix(self.targets["root"].rotation))[
                [1, 2, 3, 0]   # 转换为[x, y, z, w]顺序
            ]
            self.prev_solution = q_init.copy()
            self.configuration = pink.Configuration(self.model, self.data, q_init)

        # 处理目标（包括接触状态调整）
        self.__process_targets()

        # 设置关节位置任务目标
        self.joint_pos_task.set_target(self.prev_solution)
        # 设置各框架任务目标
        for name in self.targets.keys():
            self.tasks[name].set_target(self.targets[name])

        # 初始化速度范数和迭代计数
        velocity_norm = np.inf
        iter_count = 0
        # 迭代求解直到速度低于阈值或达到最大迭代次数
        while velocity_norm > self.wbik_params.termination_velocity:
            # 检查迭代限制
            if not self.first_solution:
                if iter_count >= self.wbik_params.max_iters:
                    break
            else:
                if iter_count >= self.wbik_params.skip_iters:
                    raise NoSolutionException("Too many iterations")

            # 收集任务
            tasks = list(self.tasks.values())
            limits = [self.dof_limits]
            # 非首次求解时添加关节位置任务和加速度限制
            if not self.first_solution:
                tasks += [self.joint_pos_task]
                limits += [self.acceleration_limit]

            # 求解IK
            velocity = solve_ik(
                self.configuration,         # 当前配置
                tasks,                      # 任务列表
                self.wbik_params.step_dt,   # 时间步长
                solver=self.solver,         # QP求解器
                barriers=self.barriers,     # 位置障碍
                safety_break=False,         # 禁用安全中断
                limits=limits,              # 关节限制
            )
            # 应用求解得到的速度
            self.configuration.integrate_inplace(velocity, self.wbik_params.step_dt)
            # 计算速度范数（排除浮动基座）
            velocity_norm = np.linalg.norm(velocity[6:])
            # 如果提供了渲染器，渲染当前状态
            if renderer is not None:
                self.render_solution(renderer)
                renderer.step()
            # 收集调试信息
            self.debug_info["iterations"].append(iter_count)
            self.debug_info["errors"].append(self.compute_position_errors())
            self.debug_info["velocities"].append(velocity_norm)

            iter_count += 1
        # 保存当前解，标记首次求解完成
        self.prev_solution = self.configuration.q.copy()
        self.first_solution = False
        
        # 求解后打印调试信息
        # print(f"求解完成: {iter_count}次迭代, 最终速度: {velocity_norm:.6f}")
        # print("位置误差:", self.compute_position_errors())
        # 返回姿态数据
        return self.build_pose_data()

    def adjust_feet_contacts(self):
        # If the feet are in contact the target transform is fixed
        # otherwise previous solution is interpolated to the desired position
        # 初始化接触状态
        self.contacts = [False, False]
        # 处理左右脚
        for i, side in enumerate(["left", "right"]):
            # Estimate foot velocity from input data
            # 估计脚部速度
            foot_velocity = self.foot_vel_estimators[i](self.targets[f"{side}_foot"].translation)
            velocity_norm = np.linalg.norm(foot_velocity)
            # 根据速度判断是否接触地面
            self.contacts[i] = velocity_norm < self.wbik_params.contact_velocity
            # 如果接触地面
            if self.contacts[i]:
                # Keeping the yaw angle so that the feet won't rotate into the ground
                # when in contact
                # 使用偏航角矩阵（保持脚部朝向但防止旋转）
                target_matrix = yaw_matrix(self.prev_targets[f"{side}_foot"].rotation)
                self.targets[f"{side}_foot"].translation = self.prev_targets[f"{side}_foot"].translation
            else:
                # 使用完整旋转矩阵
                target_matrix = self.targets[f"{side}_foot"].rotation

            # 旋转插值（平滑过渡）
            current_target = quaternion.from_rotation_matrix(self.prev_targets[f"{side}_foot"].rotation)
            prev_target = quaternion.from_rotation_matrix(target_matrix)

            self.targets[f"{side}_foot"].rotation = quaternion.as_rotation_matrix(
                quaternion.slerp(current_target, prev_target, 0, 1, self.wbik_params.contact_target_lerp)
            )

    # 处理目标：保存前一帧目标，调整脚部接触状态，更新前一帧目标
    def __process_targets(self):
        """
        Process the targets and set the desired positions and orientations
        """
        if self.first_solution:
            self.prev_targets = {k: v.copy() for k, v in self.targets.items()}

        self.adjust_feet_contacts()

        self.prev_targets = {k: v.copy() for k, v in self.targets.items()}

    # 获取指定身体部位在当前配置下的位姿
    def body(self, body_name):
        """
        Returns the transform of the body in the world frame
        """
        if body_name not in self.body_to_pin_id.keys() and body_name:
            raise KeyError(
                f"""Invalid frame name: {body_name}{os.linesep}\
                    Avalable frames:{os.linesep}\
                    {list(self.body_to_pin_id.keys())}"""
            )
        return self.configuration.data.oMf[self.body_to_pin_id[body_name]]

    # 计算各身体部位当前位置与目标位置的欧氏距离
    def compute_position_errors(
        self,
    ):
        """
        Computes the position errors relative to the targets
        """
        errors = {}
        for frame_name in self.body_to_pin_id:
            if frame_name in self.targets.keys():
                errors[frame_name] = np.linalg.norm(
                    self.body(frame_name).translation - self.targets[frame_name].translation
                )
        return errors

    # 构建包含所有信息的姿态数据对象
    def build_pose_data(
        self,
    ) -> PoseData:
        """
        Builds dictionary with the frame data including positions and quaternions
        of all tracked frames
        """
        # 收集所有跟踪的身体部位位姿
        transforms = []
        added_bodies = []
        for frame_name in self.body_to_pin_id:
            mjcf_name = self.body_to_model_map[frame_name]
            added_bodies.append(mjcf_name)
            body_transform = self.body(frame_name)
            transforms.append(
                BodyTransform(
                    name=mjcf_name,
                    position=body_transform.translation.copy(),
                    quaternion=quaternion.from_rotation_matrix(body_transform.rotation.copy()),
                )
            )

        # 添加根关节和额外身体部位
        root_name = self.model.frames[list(self.model.parents).index(1)].name
        for i, frame in enumerate(self.model.frames):
            is_root_frame = frame.name == root_name and frame.name not in added_bodies
            is_extra_body = frame.name in self.wbik_params.extra_bodies and frame.name not in added_bodies
            if is_root_frame or is_extra_body:
                body_transform = self.configuration.data.oMf[i]
                transforms.append(
                    BodyTransform(
                        name=frame.name,
                        position=body_transform.translation.copy(),
                        quaternion=quaternion.from_rotation_matrix(body_transform.rotation.copy()),
                    )
                )

        # 构建姿态数据对象
        pose_data = PoseData(
            transforms=transforms,
            q=self.prev_solution.copy(),
            body_aliases=self.body_to_model_map,
            model_root=root_name,
            contacts=np.array(self.contacts),
            position_error=sum(self.compute_position_errors().values()),
            dt=self.wbik_params.step_dt,
            joint_order=self.joint_order,
        )
        return pose_data

    # 可视化当前求解结果：机器人状态、目标位置和坐标系
    def render_solution(self, renderer: MujocoRenderer):
      # 设置机器人配置
        renderer.set_configuration(self.prev_solution.copy(), pin_notation=True)
        # Render desired and target link positions and orientations
        # 渲染当前关节位置
        frame_color = np.array([1, 0, 0, 1])
        for frame_name in self.body_to_pin_id.keys():
            renderer.render_points(
                positions=[self.body(frame_name).translation],
                size=0.03,
                colors=[frame_color],
            )

        # 渲染目标位置
        # Render cached frame targets
        for name, target in self.targets.items():
            if np.any(self.tasks[name].cost[:3] > EPS):
                marker_color = np.array([0, 1, 0, 1])
                if "foot" in name:
                    foot_idx = 0
                    if "right" in name:
                        foot_idx = 1
                    if self.contacts[foot_idx]:
                        marker_color = np.array([0, 0, 0, 1])

                renderer.render_points(
                    positions=[target.translation],
                    size=0.03,
                    colors=[marker_color],
                )

            # 渲染坐标系
            if np.any(self.tasks[name].cost[3:] > EPS):
                # Draw unit axes
                if np.linalg.norm(self.tasks[name].cost[3:6]) != 0:
                    renderer.render_frames(
                        positions=[target.translation],
                        rotations=[target.rotation],
                        axis_length=0.3,
                        axis_radius=0.01,
                    )
