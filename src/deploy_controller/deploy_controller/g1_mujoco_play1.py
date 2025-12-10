#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from robot_descriptions.g1_mj_description import MJCF_PATH
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import numpy as np
import threading
import time
from mujoco import viewer
import mujoco

class G1MujocoController(Node):
    """G1机器人在MuJoCo中的仿真控制器"""
    
    def __init__(self):
        super().__init__('g1_mujoco_controller')
        
        # 参数声明
        self.declare_parameter('model_path', '/home/uneedrobot/workshops/source/xsens/ros2/src/motion_retargeting/motion_retargeting/config/robot/models/g1_edited.xml')
        # self.declare_parameter('model_path', MJCF_PATH)
        self.declare_parameter('control_rate', 100.0)  # 控制频率
        self.declare_parameter('visualization', True)  # 是否开启可视化
        
        self.model_path = self.get_parameter('model_path').value
        self.control_rate = self.get_parameter('control_rate').value
        self.visualization_enabled = self.get_parameter('visualization').value
        
        # MuJoCo模型和数据
        self.model = None
        self.data = None
        self.viewer = None
        self.viewer_thread = None
        
        # 控制状态
        self.is_initialized = False
        self.latest_joint_state = None
        self.current_target_positions = np.zeros(29)  # 29个自由度
        self.current_positions = np.zeros(29)
        
        # 控制器增益
        self.Kp = np.array([
            150, 150, 150, 200, 100, 100,      # 腿部 - 增加刚度
            150, 150, 150, 200, 100, 100,
            100, 80, 80,                       # 腰部
            80, 80, 80, 80, 80, 40, 40,        # 手臂
            80, 80, 80, 80, 80, 40, 40
        ])
      
        self.Kd = np.array([
            10, 10, 10, 15, 8, 8,             # 增加阻尼
            10, 10, 10, 15, 8, 8,
            8, 6, 6,
            6, 6, 6, 6, 6, 3, 3,
            6, 6, 6, 6, 6, 3, 3
        ])
        
        # 添加低通滤波器
        self.filter_alpha = 0.3  # 滤波系数
        self.filtered_positions = None

        # 关节索引映射
        self.joint_mapping = {
            'left_hip_pitch_joint': 0,
            'left_hip_roll_joint': 1,
            'left_hip_yaw_joint': 2,
            'left_knee_joint': 3,
            'left_ankle_pitch_joint': 4,
            'left_ankle_roll_joint': 5,
            'right_hip_pitch_joint': 6,
            'right_hip_roll_joint': 7,
            'right_hip_yaw_joint': 8,
            'right_knee_joint': 9,
            'right_ankle_pitch_joint': 10,
            'right_ankle_roll_joint': 11,
            'waist_yaw_joint': 12,
            'waist_roll_joint': 13,
            'waist_pitch_joint': 14,
            'left_shoulder_pitch_joint': 15,
            'left_shoulder_roll_joint': 16,
            'left_shoulder_yaw_joint': 17,
            'left_elbow_joint': 18,
            'left_wrist_roll_joint': 19,
            'left_wrist_pitch_joint': 20,
            'left_wrist_yaw_joint': 21,
            'right_shoulder_pitch_joint': 22,
            'right_shoulder_roll_joint': 23,
            'right_shoulder_yaw_joint': 24,
            'right_elbow_joint': 25,
            'right_wrist_roll_joint': 26,
            'right_wrist_pitch_joint': 27,
            'right_wrist_yaw_joint': 28
        }
        
        # 初始化MuJoCo
        self.init_mujoco()

        # 订阅重定向后的关节状态
        qos_profile = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )

        # 订阅重定向后的关节状态
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/retargeted_joint_states',
            self.joint_state_callback,
            qos_profile
        )
        
        # 发布仿真关节状态
        self.joint_state_pub = self.create_publisher(
            JointState,
            '/g1_sim/joint_states',
            qos_profile
        )
        
        # 创建控制定时器
        self.control_timer = self.create_timer(1.0/self.control_rate, self.control_loop)
        
        self.get_logger().info("G1 MuJoCo仿真控制器已初始化")

    def init_mujoco(self):
        """初始化MuJoCo仿真环境"""
        try:
            # 从文件加载
            self.model = mujoco.MjModel.from_xml_path(self.model_path)
            self.data = mujoco.MjData(self.model)
            
            # 初始化到站立姿势
            self.reset_to_stand_pose()

            self.stabilize_before_control()
            
            # 如果启用可视化，启动查看器
            if self.visualization_enabled:
                self.start_viewer()
                
            self.is_initialized = True
            self.get_logger().info("MuJoCo仿真环境初始化成功")
            
        except Exception as e:
            self.get_logger().error(f"MuJoCo初始化失败: {e}")
            raise

    def reset_to_stand_pose(self):
        """重置到站立姿势"""
        # 设置站立关键帧
        stand_qpos = [
            0, 0, 0.79, 0, 0, 0, 1,  # 浮基座: [x, y, z, qw, qx, qy, qz]
            0, 0, 0, 0, 0, 0,      # 腿部关节
            0, 0, 0, 0, 0, 0,      # 腿部关节
            0, 0, 0,                   # 腰部关节
            0.2, 0.2, 0, 1.28, 0, 0, 0,  # 左臂关节
            0.2, -0.2, 0, 1.28, 0, 0, 0   # 右臂关节
        ]
        
        # 设置位置
        self.data.qpos[:] = stand_qpos[:self.model.nq]
        self.data.qvel[:] = 0

        mujoco.mj_forward(self.model, self.data)


    def stabilize_before_control(self):
        """控制前稳定化步骤"""
        self.get_logger().info("执行控制前稳定化...")
        
        # 步骤1：锁定当前关节位置
        self.current_target_positions = self.data.qpos[7:7+29].copy()
        self.current_positions = self.data.qpos[7:7+29].copy()
        
        # 步骤2：运行100步仿真让系统稳定
        for i in range(500):
            # 应用小增益PD控制保持位置
            current_pos = self.data.qpos[7:7+29]
            position_error = self.current_target_positions - current_pos
                        
            # 计算简单控制
            control = self.Kp * position_error
            # 添加重力补偿
            gravity_comp = self.calculate_gravity_compensation()
            # 限制力矩
            control = control + gravity_comp

            # 限制力矩，逐步增加
            max_lim = min(20, 5 + i * 0.03)  # 逐步增加限制
            control = np.clip(control, -max_lim, max_lim)
            
            # 应用到关节
            for j in range(min(29, self.model.nu)):
                self.data.ctrl[j] = control[j]
            
            # 步进仿真
            mujoco.mj_step(self.model, self.data)
            
            # 稍微更新查看器
            if self.viewer is not None and i % 10 == 0:
                with threading.Lock():
                    self.viewer.sync()
        
        self.get_logger().info("稳定化完成")

    def start_viewer(self):
        """启动MuJoCo查看器"""
        def viewer_thread():
            try:
                self.viewer = viewer.launch_passive(self.model, self.data)
                self.get_logger().info("MuJoCo查看器已启动")
                
                # 查看器主循环
                while self.viewer.is_running():
                    with threading.Lock():
                        self.viewer.sync()
                    time.sleep(0.01)
                    
            except Exception as e:
                self.get_logger().error(f"查看器错误: {e}")
        
        self.viewer_thread = threading.Thread(target=viewer_thread)
        self.viewer_thread.daemon = True
        self.viewer_thread.start()

    def joint_state_callback(self, msg):
        """重定向关节状态回调"""
        try:
            # 解析重定向后的关节状态
            target_positions = self.parse_retargeted_joint_state(msg)
            
            # 更新目标位置
            self.current_target_positions = target_positions
            self.latest_joint_state = msg

            # self.get_logger().info(f"接收到目标位置: {target_positions}...")
            
            
            self.get_logger().debug(f"接收到目标位置: {target_positions[:5]}...")
            
        except Exception as e:
            self.get_logger().error(f"处理关节状态回调时出错: {e}")

    def parse_retargeted_joint_state(self, msg):
        """
        解析重定向后的关节状态消息
        """
        target_positions = np.zeros(29)
        temp_dof_pos = {}
        temp_local_body_pos = {}
        temp_pos = np.zeros(3)


        fps = msg.position[0]
        # 保存根节点信息用于控制浮基座
        self.root_target_pos = np.array(msg.position[1:4])
        self.root_target_rot = np.array(msg.position[4:8])

        try:
            for i, name in enumerate(msg.name):
                if name.endswith("_dof_pos"):
                    index = msg.name.index(name)
                    temp_dof_pos[name[:-8]] = msg.position[index]
                elif name[:-2].endswith("_local_body_pos"):
                    index = msg.name.index(name)
                    if name.endswith("_x"):
                        temp_pos[0] = msg.position[index]
                    elif name.endswith("_y"):
                        temp_pos[1] = msg.position[index]
                    elif name.endswith("_z"):
                        temp_pos[2] = msg.position[index]             
                    temp_local_body_pos[name[:-17]] = np.array(temp_pos)

            for key in temp_dof_pos:
                if key == "pelvis":
                    continue
                elif key == "torso_link":
                    index = self.joint_mapping["waist_pitch_joint"]
                    target_positions[index] = self.apply_coordinate_transform_to_joint("waist_pitch_joint", temp_dof_pos[key])
                elif (key[:-5]+"_joint") in self.joint_mapping:
                    index = self.joint_mapping[(key[:-5]+"_joint")]
                    target_positions[index] = self.apply_coordinate_transform_to_joint((key[:-5]+"_joint"), temp_dof_pos[key])

            # 应用关节限制
            target_positions = self.apply_joint_limits(target_positions)
            
        except Exception as e:
            self.get_logger().error(f"解析关节状态时出错: {e}")
        
        return target_positions

    def get_joint_name_by_index(self, index):
        """根据索引获取关节名称"""
        for name, idx in self.joint_mapping.items():
            if idx == index:
                return name
        return f"joint_{index}"

    def apply_coordinate_transform_to_joint(self, joint_name, angle):
        """
        应用坐标系转换到关节角度
        根据关节类型和方向调整角度符号
        """
        # 对于不同的关节类型，可能需要不同的符号调整
        # 这里是一个通用的转换，可能需要根据具体模型调整
        
        # 腿部关节通常需要符号调整
        if 'hip' in joint_name and 'roll' in joint_name:
            # 髋关节滚转需要调整符号
            return -angle
        elif 'ankle' in joint_name and 'roll' in joint_name:
            # 踝关节滚转需要调整符号
            return -angle
        elif 'shoulder' in joint_name and 'roll' in joint_name:
            # 肩关节滚转需要调整符号
            return -angle
        elif 'wrist' in joint_name and 'roll' in joint_name:
            # 腕关节滚转需要调整符号
            return -angle
        elif 'yaw' in joint_name:
            # 偏航关节可能需要调整符号
            return -angle
        else:
            # 其他关节保持原样
            return angle

    def apply_joint_limits(self, positions):
        """应用关节限制"""
        joint_limits = {
            # 左腿
            0: (-2.5307, 2.8798),   # left_hip_pitch
            1: (-0.5236, 2.9671),    # left_hip_roll
            2: (-2.7576, 2.7576),    # left_hip_yaw
            3: (-0.087267, 2.8798),  # left_knee
            4: (-0.87267, 0.5236),   # left_ankle_pitch
            5: (-0.2618, 0.2618),    # left_ankle_roll
            
            # 右腿
            6: (-2.5307, 2.8798),    # right_hip_pitch
            7: (-2.9671, 0.5236),    # right_hip_roll
            8: (-2.7576, 2.7576),    # right_hip_yaw
            9: (-0.087267, 2.8798),   # right_knee
            10: (-0.87267, 0.5236),  # right_ankle_pitch
            11: (-0.2618, 0.2618),   # right_ankle_roll
            
            # 腰部
            12: (-2.618, 2.618),     # waist_yaw
            13: (-0.52, 0.52),       # waist_roll
            14: (-0.52, 0.52),       # waist_pitch
            
            # 左臂
            15: (-3.0892, 2.6704),   # left_shoulder_pitch
            16: (-1.5882, 2.2515),   # left_shoulder_roll
            17: (-2.618, 2.618),     # left_shoulder_yaw
            18: (-1.0472, 2.0944),   # left_elbow
            19: (-1.97222, 1.97222), # left_wrist_roll
            20: (-1.61443, 1.61443), # left_wrist_pitch
            21: (-1.61443, 1.61443), # left_wrist_yaw
            
            # 右臂
            22: (-3.0892, 2.6704),   # right_shoulder_pitch
            23: (-2.2515, 1.5882),   # right_shoulder_roll
            24: (-2.618, 2.618),     # right_shoulder_yaw
            25: (-1.0472, 2.0944),   # right_elbow
            26: (-1.97222, 1.97222), # right_wrist_roll
            27: (-1.61443, 1.61443), # right_wrist_pitch
            28: (-1.61443, 1.61443)  # right_wrist_yaw
        }
        
        for i, (min_limit, max_limit) in joint_limits.items():
            positions[i] = np.clip(positions[i], min_limit, max_limit)
        return positions

    def control_loop(self):
        """控制循环"""
        if not self.is_initialized or self.data is None:
            return
            
        try:
            # 获取当前关节位置
            self.get_current_joint_positions()

            # if hasattr(self, 'root_target_pos') and hasattr(self, 'root_target_rot'):
            #     self.control_floating_base()
            
            # 计算PD控制
            self.calculate_pd_control()
            
            # 步进仿真
            self.step_simulation()
            
            # 发布当前关节状态
            self.publish_joint_states()
            
        except Exception as e:
            self.get_logger().error(f"控制循环错误: {e}")

    def control_floating_base(self):
        """控制浮基座"""
        if self.model.nq >= 7:
            # 设置浮基座位置和姿态
            # 位置控制
            position_error = self.root_target_pos - self.data.qpos[0:3]
            self.data.qvel[0:3] = position_error * 10.0  # 简单P控制
            
            # 姿态控制（四元数）
            target_quat = self.root_target_rot
            current_quat = self.data.qpos[3:7]
            
            # 转换为旋转矩阵，计算角速度
            # 这里简化处理，实际应该使用四元数误差计算
            self.data.qvel[3:6] = np.zeros(3)  # 暂时设为0

    def calculate_gravity_compensation(self):
        """计算重力补偿力矩"""
        gravity_compensation = np.zeros(29)
        
        # 为支撑关节添加基础重力补偿
        # 腿部关节需要更大的重力补偿
        gravity_compensation[0:6] = [5, 10, 5, 15, 3, 3]  # 左腿
        gravity_compensation[6:12] = [5, 10, 5, 15, 3, 3]  # 右腿
        
        return gravity_compensation

    def get_current_joint_positions(self):
        """获取当前关节位置（修正版）"""
        # 浮基座有7个自由度：pos(3) + quat(4)
        if self.model.nq > 7:
            # 确保不超过实际关节数量
            n_joints = min(29, self.model.nq - 7)
            self.current_positions = self.data.qpos[7:7+n_joints]
            
            # 如果实际关节数少于29，用0填充
            if n_joints < 29:
                self.current_positions = np.pad(self.current_positions, 
                                            (0, 29 - n_joints), 
                                            'constant')

    def apply_control_torque(self, torque):
        """应用控制力矩到仿真（修正版）"""
        # 浮基座没有执行器，从第0个执行器开始对应第一个关节
        for i in range(min(29, self.model.nu)):
            self.data.ctrl[i] = torque[i]

    def calculate_pd_control(self):
        """计算PD控制输出（修正版）"""
        # 确保数组长度一致
        # 获取目标位置
        target_pos = self.current_target_positions[:len(self.current_positions)]
        
        # 应用低通滤波器
        if self.filtered_positions is None:
            self.filtered_positions = target_pos.copy()
        else:
            self.filtered_positions = (self.filter_alpha * target_pos + 
                                    (1 - self.filter_alpha) * self.filtered_positions)
        
        # 使用滤波后的位置
        filtered_target = self.filtered_positions[:len(self.current_positions)]
        # target_pos = self.current_positions[:len(self.current_positions)]
        current_pos = self.current_positions[:len(target_pos)]

        self.get_logger().info(f"当前姿态信息: {current_pos}")
        self.get_logger().info(f"目标姿态信息: {target_pos}")
        
        # 计算位置误差
        position_error = filtered_target - current_pos
        
        # 计算速度
        if hasattr(self, 'last_positions'):
            dt = 1.0 / self.control_rate
            velocity = (current_pos - self.last_positions[:len(current_pos)]) / dt
        else:
            velocity = np.zeros_like(current_pos)

        self.get_logger().info(f"速度信息: {velocity}")
        
        # 使用对应的Kp和Kd增益
        kp = self.Kp[:len(position_error)]
        kd = self.Kd[:len(velocity)]
        
        # 计算控制力矩
        control_torque = kp * position_error - kd * velocity

        gravity_compensation = self.calculate_gravity_compensation()
        
        # 合并控制力矩
        total_torque = control_torque + gravity_compensation[:len(control_torque)]

        # 应用力矩限制
        max_torque = np.array([
            88, 139, 88, 139, 50, 50,     # 左腿
            88, 139, 88, 139, 50, 50,     # 右腿
            88, 50, 50,                   # 腰部
            25, 25, 25, 25, 25, 5, 5,     # 左臂
            25, 25, 25, 25, 25, 5, 5      # 右臂
        ])[:len(control_torque)]
        
        control_torque = np.clip(total_torque, -max_torque, max_torque)
        
        # 应用到仿真
        self.apply_control_torque(control_torque)
        
        # 保存当前位置用于下次计算速度
        self.last_positions = self.current_positions.copy()

    def step_simulation(self):
        """步进仿真"""
        # 步进仿真
        mujoco.mj_step(self.model, self.data)
        
        # 更新查看器
        if self.viewer is not None and self.viewer.is_running():
            with threading.Lock():
                self.viewer.sync()

    def publish_joint_states(self):
        """发布关节状态"""
        joint_state_msg = JointState()
        joint_state_msg.header.stamp = self.get_clock().now().to_msg()
        
        # 添加所有关节名称和位置
        for name, idx in self.joint_mapping.items():
            joint_state_msg.name.append(name)
            if idx < len(self.current_positions):
                joint_state_msg.position.append(float(self.current_positions[idx]))
            else:
                joint_state_msg.position.append(0.0)
        
        self.joint_state_pub.publish(joint_state_msg)

    def destroy_node(self):
        """节点销毁时的清理工作"""
        self.get_logger().info("正在关闭G1 MuJoCo仿真控制器...")
        
        # 关闭查看器
        if self.viewer is not None:
            self.viewer.close()
        
        super().destroy_node()
        self.get_logger().info("G1 MuJoCo仿真控制器已关闭")

def main(args=None):
    rclpy.init(args=args)
    
    try:
        controller = G1MujocoController()
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info("接收到键盘中断信号")
    except Exception as e:
        print(f"G1 MuJoCo仿真控制器错误: {e}")
    finally:
        if 'controller' in locals():
            controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()