#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import sys
import time
import numpy as np
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from xsens_mvn_ros_msgs.msg import LinkStateArray, LinkState

# 导入Unitree SDK
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

# G1机器人配置
G1_NUM_MOTOR = 29

# 控制器增益
Kp = [
    60, 60, 60, 100, 40, 40,      # legs
    60, 60, 60, 100, 40, 40,      # legs
    60, 40, 40,                   # waist
    40, 40, 40, 40,  40, 40, 40,  # arms
    40, 40, 40, 40,  40, 40, 40   # arms
]

Kd = [
    1, 1, 1, 2, 1, 1,     # legs
    1, 1, 1, 2, 1, 1,     # legs
    1, 1, 1,              # waist
    1, 1, 1, 1, 1, 1, 1,  # arms
    1, 1, 1, 1, 1, 1, 1   # arms 
]

class Mode:
    PR = 0  # Series Control for Pitch/Roll Joints
    AB = 1  # Parallel Control for A/B Joints

class G1MotionController(Node):
    """G1机器人运动控制器"""
    
    def __init__(self):
        super().__init__('g1_motion_controller')
        
        # 参数声明
        self.declare_parameter('networkInterface', 'enxc8787dbe41ad')
        self.declare_parameter('control_rate', 500.0)  # 控制频率
        self.declare_parameter('safety_timeout', 2.0)  # 安全超时
        
        self.networkInterface = self.get_parameter('networkInterface').value
        self.control_rate = self.get_parameter('control_rate').value
        self.safety_timeout = self.get_parameter('safety_timeout').value
        
        # 控制状态        
        self.is_initialized = False
        self.latest_joint_state = None
        self.last_joint_state_time = None
        self.safety_counter = 0
        self.current_dof_positions = np.zeros(G1_NUM_MOTOR)  # 存储当前自由度位置

        # 初始化Unitree SDK
        self.init_unitree_sdk()
        
        # 订阅重定向后的关节状态
        qos_profile = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )

        self.joint_state_sub = self.create_subscription(
            JointState,
            '/retargeted_joint_states',
            self.joint_state_callback,
            qos_profile
        )

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
     
        # 创建控制定时器
        self.control_timer = self.create_timer(1.0/self.control_rate, self.control_loop)
        
        self.get_logger().info("G1运动控制器已初始化")

    def init_unitree_sdk(self):
        """初始化Unitree SDK"""
        try:
            # 初始化通道
            ChannelFactoryInitialize(0, self.networkInterface)
            
            # 创建运动切换客户端
            self.motion_switcher = MotionSwitcherClient()
            self.motion_switcher.SetTimeout(5.0)
            self.motion_switcher.Init()
            
            # 检查并释放现有模式
            status, result = self.motion_switcher.CheckMode()
            while result['name']:
                self.motion_switcher.ReleaseMode()
                status, result = self.motion_switcher.CheckMode()
                time.sleep(1)
            
            # 创建命令发布器
            self.lowcmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
            self.lowcmd_publisher.Init()

            # 初始化低级命令
            self.low_cmd = unitree_hg_msg_dds__LowCmd_()
            self.low_state = None
            self.mode_machine = 0
            self.update_mode_machine = False

            # 创建状态订阅器
            self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
            self.lowstate_subscriber.Init(self.lowstate_callback, 10)
            
            # CRC校验
            self.crc = CRC()
            
            self.get_logger().info("Unitree SDK初始化成功")
            
        except Exception as e:
            self.get_logger().error(f"Unitree SDK初始化失败: {e}")
            raise

    def lowstate_callback(self, msg: LowState_):
        """低级状态回调"""
        self.low_state = msg
        
        if not self.update_mode_machine:
            self.mode_machine = msg.mode_machine
            self.update_mode_machine = True
            self.is_initialized = True
            self.get_logger().info("G1机器人状态已连接")

    def parse_retargeted_joint_state(self, msg):
        """
        解析重定向后的关节状态消息
        
        根据data_subscriber.py发布的消息格式解析：
        - 第一个数据是fps
        - 然后是自由度位置(dof_pos)
        - 然后是根位置和旋转
        
        Returns:
            np.array: G1机器人29个自由度的目标位置
        """
        dof_positions = np.zeros(G1_NUM_MOTOR)
        
        temp_dof_pos = {}
        temp_local_body_pos = {}
        temp_pos = np.zeros(3)


        fps = msg.position[0]
        root_pos = np.array(msg.position[1:4])
        root_rot = np.array(msg.position[4:8])

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
                    dof_positions[index] = self.apply_coordinate_transform_to_joint("waist_pitch_joint", temp_dof_pos[key])
                elif (key[:-5]+"_joint") in self.joint_mapping:
                    index = self.joint_mapping[(key[:-5]+"_joint")]
                    dof_positions[index] = self.apply_coordinate_transform_to_joint((key[:-5]+"_joint"), temp_dof_pos[key])

            dof_positions = self.apply_joint_limits(dof_positions)

        except Exception as e:
            self.get_logger().error(f"解析关节状态消息时出错: {e}")

        self.get_logger().info(f"目标位置: {dof_positions}")
        return dof_positions

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

    def joint_state_callback(self, msg: JointState):
        """重定向关节状态回调"""
        try:
            # 解析重定向后的关节状态
            target_positions = self.parse_retargeted_joint_state(msg)
            
            # 更新当前自由度位置
            self.current_dof_positions = target_positions
            self.latest_joint_state = msg
            self.last_joint_state_time = self.get_clock().now()
            
            # 重置安全计数器
            self.safety_counter = 0
            
            # 记录接收到的数据
            if len(msg.name) > 0:
                self.get_logger().debug(f"接收到关节状态: {len(msg.name)}个关节")
                
        except Exception as e:
            self.get_logger().error(f"处理关节状态回调时出错: {e}")

    

    def control_loop(self):
        """控制循环"""
        if not self.is_initialized or self.low_state is None:
            return
            
        try:
            # 安全检查：如果长时间没有收到新数据，停止机器人
            current_time = self.get_clock().now()
            if (self.last_joint_state_time is None or 
                (current_time - self.last_joint_state_time).nanoseconds > self.safety_timeout * 1e9):
                self.safety_counter += 1
                if self.safety_counter > 10:  # 10个控制周期后进入安全模式
                    self.enter_safety_mode()
                    return
            else:
                # 处理最新的关节状态
                self.process_joint_state()
            
        except Exception as e:
            self.get_logger().error(f"控制循环错误: {e}")

    def process_joint_state(self):
        """处理关节状态并生成控制命令"""
        # 重置低级命令
        self.low_cmd.mode_pr = Mode.PR
        self.low_cmd.mode_machine = self.mode_machine
        
        # 初始化所有关节为当前位置（安全默认值）
        for i in range(G1_NUM_MOTOR):
            self.low_cmd.motor_cmd[i].mode = 1  # 使能电机
            self.low_cmd.motor_cmd[i].tau = 0.0  # 零力矩
            self.low_cmd.motor_cmd[i].q = self.low_state.motor_state[i].q  # 保持当前位置
            self.low_cmd.motor_cmd[i].dq = 0.0  # 零速度
            self.low_cmd.motor_cmd[i].kp = Kp[i]  # 位置增益
            self.low_cmd.motor_cmd[i].kd = Kd[i]  # 速度增益
        
        # 应用目标位置到低级命令
        for i in range(G1_NUM_MOTOR):
            target_q = self.current_dof_positions[i]
            current_q = self.low_state.motor_state[i].q
            
            # 平滑过渡：使用小步长接近目标
            max_step = 0.1  # 最大步长（弧度）
            
            if abs(target_q - current_q) > max_step:
                step_direction = 1 if target_q > current_q else -1
                smoothed_q = current_q + step_direction * max_step
            else:
                smoothed_q = target_q

            
            self.low_cmd.motor_cmd[i].q = smoothed_q
        
        # 添加CRC校验并发送命令
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.get_logger().info(f"G1机器人low cmd状态: {self.low_cmd}")
        self.lowcmd_publisher.Write(self.low_cmd)
        self.get_logger().info(f"发送控制命令，目标位置: {self.current_dof_positions}") 
        # 记录控制命令
        self.get_logger().debug(f"发送控制命令，目标位置: {self.current_dof_positions[:5]}...")  # 只显示前5个位置

    def enter_safety_mode(self):
        """进入安全模式"""
        self.get_logger().warn("进入安全模式：停止所有运动")
        
        # 发送停止命令
        safety_cmd = unitree_hg_msg_dds__LowCmd_()
        safety_cmd.mode_pr = Mode.PR
        safety_cmd.mode_machine = self.mode_machine
        
        for i in range(G1_NUM_MOTOR):
            safety_cmd.motor_cmd[i].mode = 1
            safety_cmd.motor_cmd[i].tau = 0.0
            if self.low_state is not None:
                safety_cmd.motor_cmd[i].q = self.low_state.motor_state[i].q  # 保持当前位置
            else:
                safety_cmd.motor_cmd[i].q = 0.0  # 默认位置
            safety_cmd.motor_cmd[i].dq = 0.0
            safety_cmd.motor_cmd[i].kp = Kp[i]
            safety_cmd.motor_cmd[i].kd = Kd[i]
        
        safety_cmd.crc = self.crc.Crc(safety_cmd)
        self.lowcmd_publisher.Write(safety_cmd)
        
        self.get_logger().info("安全模式命令已发送")

    def destroy_node(self):
        """节点销毁时的清理工作"""
        self.get_logger().info("正在关闭G1运动控制器...")
        
        # 进入安全模式
        try:
            self.enter_safety_mode()
        except Exception as e:
            self.get_logger().error(f"进入安全模式时出错: {e}")
        
        # 释放运动模式
        if hasattr(self, 'motion_switcher'):
            try:
                self.motion_switcher.ReleaseMode()
                self.get_logger().info("运动模式已释放")
            except Exception as e:
                self.get_logger().error(f"释放运动模式时出错: {e}")
        
        super().destroy_node()
        self.get_logger().info("G1运动控制器已关闭")

def main(args=None):
    rclpy.init(args=args)
    
    try:
        controller = G1MotionController()
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info("接收到键盘中断信号")
    except Exception as e:
        print(f"G1运动控制器错误: {e}")
    finally:
        if 'controller' in locals():
            controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()