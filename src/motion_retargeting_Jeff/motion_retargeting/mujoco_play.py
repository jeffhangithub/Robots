#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import mujoco
import mujoco.viewer
import threading
import time
from typing import Dict, List, Optional
import os

# ROS2消息类型
from sensor_msgs.msg import JointState
from xsens_mvn_ros_msgs.msg import LinkStateArray, LinkState
from geometry_msgs.msg import Point, Quaternion

class MuJoCoVisualizer(Node):
    def __init__(self):
        super().__init__('mujoco_visualizer')
        
        # 参数声明
        self.declare_parameter('robot_model', 'g1')  # 机器人模型名称
        self.declare_parameter('mujoco_model_path', '')  # MuJoCo模型文件路径
        self.declare_parameter('update_rate', 60.0)  # 更新频率
        self.declare_parameter('data_source', 'xsens')  # 数据源: 'xsens' 或 'bvh'
        
        self.robot_model = self.get_parameter('robot_model').value
        self.mujoco_model_path = self.get_parameter('mujoco_model_path').value
        self.update_rate = self.get_parameter('update_rate').value
        self.data_source = self.get_parameter('data_source').value
        
        # 如果没有提供模型路径，使用默认路径
        if not self.mujoco_model_path:
            self.mujoco_model_path = self._get_default_model_path()
        
        # 加载MuJoCo模型
        self.model = self.load_mujoco_model()
        self.data = mujoco.MjData(self.model)
        
        # 存储最新的重定向数据
        self.latest_joint_states: Optional[JointState] = None
        self.latest_link_states: Optional[LinkStateArray] = None
        self.latest_timestamp = None
        
        # 关节名称映射（从重定向数据到MuJoCo模型）
        self.joint_mapping = self._create_joint_mapping()
        
        # 订阅重定向数据
        qos_profile = rclpy.qos.QoSProfile(
            depth=10,
            reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
            durability=rclpy.qos.DurabilityPolicy.VOLATILE
        )
        
        self.joint_sub = self.create_subscription(
            JointState,
            '/retargeted_joint_states',
            self.joint_state_callback,
            qos_profile
        )
        
        self.link_sub = self.create_subscription(
            LinkStateArray,
            '/retargeted_link_states',
            self.link_state_callback,
            qos_profile
        )
        
        # MuJoCo可视化线程
        self.visualization_thread = None
        self.is_running = False
        
        # 启动可视化
        self.start_visualization()
        
        self.get_logger().info(f"MuJoCo可视化节点已启动，数据源: {self.data_source}")
        self.get_logger().info(f"使用模型: {self.mujoco_model_path}")
    
    def _get_default_model_path(self) -> str:
        """获取默认的MuJoCo模型路径"""
        # 根据机器人模型名称返回对应的默认模型文件
        model_dir = "/home/uneedrobot/workshops/source/xsens/ros2/src/motion_retargeting/motion_retargeting/config/robot/models"
        
        model_files = {
            'g1': 'g1_edited.xml',
            'h1': 'h1_edited.xml', 
            'loong': 'loong_robot.xml',
            'hi_1': 'hi1_robot.xml'
        }
        
        filename = model_files.get(self.robot_model, 'g1_robot.xml')
        return os.path.join(model_dir, filename)
    
    def load_mujoco_model(self) -> mujoco.MjModel:
        """加载MuJoCo模型"""
        try:
            if not os.path.exists(self.mujoco_model_path):
                self.get_logger().error(f"模型文件不存在: {self.mujoco_model_path}")
                # 创建一个简单的默认模型
                return self._create_default_model()
            
            model = mujoco.MjModel.from_xml_path(self.mujoco_model_path)
            self.get_logger().info(f"成功加载MuJoCo模型: {self.mujoco_model_path}")
            return model
            
        except Exception as e:
            self.get_logger().error(f"加载MuJoCo模型失败: {e}")
            return self._create_default_model()
    
    def _create_default_model(self) -> mujoco.MjModel:
        """创建默认的MuJoCo模型（当模型文件不存在时）"""
        # 创建一个简单的人形机器人模型
        model_xml = """
        <mujoco>
            <option timestep="0.01"/>
            <worldbody>
                <light name="light" pos="0 0 3" dir="0 0 -1"/>
                <camera name="fixed" pos="3 3 1" xyaxes="1 0 0 0 1 0"/>
                
                <!-- 地面 -->
                <geom name="ground" type="plane" size="10 10 0.1" pos="0 0 -0.1" rgba="0.8 0.9 0.8 1"/>
                
                <!-- 骨盆 -->
                <body name="pelvis" pos="0 0 1.0">
                    <joint name="pelvis_joint" type="free"/>
                    <geom name="pelvis_geom" type="sphere" size="0.08" rgba="0.8 0.3 0.3 1"/>
                    
                    <!-- 脊柱 -->
                    <body name="spine" pos="0 0 0.1">
                        <joint name="spine_joint" type="ball" axis="0 1 0"/>
                        <geom name="spine_geom" type="capsule" fromto="0 0 0 0 0 0.15" size="0.05" rgba="0.3 0.8 0.3 1"/>
                        
                        <!-- 胸部 -->
                        <body name="spine1" pos="0 0 0.15">
                            <joint name="spine1_joint" type="ball"/>
                            <geom name="spine1_geom" type="capsule" fromto="0 0 0 0 0 0.15" size="0.05" rgba="0.3 0.3 0.8 1"/>
                            
                            <!-- 颈部 -->
                            <body name="neck" pos="0 0 0.15">
                                <joint name="neck_joint" type="ball"/>
                                <geom name="neck_geom" type="capsule" fromto="0 0 0 0 0 0.1" size="0.04" rgba="0.8 0.8 0.3 1"/>
                                
                                <!-- 头部 -->
                                <body name="head" pos="0 0 0.1">
                                    <joint name="head_joint" type="ball"/>
                                    <geom name="head_geom" type="sphere" size="0.1" rgba="0.8 0.6 0.3 1"/>
                                </body>
                            </body>
                            
                            <!-- 左肩 -->
                            <body name="left_shoulder" pos="0.1 0 0.1">
                                <joint name="left_shoulder_joint" type="ball"/>
                                <geom name="left_shoulder_geom" type="sphere" size="0.05" rgba="0.8 0.3 0.8 1"/>
                                
                                <!-- 左上臂 -->
                                <body name="left_upper_arm" pos="0.1 0 0">
                                    <joint name="left_upper_arm_joint" type="ball"/>
                                    <geom name="left_upper_arm_geom" type="capsule" fromto="0 0 0 0.2 0 0" size="0.04" rgba="0.3 0.8 0.8 1"/>
                                    
                                    <!-- 左前臂 -->
                                    <body name="left_forearm" pos="0.2 0 0">
                                        <joint name="left_forearm_joint" type="ball"/>
                                        <geom name="left_forearm_geom" type="capsule" fromto="0 0 0 0.2 0 0" size="0.035" rgba="0.8 0.8 0.3 1"/>
                                        
                                        <!-- 左手 -->
                                        <body name="left_hand" pos="0.2 0 0">
                                            <joint name="left_hand_joint" type="ball"/>
                                            <geom name="left_hand_geom" type="sphere" size="0.04" rgba="0.3 0.3 0.3 1"/>
                                        </body>
                                    </body>
                                </body>
                            </body>
                            
                            <!-- 右肩 -->
                            <body name="right_shoulder" pos="-0.1 0 0.1">
                                <joint name="right_shoulder_joint" type="ball"/>
                                <geom name="right_shoulder_geom" type="sphere" size="0.05" rgba="0.8 0.3 0.8 1"/>
                                
                                <!-- 右上臂 -->
                                <body name="right_upper_arm" pos="-0.1 0 0">
                                    <joint name="right_upper_arm_joint" type="ball"/>
                                    <geom name="right_upper_arm_geom" type="capsule" fromto="0 0 0 -0.2 0 0" size="0.04" rgba="0.3 0.8 0.8 1"/>
                                    
                                    <!-- 右前臂 -->
                                    <body name="right_forearm" pos="-0.2 0 0">
                                        <joint name="right_forearm_joint" type="ball"/>
                                        <geom name="right_forearm_geom" type="capsule" fromto="0 0 0 -0.2 0 0" size="0.035" rgba="0.8 0.8 0.3 1"/>
                                        
                                        <!-- 右手 -->
                                        <body name="right_hand" pos="-0.2 0 0">
                                            <joint name="right_hand_joint" type="ball"/>
                                            <geom name="right_hand_geom" type="sphere" size="0.04" rgba="0.3 0.3 0.3 1"/>
                                        </body>
                                    </body>
                                </body>
                            </body>
                        </body>
                    </body>
                    
                    <!-- 左髋 -->
                    <body name="left_hip" pos="0.1 0 -0.1">
                        <joint name="left_hip_joint" type="ball"/>
                        <geom name="left_hip_geom" type="sphere" size="0.06" rgba="0.8 0.5 0.3 1"/>
                        
                        <!-- 左大腿 -->
                        <body name="left_thigh" pos="0 0 -0.1">
                            <joint name="left_thigh_joint" type="ball"/>
                            <geom name="left_thigh_geom" type="capsule" fromto="0 0 0 0 0 -0.3" size="0.05" rgba="0.3 0.5 0.8 1"/>
                            
                            <!-- 左小腿 -->
                            <body name="left_shin" pos="0 0 -0.3">
                                <joint name="left_shin_joint" type="ball"/>
                                <geom name="left_shin_geom" type="capsule" fromto="0 0 0 0 0 -0.3" size="0.045" rgba="0.8 0.5 0.8 1"/>
                                
                                <!-- 左脚 -->
                                <body name="left_foot" pos="0 0 -0.3">
                                    <joint name="left_foot_joint" type="ball"/>
                                    <geom name="left_foot_geom" type="box" size="0.08 0.04 0.02" rgba="0.5 0.5 0.5 1"/>
                                </body>
                            </body>
                        </body>
                    </body>
                    
                    <!-- 右髋 -->
                    <body name="right_hip" pos="-0.1 0 -0.1">
                        <joint name="right_hip_joint" type="ball"/>
                        <geom name="right_hip_geom" type="sphere" size="0.06" rgba="0.8 0.5 0.3 1"/>
                        
                        <!-- 右大腿 -->
                        <body name="right_thigh" pos="0 0 -0.1">
                            <joint name="right_thigh_joint" type="ball"/>
                            <geom name="right_thigh_geom" type="capsule" fromto="0 0 0 0 0 -0.3" size="0.05" rgba="0.3 0.5 0.8 1"/>
                            
                            <!-- 右小腿 -->
                            <body name="right_shin" pos="0 0 -0.3">
                                <joint name="right_shin_joint" type="ball"/>
                                <geom name="right_shin_geom" type="capsule" fromto="0 0 0 0 0 -0.3" size="0.045" rgba="0.8 0.5 0.8 1"/>
                                
                                <!-- 右脚 -->
                                <body name="right_foot" pos="0 0 -0.3">
                                    <joint name="right_foot_joint" type="ball"/>
                                    <geom name="right_foot_geom" type="box" size="0.08 0.04 0.02" rgba="0.5 0.5 0.5 1"/>
                                </body>
                            </body>
                        </body>
                    </body>
                </body>
            </worldbody>
        </mujoco>
        """
        
        try:
            model = mujoco.MjModel.from_xml_string(model_xml)
            self.get_logger().info("已创建默认人形机器人模型")
            return model
        except Exception as e:
            self.get_logger().error(f"创建默认模型失败: {e}")
            # 创建一个最简模型
            simple_xml = """
            <mujoco>
                <option timestep="0.01"/>
                <worldbody>
                    <light pos="0 0 3"/>
                    <geom type="plane" size="10 10 0.1" pos="0 0 -0.1"/>
                    <body pos="0 0 1">
                        <joint type="free"/>
                        <geom type="sphere" size="0.1"/>
                    </body>
                </worldbody>
            </mujoco>
            """
            return mujoco.MjModel.from_xml_string(simple_xml)
    
    def _create_joint_mapping(self) -> Dict[str, str]:
        """创建关节名称映射（从重定向数据到MuJoCo模型）"""
        # 基本的人形机器人关节映射
        mapping = {
            'pelvis': 'pelvis_joint',
            'spine': 'spine_joint',
            'spine1': 'spine1_joint',
            'spine2': 'spine1_joint',  # 简化映射
            'spine3': 'spine1_joint',  # 简化映射
            'neck': 'neck_joint',
            'head': 'head_joint',
            'left_shoulder': 'left_shoulder_joint',
            'left_upper_arm': 'left_upper_arm_joint',
            'left_forearm': 'left_forearm_joint',
            'left_hand': 'left_hand_joint',
            'right_shoulder': 'right_shoulder_joint',
            'right_upper_arm': 'right_upper_arm_joint',
            'right_forearm': 'right_forearm_joint',
            'right_hand': 'right_hand_joint',
            'left_hip': 'left_hip_joint',
            'left_thigh': 'left_thigh_joint',
            'left_shin': 'left_shin_joint',
            'left_foot': 'left_foot_joint',
            'right_hip': 'right_hip_joint',
            'right_thigh': 'right_thigh_joint',
            'right_shin': 'right_shin_joint',
            'right_foot': 'right_foot_joint'
        }
        
        return mapping
    
    def joint_state_callback(self, msg: JointState):
        """处理关节状态消息"""
        self.latest_joint_states = msg
        self.latest_timestamp = msg.header.stamp
        
        # 调试信息
        if hasattr(self, 'joint_callback_count'):
            self.joint_callback_count += 1
        else:
            self.joint_callback_count = 1
            
        if self.joint_callback_count % 100 == 0:
            self.get_logger().info(f"已接收 {self.joint_callback_count} 个关节状态消息")
    
    def link_state_callback(self, msg: LinkStateArray):
        """处理链接状态消息"""
        self.latest_link_states = msg
        self.latest_timestamp = msg.states[0].header.stamp if msg.states else None
        
        # 调试信息
        if hasattr(self, 'link_callback_count'):
            self.link_callback_count += 1
        else:
            self.link_callback_count = 1
            
        if self.link_callback_count % 100 == 0:
            self.get_logger().info(f"已接收 {self.link_callback_count} 个链接状态消息")
    
    def start_visualization(self):
        """启动MuJoCo可视化"""
        self.is_running = True
        self.visualization_thread = threading.Thread(target=self._visualization_loop)
        self.visualization_thread.daemon = True
        self.visualization_thread.start()
        
        self.get_logger().info("MuJoCo可视化线程已启动")
    
    def _visualization_loop(self):
        """MuJoCo可视化主循环"""
        try:
            # 启动MuJoCo查看器
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                self.get_logger().info("MuJoCo查看器已启动")
                
                # 设置查看器选项
                viewer.cam.distance = 3.0
                viewer.cam.azimuth = 45
                viewer.cam.elevation = -20
                
                last_update_time = time.time()
                update_interval = 1.0 / self.update_rate
                
                while viewer.is_running and self.is_running:
                    current_time = time.time()
                    
                    # 控制更新频率
                    if current_time - last_update_time >= update_interval:
                        # 更新MuJoCo数据
                        self._update_mujoco_data()
                        
                        # 同步查看器
                        viewer.sync()
                        
                        last_update_time = current_time
                    
                    # 短暂休眠以避免过度占用CPU
                    time.sleep(0.001)
                
                self.get_logger().info("MuJoCo查看器已关闭")
                
        except Exception as e:
            self.get_logger().error(f"MuJoCo可视化错误: {e}")
    
    def _update_mujoco_data(self):
        """更新MuJoCo数据"""
        try:
            # 如果有新的关节状态数据，更新关节位置
            if self.latest_joint_states:
                self._update_joint_positions()
            
            # 如果有新的链接状态数据，更新身体位置（可选）
            if self.latest_link_states:
                self._update_body_positions()
            
            # 前向动力学计算
            mujoco.mj_forward(self.model, self.data)
            
        except Exception as e:
            self.get_logger().error(f"更新MuJoCo数据错误: {e}")
    
    def _update_joint_positions(self):
        """根据关节状态更新MuJoCo关节位置"""
        if not self.latest_joint_states:
            return
        
        for i, joint_name in enumerate(self.latest_joint_states.name):
            # 查找对应的MuJoCo关节名称
            mujoco_joint_name = self.joint_mapping.get(joint_name)
            if not mujoco_joint_name:
                continue
            
            # 查找关节ID
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, mujoco_joint_name)
            if joint_id == -1:
                continue
            
            # 获取关节数据地址
            qpos_adr = self.model.jnt_qposadr[joint_id]
            joint_type = self.model.jnt_type[joint_id]
            
            # 根据关节类型设置位置
            if joint_type == mujoco.mjtJoint.mjJNT_FREE:
                # 自由关节（6自由度）
                if i < len(self.latest_joint_states.position) - 6:
                    pos = self.latest_joint_states.position[i:i+3]
                    quat = self.latest_joint_states.position[i+3:i+7]
                    self.data.qpos[qpos_adr:qpos_adr+3] = pos
                    self.data.qpos[qpos_adr+3:qpos_adr+7] = quat
            elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
                # 球关节（3自由度）
                if i < len(self.latest_joint_states.position) - 3:
                    quat = self.latest_joint_states.position[i:i+4]
                    self.data.qpos[qpos_adr:qpos_adr+4] = quat
            else:
                # 旋转关节或滑动关节（1自由度）
                if i < len(self.latest_joint_states.position):
                    self.data.qpos[qpos_adr] = self.latest_joint_states.position[i]
    
    def _update_body_positions(self):
        """根据链接状态更新MuJoCo身体位置（使用直接位置设置）"""
        if not self.latest_link_states:
            return
        
        for link_state in self.latest_link_states.states:
            body_name = link_state.header.frame_id
            
            # 查找身体ID
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id == -1:
                continue
            
            # 直接设置身体位置和方向
            # 注意：这可能会破坏物理约束，主要用于可视化
            self.data.xpos[body_id] = [
                link_state.pose.position.x,
                link_state.pose.position.y,
                link_state.pose.position.z
            ]
            
            self.data.xquat[body_id] = [
                link_state.pose.orientation.w,
                link_state.pose.orientation.x,
                link_state.pose.orientation.y,
                link_state.pose.orientation.z
            ]
    
    def destroy_node(self):
        """清理资源"""
        self.is_running = False
        if self.visualization_thread and self.visualization_thread.is_alive():
            self.visualization_thread.join(timeout=2.0)
        
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    
    try:
        visualizer = MuJoCoVisualizer()
        rclpy.spin(visualizer)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"MuJoCo可视化节点错误: {e}")
    finally:
        if 'visualizer' in locals():
            visualizer.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()