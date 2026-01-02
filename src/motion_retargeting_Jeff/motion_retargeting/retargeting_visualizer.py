#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from xsens_mvn_ros_msgs.msg import LinkStateArray
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.colors as mcolors
import threading
import time
from typing import Dict, List, Optional
import os
import json
from datetime import datetime

class RetargetingVisualizer(Node):
    def __init__(self):
        super().__init__('retargeting_visualizer')
        
        # 参数声明
        self.declare_parameter('visualization_type', 'stick_figure')  # stick_figure, mujoco, both
        self.declare_parameter('update_rate', 30.0)
        self.declare_parameter('save_directory', './visualization_frames')
        self.declare_parameter('save_interval', 10)  # 每10帧保存一次
        self.declare_parameter('enable_mujoco', False)
        
        self.visualization_type = self.get_parameter('visualization_type').value
        self.update_rate = self.get_parameter('update_rate').value
        self.save_directory = self.get_parameter('save_directory').value
        self.save_interval = self.get_parameter('save_interval').value
        self.enable_mujoco = self.get_parameter('enable_mujoco').value
        
        # 数据存储
        self.positions: Dict[str, np.ndarray] = {}
        self.rotations: Dict[str, np.ndarray] = {}
        self.latest_data_time = 0
        self.frame_count = 0
        self.save_count = 0
        
        # 新增：自定义QoS配置
        qos_profile = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,  # 与发布者保持一致
            history=rclpy.qos.HistoryPolicy.KEEP_LAST,
            depth=10
        )


        # 订阅统一的重定向数据主题
        self.joint_sub = self.create_subscription(
            JointState,
            '/retargeted_joint_states',
            self.joint_callback,
            qos_profile
        )
        
        self.link_sub = self.create_subscription(
            LinkStateArray,
            '/retargeted_link_states',
            self.link_callback,
            qos_profile
        )
        
        # 创建保存目录
        os.makedirs(self.save_directory, exist_ok=True)
        
        # 可视化设置
        self.fig = None
        self.ax = None
        self.scatter = None
        self.lines = None
        
        # 关节连接定义
        self.joint_connections = [
            # 脊柱
            ('pelvis', 'spine'),
            ('spine', 'spine1'),
            ('spine1', 'spine2'),
            ('spine2', 'spine3'),
            ('spine3', 'neck'),
            ('neck', 'head'),
            
            # 左臂
            ('spine3', 'left_shoulder'),
            ('left_shoulder', 'left_upper_arm'),
            ('left_upper_arm', 'left_forearm'),
            ('left_forearm', 'left_hand'),
            
            # 右臂
            ('spine3', 'right_shoulder'),
            ('right_shoulder', 'right_upper_arm'),
            ('right_upper_arm', 'right_forearm'),
            ('right_forearm', 'right_hand'),
            
            # 左腿
            ('pelvis', 'left_hip'),
            ('left_hip', 'left_thigh'),
            ('left_thigh', 'left_shin'),
            ('left_shin', 'left_foot'),
            
            # 右腿
            ('pelvis', 'right_hip'),
            ('right_hip', 'right_thigh'),
            ('right_thigh', 'right_shin'),
            ('right_shin', 'right_foot')
        ]
        
        # 颜色映射
        self.colors = list(mcolors.TABLEAU_COLORS.values())
        self.part_colors = {
            "head": self.colors[0],
            "neck": self.colors[1],
            "back": self.colors[2],
            "shoulder": self.colors[3],
            "arm": self.colors[4],
            "elbow": self.colors[5],
            "hip": self.colors[6],
            "leg": self.colors[7],
            "knee": self.colors[8],
            "foot": self.colors[9]
        }
        
        # 初始化可视化
        self.init_visualization()
        
        # 定时器
        self.timer = self.create_timer(1.0/self.update_rate, self.update_visualization)
        
        self.get_logger().info("Retargeting visualizer initialized")

    def init_visualization(self):
        """初始化可视化界面"""
        plt.ion()  # 开启交互模式
    
        self.fig = plt.figure(figsize=(12, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # 设置初始视图
        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)
        self.ax.set_zlim(-1, 1)
        
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        self.ax.set_title('Motion Retargeting Visualization')
        
        # 添加网格
        self.ax.grid(True, linestyle='--', alpha=0.6)
        
        # 初始化散点图和连线
        self.scatter = self.ax.scatter([], [], [], c='blue', s=80, alpha=0.8, zorder=10)
        self.lines = Line3DCollection([], colors=[], linewidths=3.0, alpha=0.9)
        self.ax.add_collection3d(self.lines)
        
        # 添加图例
        legend_elements = [
            plt.Line2D([0], [0], color=self.part_colors["head"], lw=3, label='head'),
            plt.Line2D([0], [0], color=self.part_colors["neck"], lw=3, label='neck'),
            plt.Line2D([0], [0], color=self.part_colors["back"], lw=3, label='back'),
            plt.Line2D([0], [0], color=self.part_colors["shoulder"], lw=3, label='shoulder'),
            plt.Line2D([0], [0], color=self.part_colors["arm"], lw=3, label='arm'),
            plt.Line2D([0], [0], color=self.part_colors["elbow"], lw=3, label='elbow'),
            plt.Line2D([0], [0], color=self.part_colors["leg"], lw=3, label='leg'),
            plt.Line2D([0], [0], color=self.part_colors["knee"], lw=3, label='knee'),
            plt.Line2D([0], [0], color=self.part_colors["hip"], lw=3, label='hip'),
            plt.Line2D([0], [0], color=self.part_colors["foot"], lw=3, label='foot')
        ]
        self.ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1))
        plt.tight_layout()

    def joint_callback(self, msg: JointState):
        """关节状态回调函数"""
        try:
            # 提取位置数据
            for i, name in enumerate(msg.name):
                if i * 3 + 2 < len(msg.position):
                    self.positions[name] = np.array([
                        msg.position[i*3],
                        msg.position[i*3+1],
                        msg.position[i*3+2]
                    ])
            
            self.latest_data_time = time.time()
            self.frame_count += 1
            
        except Exception as e:
            self.get_logger().error(f"Error processing joint states: {e}")

    def link_callback(self, msg: LinkStateArray):
        """链接状态回调函数"""
        try:
            # 提取位置和旋转数据
            for link_state in msg.states:
                joint_name = link_state.header.frame_id
                
                # 位置数据
                self.positions[joint_name] = np.array([
                    link_state.pose.position.x,
                    link_state.pose.position.y,
                    link_state.pose.position.z
                ])
                
                # 旋转数据
                self.rotations[joint_name] = np.array([
                    link_state.pose.orientation.x,
                    link_state.pose.orientation.y,
                    link_state.pose.orientation.z,
                    link_state.pose.orientation.w
                ])
            
            self.latest_data_time = time.time()
            
        except Exception as e:
            self.get_logger().error(f"Error processing link states: {e}")

    def update_visualization(self):
        """更新可视化"""
        if not self.positions:
            return
            
        try:
            # 清除之前的绘图
            self.scatter._offsets3d = ([], [], [])
            self.lines.set_segments([])
            
            # 收集所有关节位置
            positions_list = []
            joint_names = []
            
            for name, pos in self.positions.items():
                positions_list.append(pos)
                joint_names.append(name)
            
            if not positions_list:
                return
                
            positions_array = np.array(positions_list)
            
            # 更新散点图
            self.scatter._offsets3d = (
                positions_array[:, 0],
                positions_array[:, 1],
                positions_array[:, 2]
            )
            
            # 创建连接线
            lines = []
            line_colors = []
            
            for connection in self.joint_connections:
                joint1, joint2 = connection

                # # 获取对应颜色
                # color1 = self.part_colors.get(joint1, self.colors[0])
                # color2 = self.part_colors.get(joint2, self.colors[0])
                
                # # 使用渐变色连接（可选）
                # line_color = [(c1 + c2)/2 for c1, c2 in zip(color1, color2)]
                
                # if joint1 in self.positions and joint2 in self.positions:
                #     lines.append([pos1, pos2])
                #     line_colors.append(line_color)
                
                if joint1 in self.positions and joint2 in self.positions:
                    pos1 = self.positions[joint1]
                    pos2 = self.positions[joint2]
                    
                    lines.append([pos1, pos2])
                    
                    # 根据关节名称确定颜色
                    color = self.colors[0]  # 默认颜色
                    
                    if "head" in joint1.lower() or "head" in joint2.lower():
                        color = self.part_colors["head"]
                    elif "neck" in joint1.lower() or "neck" in joint2.lower():
                        color = self.part_colors["neck"]
                    elif "spine" in joint1.lower() or "spine" in joint2.lower():
                        color = self.part_colors["back"]
                    elif "shoulder" in joint1.lower() or "shoulder" in joint2.lower():
                        color = self.part_colors["shoulder"]
                    elif "arm" in joint1.lower() or "arm" in joint2.lower():
                        color = self.part_colors["arm"]
                    elif "elbow" in joint1.lower() or "elbow" in joint2.lower():
                        color = self.part_colors["elbow"]
                    elif "hand" in joint1.lower() or "hand" in joint2.lower():
                        color = self.part_colors["hand"]
                    elif "hip" in joint1.lower() or "hip" in joint2.lower():
                        color = self.part_colors["hip"]
                    elif "thigh" in joint1.lower() or "thigh" in joint2.lower():
                        color = self.part_colors["leg"]
                    elif "knee" in joint1.lower() or "knee" in joint2.lower():
                        color = self.part_colors["knee"]
                    elif "foot" in joint1.lower() or "foot" in joint2.lower():
                        color = self.part_colors["foot"]
                    
                    line_colors.append(color)
            
            # 更新连线
            if lines:
                self.lines.set_segments(lines)
                self.lines.set_colors(line_colors)
            
            # 自动调整视图范围（修复版）
            if positions_array.size > 0:
                x_min, x_max = positions_array[:, 0].min(), positions_array[:, 0].max()
                y_min, y_max = positions_array[:, 1].min(), positions_array[:, 1].max()
                z_min, z_max = positions_array[:, 2].min(), positions_array[:, 2].max()
                
                padding = (max(x_max - x_min, y_max - y_min, z_max - z_min) * 0.1)
                
                self.ax.set_xlim(x_min - padding, x_max + padding)
                self.ax.set_ylim(y_min - padding, y_max + padding)
                self.ax.set_zlim(z_min - padding, z_max + padding)
            
            # 更新标题
            self.ax.set_title(f'Motion Retargeting Visualization - Frame {self.frame_count}')
            
            # 重绘图形
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            
            # 定期保存帧
            self.save_count += 1
            if self.save_count >= self.save_interval:
                self.save_frame()
                self.save_count = 0
            
        except Exception as e:
            self.get_logger().error(f"Error updating visualization: {e}")

    def save_frame(self):
        """保存当前帧为图像"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.save_directory, f"frame_{self.frame_count:06d}_{timestamp}.png")
            self.fig.savefig(filename, dpi=150, bbox_inches='tight')
            
            # 每保存100帧打印一次信息
            if self.frame_count % 100 == 0:
                self.get_logger().info(f"Saved frame {self.frame_count} to {filename}")
                
        except Exception as e:
            self.get_logger().error(f"Error saving frame: {e}")

    def save_data(self):
        """保存位置数据到文件"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.save_directory, f"data_{timestamp}.json")
            
            data = {
                "timestamp": timestamp,
                "frame_count": self.frame_count,
                "positions": {name: pos.tolist() for name, pos in self.positions.items()},
                "rotations": {name: rot.tolist() for name, rot in self.rotations.items()}
            }
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
                
            self.get_logger().info(f"Saved data to {filename}")
            
        except Exception as e:
            self.get_logger().error(f"Error saving data: {e}")

    def destroy_node(self):
        """重写销毁方法，确保正确关闭可视化"""
        try:
            if self.fig:
                plt.close(self.fig)
            self.get_logger().info("Visualization closed")
        except Exception as e:
            self.get_logger().error(f"Error closing visualization: {e}")
        
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    
    try:
        visualizer = RetargetingVisualizer()
        rclpy.spin(visualizer)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error in retargeting visualizer: {e}")
    finally:
        if 'visualizer' in locals():
            visualizer.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()