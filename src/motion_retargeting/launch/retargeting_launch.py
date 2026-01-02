# 作用：ROS2 launch 文件，启动 motion_retargeting 包的 data_subscriber 节点，并设置模型名与参考/目标坐标系参数。
# 注释人：Jeff
# 日期：2026-01-01
# 用法很简单，命令行运行 launch 文件即可启动 data_subscriber 节点：
# 新终端
# 1) 加载 ROS 2 环境
# source /opt/ros/humble/setup.bash
# 2) 加载工作区（已 colcon build 过）
# source install/setup.bash
# 3) 启动 launch
# ros2 launch motion_retargeting retargeting_launch.py

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='motion_retargeting',
            executable='data_subscriber',
            name='motion_retargeting_node',
            output='screen',
            parameters=[
                {
                    'model_name': 'skeleton',
                    'reference_frame': 'world', 
                    'target_frame': 'base_link'
                }
            ]
        )
    ])