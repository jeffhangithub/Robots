from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # 获取包共享目录
    pkg_dir = get_package_share_directory('deploy_controller')
    
    # 机器人参数
    robot_ip = '192.168.123.164'
    robot_port = 8000
    control_rate = 50.0
    publish_rate = 50.0
    
    # 创建节点
    g1_controller_node = Node(
        package='deploy_controller',
        executable='g1_controller',
        name='g1_controller',
        output='screen',
        parameters=[{
            'robot_ip': robot_ip,
            'robot_port': robot_port,
            'control_rate': control_rate,
            'publish_rate': publish_rate,
            'cmd_timeout': 1.0
        }],
        remappings=[
            ('/g1/cmd_vel', '/cmd_vel'),
            ('/g1/joint_commands', '/joint_commands'),
            ('/g1/joint_states', '/joint_states'),
            ('/g1/odom', '/odom'),
            ('/g1/imu', '/imu')
        ]
    )
    
    # 测试节点（可选）
    test_controller_node = Node(
        package='deploy_controller',
        executable='test_controller',
        name='test_controller',
        output='screen',
        parameters=[]
    )
    
    return LaunchDescription([
        g1_controller_node,
        # test_controller_node  # 取消注释以启用测试节点
    ])