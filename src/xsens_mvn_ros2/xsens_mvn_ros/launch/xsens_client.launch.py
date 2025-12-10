import os
from ament_index_python.packages import get_package_share_directory
from ament_index_python.packages import get_package_prefix
from launch import LaunchDescription
from launch.actions import SetEnvironmentVariable
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.conditions import IfCondition


def generate_launch_description():
    # 参数定义
    model_name = LaunchConfiguration('model_name', default='skeleton')
    reference_frame = LaunchConfiguration('reference_frame', default='world')
    udp_port = LaunchConfiguration('udp_port', default='8001')
    launch_rviz = LaunchConfiguration('launch_rviz', default='true')

    # 设置库路径
    env_var = SetEnvironmentVariable(
        name='LD_LIBRARY_PATH',
        value=os.path.join(get_package_prefix('xsens_mvn_ros'), 'lib/xsens_mvn_ros') + ':' + os.environ.get('LD_LIBRARY_PATH', '')
    )

    # 获取RViz配置文件路径
    rviz_config_file = os.path.join(
        get_package_share_directory('xsens_mvn_ros'),
        'rviz',
        'xsens_visualization.rviz'
    )
    
    # 定义XSens节点
    xsens_node = Node(
        package='xsens_mvn_ros',
        executable='xsens_client_node',
        name='xsens',
        output='screen',
        parameters=[{
            'model_name': model_name,
            'reference_frame': reference_frame,
            'udp_port': udp_port
        }]
    )
    
    # 定义RViz节点
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='xsens_rviz',
        output='screen',
        arguments=['-d', rviz_config_file],
        condition=IfCondition(launch_rviz)
    )
    
    return LaunchDescription([
        DeclareLaunchArgument('model_name', default_value='skeleton'),
        DeclareLaunchArgument('reference_frame', default_value='world'),
        DeclareLaunchArgument('udp_port', default_value='8001'),
        DeclareLaunchArgument('launch_rviz', default_value='true'),
        env_var,
        xsens_node,
        rviz_node
    ])