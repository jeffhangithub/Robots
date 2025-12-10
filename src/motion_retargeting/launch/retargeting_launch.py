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