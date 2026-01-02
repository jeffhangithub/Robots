from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'motion_retargeting'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), 
         glob('launch/*.py')),
        (os.path.join('share', package_name, 'msg'), glob('msg/*.msg')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your-email@example.com',
    description='ROS2 Python package for motion retargeting from XSens data',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'data_subscriber = motion_retargeting.data_subscriber:main',
            'bvh_parser = motion_retargeting.bvh_parser:main',
            'mujoco_play = motion_retargeting.mujoco_play:main',
            'visualizer = motion_retargeting.retargeting_visualizer:main'
        ],
    },
)