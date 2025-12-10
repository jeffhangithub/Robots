from setuptools import setup
import os
from glob import glob

package_name = 'deploy_controller'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), 
         glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), 
         glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='宇树G1机器人部署控制器',
    license='Apache-2.0',
    # tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'g1_controller = deploy_controller.g1_controller:main',
            'g1_mujoco_play = deploy_controller.g1_mujoco_play:main',
        ],
    },
)