"""
文件作用：定义 motion_retargeting 这个 ROS2 Python 包的打包与安装配置，包括依赖、数据文件以及可执行入口。
作者：Jeff
日期：2026-01-01

安装后生成可执行命令：

data_subscriber → 调用 motion_retargeting.data_subscriber:main
bvh_parser → 调用 motion_retargeting.bvh_parser:main
mujoco_play → 调用 motion_retargeting.mujoco_play:main
visualizer → 调用 motion_retargeting.retargeting_visualizer:main
安装完成后可直接用这些命令启动对应脚本（如 ros2 run motion_retargeting bvh_parser）

"""

from setuptools import setup, find_packages # 用于打包和安装 Python 包
import os
from glob import glob

package_name = 'motion_retargeting'

setup(
    name=package_name, # 包名
    version='1.0.0',# 版本号
    packages=find_packages(),# 自动发现所有子包
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]), # 把 resource/motion_retargeting 复制到 ament 索引目录，用于让 ROS 发现该包。
        ('share/' + package_name, ['package.xml']),# 把 package.xml 安装到 share/motion_retargeting，供 ROS 读取包元数据。
        (os.path.join('share', package_name, 'launch'), 
         glob('launch/*.py')),# 把 launch 目录下所有 .py 启动文件安装到对应的 launch 目录，运行 ros2 launch 时可用
        (os.path.join('share', package_name, 'msg'), glob('msg/*.msg')),# 把当前包下 msg 目录里的所有 .msg 文件，安装到目标路径 share/motion_retargeting/msg（即 share/<package_name>/msg），以便 ROS 2 构建与分发时能找到消息定义文件。
    ],
    install_requires=['setuptools'], #
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