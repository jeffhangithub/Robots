#!/bin/bash
# 激活 robots_env 并清理 ROS 污染的环境变量

conda activate robots_env

# 清理 ROS 的 Python 路径和 LD_LIBRARY_PATH
unset PYTHONPATH
unset LD_LIBRARY_PATH

# 设置 conda 环境优先
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${CONDA_PREFIX}/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export PYTHONDONTWRITEBYTECODE=1

echo "robots_env 已激活（ROS 污染已清理）"
echo "CONDA_PREFIX: $CONDA_PREFIX"
