#!/bin/bash
# 基于 motion_retargeting 的完整 BVH 转换脚本

set -e  # 遇到错误立即退出

echo "======================================"
echo "BVH 转换 - 完整 Retargeting 管道"
echo "======================================"
echo ""

# 1. 激活环境
echo "📦 激活 robots_env 环境..."
source /home/jeff/miniforge/etc/profile.d/conda.sh
conda activate /home/jeff/miniforge/envs/robots_env
unset PYTHONPATH
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# 2. 确保 quaternion 已装
echo "✓ 检查 quaternion 包..."
if ! python -c "import quaternion" 2>/dev/null; then
    echo "⚠️  quaternion 未安装，正在安装..."
    pip install quaternion -q || pip install --upgrade quaternion -q
    echo "✓ quaternion 已安装"
else
    echo "✓ quaternion 已存在"
fi

# 3. 验证 motion_retargeting 可导入
echo ""
echo "✓ 验证 motion_retargeting 导入..."
python -c "
from motion_retargeting.retarget.retarget import BVHRetarget, Joint
from motion_retargeting.config.robot.g1 import G1_BVH_CONFIG
print('✅ motion_retargeting 完整管道已就绪！')
" || {
    echo "❌ 导入失败，将使用 fallback 模式"
    exit 1
}

# 4. 运行完整转换
echo ""
echo "🔄 运行 BVH → pickle 转换（完整 retargeting）..."
echo "   输入：/home/jeff/Codes/Robots/data/Geely test-001(1).bvh"
echo "   输出：/home/jeff/Codes/Robots/output/g1/Geely test-001(1).pkl"
echo ""

python /home/jeff/Codes/Robots/convert_bvh_to_pickle.py

echo ""
echo "✅ 转换完成！"
echo ""
echo "下一步：可视化结果"
echo "-------"
echo "python /home/jeff/Codes/Robots/src/vis_robot_motion.py \\"
echo "  --xml_path /home/jeff/Codes/Robots/src/motion_retargeting/robots/g1/urdf/g1.xml \\"
echo "  --robot_motion_path '/home/jeff/Codes/Robots/output/g1/Geely test-001(1).pkl'"
