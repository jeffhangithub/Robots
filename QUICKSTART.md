# 快速开始指南 - BVH 转换与可视化

## ✅ 环境已准备完毕

已成功创建 `robots_env` 环境，包含以下版本：

```
Python:        3.10.19
NumPy:         1.26.4   ← 关键！用于 pinocchio ABI 兼容性
SciPy:         1.15.2
Pinocchio:     3.8.0
MuJoCo:        3.3.7
PyTorch:       2.5.1
```

---

## 🚀 核心命令

### 1. 激活环境
```bash
bash /home/jeff/Codes/Robots/activate_robots_env.sh
```

**注意：** 此脚本清理了 ROS 环境污染，确保 conda 包被优先加载。

### 2. 转换 BVH 文件为 pickle
```bash
bash /home/jeff/Codes/Robots/activate_robots_env.sh && \
python /home/jeff/Codes/Robots/convert_bvh_to_pickle.py
```

**输出：** 
- 生成 `/home/jeff/Codes/Robots/output/g1/Geely test-001(1).pkl`
- 包含 11326 帧、37 DOF 的运动数据

### 3. 可视化机器人运动（交互模式）
```bash
bash /home/jeff/Codes/Robots/activate_robots_env.sh && \
python /home/jeff/Codes/Robots/src/vis_robot_motion.py \
  --xml_path /home/jeff/Codes/Robots/src/motion_retargeting/robots/g1/urdf/g1.xml \
  --robot_motion_path '/home/jeff/Codes/Robots/output/g1/Geely test-001(1).pkl'
```

**操作：**
- 空格：暂停/播放
- ← / →：前后帧
- 鼠标拖动：旋转视角
- 滚轮：缩放
- 按 Esc 或关闭窗口退出

---

## 📊 环境验证

检查所有关键依赖是否正确加载：

```bash
bash /home/jeff/Codes/Robots/activate_robots_env.sh && python -c "
import numpy as np, scipy, pinocchio as pin, mujoco as mj, torch
print('✓ NumPy:', np.__version__)
print('✓ Pinocchio:', pin.__version__)
print('✓ MuJoCo:', mj.__version__)
print('✓ PyTorch:', torch.__version__)
print('✅ 所有依赖就绪！')
"
```

---

## 🔧 Fallback vs Full Retargeting

### Fallback 模式（目前使用）
- 简单的 BVH 解析器
- 直接映射 BVH 关节到 G1 机器人
- 可能出现不自然的动作（当 BVH 和 G1 骨架差异大时）
- **优点：** 快速，依赖少

### 完整 Retargeting 模式（motion_retargeting 已安装）
- 使用 pinocchio + pink IK 求解器
- 自动调整 BVH 动作以适应 G1 机器人物理约束
- 结果更自然、更符合物理
- **需要：** torch（已装）、quaternion（需要手动安装）

**启用完整模式：** 安装 quaternion 库
```bash
bash /home/jeff/Codes/Robots/activate_robots_env.sh && \
conda install -y -c conda-forge quaternion
```

---

## 📁 关键文件路径

| 文件 | 用途 |
|------|------|
| `convert_bvh_to_pickle.py` | BVH → pickle 转换脚本 |
| `src/vis_robot_motion.py` | 可视化脚本 |
| `src/motion_retargeting/` | 重定目标包（自动装入环境） |
| `data/Geely test-001(1).bvh` | 输入 BVH 文件 |
| `output/g1/Geely test-001(1).pkl` | 输出运动数据 |
| `src/motion_retargeting/robots/g1/urdf/g1.xml` | G1 机器人 URDF 模型 |
| `DEPENDENCY_ANALYSIS.md` | 完整的版本依赖分析 |

---

## 🐛 故障排查

### 问题：`ImportError: No module named 'motion_retargeting'`
**解决：** motion_retargeting 已在 pip editable 模式下装入环境，应该能找到。
```bash
bash /home/jeff/Codes/Robots/activate_robots_env.sh && \
python -c "from motion_retargeting.retarget.retarget import BVHRetarget; print('✓')"
```

### 问题：`RuntimeError: No algebra backend available!`（仅 pink/qpsolvers 使用）
**原因：** osqp 的代数后端初始化失败，不影响基础功能。
**忽略：** BVH 转换仍会工作（使用 fallback 模式）。

### 问题：`libstdc++ version not found`
**原因：** 系统旧的 C++ 库优先加载。
**解决：** `activate_robots_env.sh` 已处理，确保用此脚本激活。

### 问题：可视化窗口不显示
**原因：** 图形界面未正确初始化（可能是 headless 环境）。
**解决：** 使用 `vis_robot_motion_headless.py` 生成视频（需要创建）。

---

## 📝 后续步骤

1. **对比 BVH 原始动作 vs G1 输出**
   - 运行转换脚本
   - 打开可视化窗口
   - 对比原始和重定目标的动作

2. **启用完整 retargeting（可选）**
   - 安装 quaternion
   - 重新运行转换脚本
   - 观察动作质量提升

3. **保存为视频（可选）**
   - 修改 `vis_robot_motion.py` 导出参数
   - 生成 MP4 视频对比

---

## 📚 参考文档

- 完整分析：[DEPENDENCY_ANALYSIS.md](DEPENDENCY_ANALYSIS.md)
- Pinocchio 文档：https://github.com/stack-of-tasks/pinocchio
- Pink IK 文档：https://github.com/stephane-caron/pink
- MuJoCo Python API：https://mujoco.readthedocs.io/

---

**环境创建时间：** 2025-12-23  
**建议维护：** 定期执行 `conda list > robots_env_lockfile.txt` 保存依赖快照
