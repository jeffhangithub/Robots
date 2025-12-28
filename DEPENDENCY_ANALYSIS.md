# 程序依赖关系分析：BVH → Pickle 转换环境

## 1. 核心依赖链分析

### 直接导入依赖（来自代码分析）

**convert_bvh_to_pickle.py 导入：**
```python
import numpy as np                    # ✓ 必需
import pickle                         # ✓ 内置
from scipy.spatial.transform import Rotation  # ✓ 必需
from motion_retargeting.retarget.retarget import BVHRetarget, Joint  # 条件导入
from motion_retargeting.config.robot.g1 import G1_BVH_CONFIG        # 条件导入
```

**vis_robot_motion.py 导入：**
```python
import mujoco as mj                  # ✓ 必需（可视化用）
import mujoco.viewer as mjv          # ✓ 必需（可视化用）
import numpy as np                   # ✓ 必需
from scipy.spatial.transform import Rotation  # ✓ 必需
```

**motion_retargeting/retarget.py 导入：**
```python
import numpy as np                   # ✓ 必需
from scipy.spatial.transform import Rotation  # ✓ 必需
from motion_retargeting.utils.mapped_ik import MappedIK  # 必需（如果使用完整 retargeting）
```

**motion_retargeting/wbik_solver.py 导入（关键！）：**
```python
import numpy as np                   # ✓ 必需
import qpsolvers                     # ✓ 必需（QP求解）
import pinocchio as pin              # ✓ 必需（机器人运动学）
import quaternion                    # ✓ 必需（四元数计算）
import pink                          # ✓ 必需（IK求解框架）
from pink import solve_ik            # IK求解函数
from pink.limits import ...          # 关节限制
from pink.tasks import ...           # IK任务定义
```

**motion_retargeting/renderer.py 导入：**
```python
from motion_retargeting.utils.mujoco.renderer import MujocoRenderer  # 使用 mujoco
```

---

## 2. 版本约束关系

### 2.1 Python 版本
**推荐：Python 3.10 或 3.12**
- `wbik_solver.py` 使用 `pinocchio` 和 `pink`，这两个库要求 Python >= 3.8
- NumPy 2.4.0 要求 Python >= 3.9
- PyTorch 2.5.1（如果使用）建议 Python 3.10-3.12
- **结论：Python 3.10（稳定）或 3.12（最新）**

### 2.2 NumPy 版本约束
| 关键依赖 | NumPy 支持 | 建议版本 | 原因 |
|--------|-----------|--------|------|
| scipy | 1.x, 2.x | 1.26.4 | 向后兼容性好 |
| pinocchio | 1.x, 2.x | 1.26.4 | C++ 扩展 ABI 兼容 |
| pink | 1.x, 2.x | 1.26.4 | 依赖 pinocchio |
| numpy | N/A | 1.26.4 | 避免 ABI 问题 |

**关键点：** NumPy 1.x 和 2.x 的 ABI（应用二进制接口）**不兼容**。如果安装的是 NumPy 2.x，pinocchio/pink 的 C++ 扩展会加载失败。

**结论：NumPy = 1.26.4（固定版本）**

### 2.3 PyTorch 版本（如果未来需要）
| 包 | 依赖 | 说明 |
|---|------|------|
| torch | CUDA 11.8 | 当前 dry-run 包括 pytorch-2.5.1-cuda11.8 |
| torchvision | torch 2.5.1 | 需要版本匹配 |
| torchaudio | torch 2.5.1 | 需要版本匹配 |

**结论：PyTorch 2.5.1（如需要）**

### 2.4 MuJoCo 版本
**当前使用：** `import mujoco as mj`（最新 Python API，版本 >= 2.3.0）

从 `vis_robot_motion.py` 看：
```python
import mujoco as mj              # 新版 Python API
import mujoco.viewer as mjv      # 新版 viewer API
```

这表示使用的是 **MuJoCo 2.3.0+** 的 Python API（而非旧的 dm_control）。

**结论：MuJoCo >= 2.3.0（推荐 3.x 最新版）**

### 2.5 关键 robotics 库版本

| 库 | 版本 | 用途 | 兼容性要点 |
|----|------|------|----------|
| pinocchio | 2.x | 机器人运动学模型构建 | 需要 NumPy 1.26.4（C++ 扩展）|
| pink | 0.x | IK 求解框架 | 依赖 pinocchio，需要 NumPy 兼容 |
| qpsolvers | 4.x | QP 问题求解 | 相对独立，但需要 numpy |
| scipy | 1.11+ | 旋转/科学计算 | 与 numpy 版本应匹配 |

**关键观察：** 所有 robotics 库都**强烈依赖 NumPy 1.x**（因为 C++ 扩展的 ABI）。

---

## 3. 推荐环境配置

### 场景 A：转换 BVH（fallback 模式）— 最小依赖
```
python=3.10
numpy=1.26.4
scipy>=1.11
mujoco>=2.3.0  （仅用于可视化）
```
**特点：** 无需 PyTorch、pinocchio、pink，运行快，占用空间小（~500MB）
**缺点：** BVH 动作可能不够自然（使用简单 joint 映射）

### 场景 B：转换 BVH（完整 retargeting）— 推荐配置
```
python=3.10              # 或 3.12
numpy=1.26.4             # 固定！避免 ABI 问题
scipy>=1.11              # 旋转计算
pinocchio>=2.6.0         # 机器人运动学
pink>=0.7.0              # IK 求解
qpsolvers>=4.0           # QP 求解器
mujoco>=2.3.0            # 可视化
matplotlib               # 骨架可视化
```
**特点：** 完整重定目标管道，动作自然，结果高质量
**缺点：** 依赖复杂，环境配置时间长（~2GB）

### 场景 C：包含 GPU PyTorch 支持
```
python=3.10 或 3.12
numpy=1.26.4
scipy>=1.11
pinocchio>=2.6.0
pink>=0.7.0
qpsolvers>=4.0
pytorch::pytorch=2.5.1   # 仅当需要神经网络时
pytorch::pytorch-cuda=11.8
pytorch::torchvision=0.20.1
pytorch::torchaudio=2.5.1
mujoco>=2.3.0
```
**特点：** 支持 GPU 加速的未来扩展
**缺点：** 环境体积最大（~4GB），GPU 驱动兼容性要求严格

---

## 4. 为什么之前的离线安装失败了？

### 问题 1：NumPy ABI 不匹配
- 下载的 `numpy-2.4.0-py312` + `pinocchio` 的 C++ 扩展不兼容
- **解决：** 锁定 `numpy=1.26.4`，重新下载

### 问题 2：缺少 CUDA runtime 库
- `pytorch-2.5.1-cuda11.8` 需要 `libcudart.so.11` 等系统库
- 离线包清单中未包含完整的 CUDA toolkit
- **解决：** 
  - 方案 A：使用在线 conda（推荐，自动解析依赖）
  - 方案 B：下载完整 CUDA toolkit（~5GB+ 额外下载）

### 问题 3：Python 版本不匹配
- 脚本装了 `pytorch-2.5.1-py3.12` 但环境是 Python 3.10
- **解决：** 创建 Python 3.12 环境并重新安装

### 问题 4：缺少 typing_extensions 等依赖
- PyTorch 运行时依赖的 transitive dependencies 未被完整包含
- **解决：** conda 的自动依赖解析（在线模式）

---

## 5. 最简单的快速修复方案（推荐现在立即执行）

### 对于当前的机器（有网络）：
```bash
# 1. 清理旧环境
conda env remove -n robots_env -y

# 2. 从 conda-forge 创建完整环境（自动处理所有依赖）
conda create -n robots_env -y -c conda-forge \
  python=3.10 \
  numpy=1.26.4 \
  scipy \
  pinocchio \
  pink \
  qpsolvers \
  mujoco \
  matplotlib

# 3. 激活并验证
conda activate robots_env

# 4. 运行转换脚本
python /home/jeff/Codes/Robots/convert_bvh_to_pickle.py

# 5. 可视化结果
python /home/jeff/Codes/Robots/src/vis_robot_motion.py \
  --motion-file output/g1/Geely\ test-001\(1\).pkl
```

**预期时间：** ~15 分钟（自动下载依赖）
**预期结果：** 完整 retargeting 管道工作，动作高质量

---

## 6. 官方版本说明（根据项目代码推断）

基于代码中的导入和调用模式：

| 库 | 版本 | 备注 |
|----|------|------|
| **Python** | 3.10 LTS / 3.12 | 项目支持两个版本 |
| **NumPy** | 1.26.4 | **固定**（ABI 兼容性） |
| **SciPy** | 1.11+ | 与 numpy 版本匹配 |
| **pinocchio** | 2.6.0+ | C++ 库，需要编译或预编译包 |
| **pink** | 0.7.0+ | IK 框架，依赖 pinocchio |
| **MuJoCo** | 2.3.3+ / 3.x | Python API（新版） |
| **PyTorch** | 2.5.1（可选） | 仅当需要 GPU 时 |
| **CUDA** | 11.8（如需 GPU） | 驱动兼容性很关键 |

---

## 7. 后续建议

1. **立即执行：** 使用上面的在线 conda 命令快速构建环境
2. **测试转换：** 运行 BVH 转换，验证 retargeting 工作
3. **记录环境：** 执行 `conda list > robots_env_lockfile.txt`，保存配置
4. **如需离线安装：** 使用生成的 lockfile，按 pin 版本重新下载全套依赖

---

## 附录：核心导入依赖链流程图

```
convert_bvh_to_pickle.py
  ├── numpy (1.26.4)
  ├── scipy (1.11+)
  └── motion_retargeting.retarget (if import succeeds)
      └── WBIKSolver
          ├── pinocchio (2.6+) → numpy (1.26.4!)
          ├── pink (0.7+) → pinocchio → numpy (1.26.4!)
          ├── qpsolvers (4.0+) → numpy
          └── quaternion → numpy

vis_robot_motion.py
  ├── numpy (1.26.4)
  ├── scipy (1.11+)
  └── mujoco (2.3.3+ / 3.x)
```

**关键观察：** 所有路径都汇聚到 **numpy=1.26.4**，这是整个环境的 ABI 基石。

