# 如何执行基于 Motion_Retargeting 的转换

## 当前状态

✅ **已成功安装的环境：**
- Python 3.10.19
- NumPy 1.26.4
- Pinocchio 3.8.0
- Pink 3.5.0
- PyTorch 2.5.1

⚠️ **Fallback 模式：** 当前使用

---

## 两种转换模式对比

### 模式 1: Fallback 解析器（现在激活）

**用途：** 快速预览、低依赖

**工作原理：**
```python
from motion_retargeting.retarget.retarget import BVHRetarget  # 导入失败
# → 自动降级到 fallback 解析器
# → 简单 BVH 关节直接映射到 G1 DOF
```

**启动命令：**
```bash
bash /home/jeff/Codes/Robots/activate_robots_env.sh
python /home/jeff/Codes/Robots/convert_bvh_to_pickle.py
```

**输出示例：**
```
⚠️  无法导入 motion_retargeting: No module named '...'
📖 读取 BVH 文件: /home/jeff/Codes/Robots/data/Geely test-001(1).bvh
✅ 解析完成，找到 11326 帧，53 个关节
✅ 已保存到: /home/jeff/Codes/Robots/output/g1/Geely test-001(1).pkl
```

**特点：**
- ✓ 速度快（< 1 分钟）
- ✓ 依赖少
- ✗ 动作可能不够自然
- ✗ 可能出现关节越限

---

### 模式 2: 完整 Retargeting（IK 求解）

**用途：** 高质量运动转换、物理约束感知

**工作原理：**
```python
from motion_retargeting.retarget.retarget import BVHRetarget  # 导入成功
# → 使用 pinocchio 构建机器人模型
# → 用 pink IK 求解器调整 BVH 动作
# → 自动修正关节越限、实现物理约束
```

**需要的额外包：**
- `quaternion` 或 `numpy-quaternion` ✓ 已装
- `robot_descriptions` ✓ 已装（但首次导入有网络下载）
- 正常网络连接（首次时需要 clone mujoco_menagerie）

---

## 启用完整 Retargeting 的步骤

### 步骤 1: 确保网络连接
完整 retargeting 的第一次导入会自动从 GitHub 克隆 MuJoCo Menagerie (~1.5GB)：
```
Cloning https://github.com/deepmind/mujoco_menagerie.git...
```

**如果网络不稳定，可能需要：**
```bash
# 手动预先克隆（在有网的机器上）
git clone https://github.com/deepmind/mujoco_menagerie.git \
  ~/.cache/robot_descriptions/mujoco_menagerie
```

### 步骤 2: 运行转换

一旦网络问题解决，执行：
```bash
bash /home/jeff/Codes/Robots/activate_robots_env.sh
python /home/jeff/Codes/Robots/run_full_retargeting.sh
```

### 步骤 3: 观察输出

**预期输出（完整模式）：**
```
✅ motion_retargeting 完整管道已就绪！
🔄 运行 BVH → pickle 转换（完整 retargeting）...
   Frame 0 / 11326  [IK求解进度...]
   Frame 1000 / 11326
   ...
✅ 转换完成！
```

**耗时：** 5-15 分钟（取决于机器性能）

---

## 快速执行命令

### 现在立即可用（Fallback）
```bash
bash /home/jeff/Codes/Robots/activate_robots_env.sh && \
python /home/jeff/Codes/Robots/convert_bvh_to_pickle.py
```

### 当网络就绪后（完整 Retargeting）
```bash
bash /home/jeff/Codes/Robots/activate_robots_env.sh && \
python /home/jeff/Codes/Robots/run_full_retargeting.sh
```

### 检查当前模式
```bash
bash /home/jeff/Codes/Robots/activate_robots_env.sh && \
python /home/jeff/Codes/Robots/analyze_conversion.py
```

---

## 转换结果对比

### Fallback 模式结果（已生成）
```
文件：output/g1/Geely test-001(1).pkl
帧数：11326
DOF：37
关节数：53
运动范围：
  X轴：[-1.241, 0.000] (行走距离)
  Y轴：[0.871, 0.913] (侧向稳定)
  Z轴：[0.000, 4.023] (垂直变化)
```

### Retargeting 结果（待生成）
- 更自然的动作过渡
- 自动矫正关节越限
- 更好的物理约束满足

---

## 故障排查

### 问题：导入失败 `No module named 'robot_descriptions'`
**原因：** 包未安装或网络克隆失败
**解决：**
```bash
bash /home/jeff/Codes/Robots/activate_robots_env.sh
python -m pip install robot_descriptions
```

### 问题：Git 克隆超时
**原因：** 网络连接不稳定，MuJoCo Menagerie (~1.5GB) 下载失败
**解决：**
```bash
# 手动预下载（用代理或更快的网络）
git clone --depth 1 https://github.com/deepmind/mujoco_menagerie.git \
  ~/.cache/robot_descriptions/mujoco_menagerie
```

### 问题：内存不足
**原因：** IK 求解器计算量大
**解决：** 
- 使用更小的 BVH 文件测试
- 或减少帧率（修改脚本）

---

## 推荐工作流

1. **快速验证（Fallback）** — 2 分钟
   ```bash
   bash /home/jeff/Codes/Robots/activate_robots_env.sh && \
   python /home/jeff/Codes/Robots/convert_bvh_to_pickle.py && \
   python /home/jeff/Codes/Robots/src/vis_robot_motion.py \
     --xml_path /home/jeff/Codes/Robots/src/motion_retargeting/robots/g1/urdf/g1.xml \
     --robot_motion_path '/home/jeff/Codes/Robots/output/g1/Geely test-001(1).pkl'
   ```

2. **质量优化（Retargeting）** — 10 分钟
   ```bash
   bash /home/jeff/Codes/Robots/activate_robots_env.sh && \
   python /home/jeff/Codes/Robots/run_full_retargeting.sh && \
   python /home/jeff/Codes/Robots/src/vis_robot_motion.py \
     --xml_path /home/jeff/Codes/Robots/src/motion_retargeting/robots/g1/urdf/g1.xml \
     --robot_motion_path '/home/jeff/Codes/Robots/output/g1/Geely test-001(1).pkl'
   ```

3. **对比分析** — 1 分钟
   ```bash
   bash /home/jeff/Codes/Robots/activate_robots_env.sh && \
   python /home/jeff/Codes/Robots/analyze_conversion.py
   ```

---

## 总结

| 特性 | Fallback | Retargeting |
|------|----------|-------------|
| 启动 | 立即 | 需要网络首次初始化 |
| 速度 | 快 (< 1分钟) | 慢 (5-15分钟) |
| 质量 | 基础 | 高（IK约束） |
| 依赖 | 少 | 多（pinocchio, pink） |
| 用途 | 预览 | 生产 |

**建议：** 先用 Fallback 快速验证，再用 Retargeting 生成最终高质量数据。
