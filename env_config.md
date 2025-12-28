
  python=3.10 \
  pinocchio=2.6.10 \
  eigenpy=2.7.14 \
  pin-pink=3.5.0 \
  numpy=1.26.4 \
  scipy \
  qpsolvers \
  mujoco=2.3.7 \
  pytorch::pytorch=2.5.1 \
  pytorch::pytorch-cuda=12.4 \
  pytorch::torchvision \
  pytorch::torchaudio \
 
eigenpy 2.7.14 或 2.9.2 应该与 pinocchio 2.6.10 兼容。让我重建环境：

最终诊断和解决方案：

问题	原因	解决方案
pinocchio 导入失败（undefined symbol）	eigenpy 3.1.0 与 pinocchio 2.6.17 ABI不兼容	使用匹配的eigenpy 2.7.14
pinocchio 导入不到	环境包不兼配	从零重建 robots_env，使用协调的版本
pink API问题	可能与pinocchio版本有关	使用pinocchio 2.6.10（稳定版本）

版本关系一览表
组件	推荐版本	关键原因
Python	3.10 或 3.12	两个版本都支持，3.10 更稳定
NumPy	1.26.4（固定！）	pinocchio/pink 的 C++ 扩展 ABI 不兼容 NumPy 2.x
SciPy	1.11+	与 NumPy 1.26.4 版本匹配
MuJoCo	2.3.3+ 或 3.x	Python API（新版），用于可视化
pinocchio	2.6.0+	机器人运动学库，需要 NumPy 1.26.4
pink	0.7.0+	IK 求解框架，需要 pinocchio + NumPy 1.26.4
qpsolvers	4.0+	二次规划求解
PyTorch	2.5.1（可选）	仅当需要 GPU 加速时
