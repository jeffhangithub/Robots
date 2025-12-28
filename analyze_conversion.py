#!/usr/bin/env python3
"""
对比脚本：Fallback 解析器 vs 完整 Retargeting 管道

用途：
1. 展示两种模式的导入状态
2. 比较转换后的动作数据质量
3. 说明如何启用完整 retargeting
"""

import sys
sys.path.insert(0, '/home/jeff/Codes/Robots/src')

import pickle
import numpy as np

def check_retargeting_availability():
    """检查完整 retargeting 管道是否可用"""
    print("=" * 60)
    print("依赖检查")
    print("=" * 60)
    
    checks = {
        "numpy": None,
        "scipy": None,
        "pinocchio": None,
        "pink": None,
        "quaternion": None,
        "motion_retargeting": None,
    }
    
    # NumPy
    try:
        import numpy
        checks["numpy"] = f"✓ {numpy.__version__}"
    except ImportError as e:
        checks["numpy"] = f"✗ {e}"
    
    # SciPy
    try:
        import scipy
        checks["scipy"] = f"✓ {scipy.__version__}"
    except ImportError as e:
        checks["scipy"] = f"✗ {e}"
    
    # Pinocchio
    try:
        import pinocchio
        checks["pinocchio"] = f"✓ {pinocchio.__version__}"
    except ImportError as e:
        checks["pinocchio"] = f"✗ {e}"
    
    # Pink
    try:
        import pink
        checks["pink"] = f"✓ {pink.__version__}"
    except ImportError as e:
        checks["pink"] = f"✗ {e}"
    
    # Quaternion
    try:
        import quaternion
        checks["quaternion"] = f"✓ (installed)"
    except ImportError as e:
        checks["quaternion"] = f"✗ {e}"
    
    # Motion Retargeting
    try:
        from motion_retargeting.retarget.retarget import BVHRetarget
        from motion_retargeting.config.robot.g1 import G1_BVH_CONFIG
        checks["motion_retargeting"] = "✓ 完整管道可用"
    except ImportError as e:
        checks["motion_retargeting"] = f"✗ {str(e)[:50]}..."
    
    for lib, status in checks.items():
        print(f"  {lib:20} : {status}")
    
    print()
    
    # 检查 retargeting 是否完全可用
    retargeting_available = "✓" in checks["motion_retargeting"]
    quaternion_available = "✓" in checks["quaternion"]
    pinocchio_available = "✓" in checks["pinocchio"]
    
    if retargeting_available and quaternion_available and pinocchio_available:
        print("🎯 状态：完整 Retargeting 管道可用")
        return True
    else:
        missing = []
        if not quaternion_available:
            missing.append("quaternion")
        if not pinocchio_available:
            missing.append("pinocchio")
        if not retargeting_available:
            missing.append("motion_retargeting 模块")
        
        print(f"⚠️  状态：缺失依赖 - {', '.join(missing)}")
        print(f"   将使用 Fallback 解析器")
        return False


def analyze_converted_motion(pkl_path):
    """分析转换后的运动数据"""
    print("=" * 60)
    print("运动数据分析")
    print("=" * 60)
    print(f"\n加载：{pkl_path}")
    
    try:
        with open(pkl_path, 'rb') as f:
            motion_data = pickle.load(f)
    except FileNotFoundError:
        print(f"❌ 文件不存在")
        return
    
    print("\n数据结构：")
    for key, value in motion_data.items():
        if isinstance(value, np.ndarray):
            print(f"  {key:20} : shape={value.shape}, dtype={value.dtype}")
        elif isinstance(value, (list, tuple)):
            print(f"  {key:20} : length={len(value)}")
        elif isinstance(value, (int, float)):
            print(f"  {key:20} : {value}")
        else:
            print(f"  {key:20} : {type(value).__name__}")
    
    # 统计数据
    print("\n统计信息：")
    if 'fps' in motion_data:
        print(f"  FPS：{motion_data['fps']}")
    if 'root_pos' in motion_data:
        frames = motion_data['root_pos'].shape[0]
        print(f"  帧数：{frames}")
    if 'dof_pos' in motion_data:
        dofs = motion_data['dof_pos'].shape[1] if len(motion_data['dof_pos'].shape) > 1 else 1
        print(f"  DOF：{dofs}")
    if 'link_body_list' in motion_data:
        joints = len(motion_data['link_body_list'])
        print(f"  关节数：{joints}")
    
    # 运动范围分析
    print("\n运动范围（root position）：")
    if 'root_pos' in motion_data:
        pos = motion_data['root_pos']
        for i, axis in enumerate(['X', 'Y', 'Z']):
            min_val = pos[:, i].min()
            max_val = pos[:, i].max()
            range_val = max_val - min_val
            print(f"  {axis}轴：[{min_val:7.3f}, {max_val:7.3f}] (范围: {range_val:.3f})")
    
    print()


def main():
    # 标题
    print("\n" + "=" * 60)
    print("BVH 转换对比分析")
    print("=" * 60)
    print()
    
    # 检查依赖
    retargeting_available = check_retargeting_availability()
    
    # 分析转换结果
    pkl_file = '/home/jeff/Codes/Robots/output/g1/Geely test-001(1).pkl'
    analyze_converted_motion(pkl_file)
    
    # 给出建议
    print("=" * 60)
    print("使用建议")
    print("=" * 60)
    print()
    
    if retargeting_available:
        print("✅ 完整 Retargeting 已启用")
        print("   转换使用的是物理约束感知的 IK 求解器")
        print("   结果应该更自然、更符合物理")
        print()
        print("运行转换命令：")
        print("  bash /home/jeff/Codes/Robots/run_full_retargeting.sh")
    else:
        print("⚠️  使用的是 Fallback 解析器")
        print()
        print("启用完整 Retargeting 的步骤：")
        print()
        print("1. 安装缺失的包：")
        print("   bash /home/jeff/Codes/Robots/activate_robots_env.sh")
        print("   pip install quaternion")
        print()
        print("2. 重新运行转换：")
        print("   bash /home/jeff/Codes/Robots/run_full_retargeting.sh")
        print()
        print("差异解释：")
        print("  Fallback：直接映射 BVH 关节到 G1")
        print("           快速，但可能出现不自然的动作")
        print()
        print("  Retargeting：使用 pinocchio + pink IK 求解")
        print("             调整动作以满足机器人物理约束")
        print("             结果更自然")
    
    print()


if __name__ == '__main__':
    main()
