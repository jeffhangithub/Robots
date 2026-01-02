import sys
import os

print(f"Python Executable: {sys.executable}")

# 1. 检查 NumPy
try:
    import numpy
    print(f"NumPy Version: {numpy.__version__}", end=" ")
    if numpy.__version__.startswith("1."):
        print("✅ (Compatible)")
    else:
        print("❌ (CRITICAL: Must be 1.x)")
except ImportError:
    print("❌ NumPy missing")

# 2. 检查 Pinocchio
try:
    import pinocchio
    try:
        model = pinocchio.Model()
        print(f"Pinocchio: {pinocchio.__file__} ✅")
    except AttributeError:
        print(f"Pinocchio: {pinocchio.__file__} ❌ (Wrong package installed!)")
except ImportError:
    print("❌ Pinocchio missing")

# 3. 检查 MuJoCo
try:
    import mujoco
    print(f"MuJoCo Version: {mujoco.__version__}", end=" ")
    if hasattr(mujoco, 'MjSpec'):
        print("✅ (API Compatible)")
    else:
        print("❌ (Too old, need 3.x)")
except ImportError:
    print("❌ MuJoCo missing")

# 4. 检查 PyTorch & MKL
try:
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"PyTorch Path: {torch.__file__}")
    if "envs/robots_env" in torch.__file__:
        print("PyTorch Source: Conda ✅")
    else:
        print("PyTorch Source: System ❌ (Potential Conflict)")
        
    # 测试 MKL
    t = torch.zeros(1)
    print("MKL Linking: OK ✅")
except ImportError:
    print("❌ PyTorch missing")
except Exception as e:
    print(f"❌ MKL/Linking Error: {e}")