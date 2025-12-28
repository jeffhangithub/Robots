import time
import argparse
import numpy as np

import mujoco
from mujoco import viewer

from fairmotion.data import bvh


def make_xml(joint_names, radius=0.02):
    # 用 mocap body 放一堆 sphere，运行时每帧改 mocap_pos 即可
    bodies = []
    for name in joint_names:
        safe = name.replace(" ", "_").replace(":", "_").replace("/", "_")
        bodies.append(f"""
        <body name="{safe}" mocap="true">
          <geom type="sphere" size="{radius}" rgba="0.9 0.9 0.9 1"/>
        </body>
        """)
    xml = f"""
<mujoco model="bvh_skeleton">
  <option timestep="0.016"/>
  <visual>
    <global offwidth="1280" offheight="720"/>
  </visual>
  <worldbody>
    {''.join(bodies)}
  </worldbody>
</mujoco>
"""
    return xml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bvh", required=True, help="path to .bvh")
    ap.add_argument("--scale", type=float, default=0.01, help="position scale (BVH often in cm)")
    ap.add_argument("--axis", type=str, default="xzy",
                    help="axis mapping from BVH(x,y,z) -> MuJoCo(x,?,?) e.g. xzy means (x,z,y)")
    ap.add_argument("--fps", type=float, default=60.0)
    args = ap.parse_args()

    motion = bvh.load(args.bvh)  # fairmotion BVH loader :contentReference[oaicite:4]{index=4}
    mats = motion.to_matrix()    # (T, J, 4, 4)  :contentReference[oaicite:5]{index=5}

    # 关节名顺序
    joint_names = [j.name for j in motion.skel.joints]

    # 位置在矩阵的第 4 行前三个（fairmotion README 的示例暗示这种布局）:contentReference[oaicite:6]{index=6}
    pos = mats[:, :, 3, :3].copy()  # (T,J,3)

    # 轴映射：MuJoCo 默认 z-up；BVH 很多是 y-up，所以默认 xzy（x,z,y）更常用
    idx = {"x": 0, "y": 1, "z": 2}
    order = [idx[c] for c in args.axis]
    pos = pos[:, :, order]

    # 缩放 + 去掉初始 root 偏移，让骨架在原点附近
    pos *= args.scale
    root0 = pos[0, 0].copy()
    pos -= root0

    xml = make_xml(joint_names, radius=0.02)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    # mocap bodies 的顺序就是 worldbody 里定义的顺序
    # MuJoCo: data.mocap_pos shape = (nmocap, 3)
    nmocap = model.nmocap
    assert nmocap == len(joint_names), (nmocap, len(joint_names))

    dt = 1.0 / args.fps
    with viewer.launch_passive(model, data) as v:
        t = 0
        while v.is_running():
            frame = t % pos.shape[0]
            data.mocap_pos[:] = pos[frame]
            mujoco.mj_step(model, data)
            v.sync()
            time.sleep(dt)
            t += 1


if __name__ == "__main__":
    main()
