#!/usr/bin/env python3
import sys
import os
import pickle
import imageio
import mujoco as mj
import numpy as np

# 参数
xml_path = '/home/jeff/Codes/Robots/src/motion_retargeting/robots/g1/urdf/g1.xml'
motion_pkl = '/home/jeff/Codes/Robots/output/g1/Geely test-001(1).pkl'
video_path = '/home/jeff/Codes/Robots/output/g1/Geely_test_bvh_preview_headless.mp4'
frames = 300
width, height = 640, 480
motion_fps = 30

with open(motion_pkl, 'rb') as f:
    motion_data = pickle.load(f)

root_pos = motion_data['root_pos']
root_rot = motion_data['root_rot']
dof_pos = motion_data['dof_pos']

model = mj.MjModel.from_xml_path(xml_path)
data = mj.MjData(model)
# renderer
renderer = mj.Renderer(model, height=height, width=width)
writer = imageio.get_writer(video_path, fps=motion_fps)

nframes = min(frames, len(root_pos))

# optional offsets from vis script
offset_x = 4.809377899566405
offset_y = -3.105833551937692
offset_z = -1.0257531046377046

for i in range(nframes):
    # set qpos
    data.qpos[0] = root_pos[i][0] + offset_x
    data.qpos[1] = root_pos[i][1] + offset_y
    data.qpos[2] = root_pos[i][2] + offset_z
    # quaternion assumed w,x,y,z in stored data
    data.qpos[3:7] = root_rot[i]
    # fill dofs
    dq = dof_pos[i]
    nv_expected = model.nv - 6
    if len(dq) != nv_expected:
        dq = dq[:nv_expected]
    data.qpos[7:7+len(dq)] = dq

    mj.mj_forward(model, data)
    renderer.update_scene(data)
    img = renderer.render()
    writer.append_data(img)

writer.close()
print('Saved headless preview to', video_path)
