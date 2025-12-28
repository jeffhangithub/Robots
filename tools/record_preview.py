#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from vis_robot_motion import RobotMotionViewer, load_robot_motion
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--xml_path', required=True)
parser.add_argument('--motion_pkl', required=True)
parser.add_argument('--video_path', required=True)
parser.add_argument('--frames', type=int, default=300)
parser.add_argument('--motion_fps', type=int, default=30)
args = parser.parse_args()

motion_data, motion_fps, motion_root_pos, motion_root_rot, motion_dof_pos, motion_local_body_pos, motion_link_body_list = load_robot_motion(args.motion_pkl)

env = RobotMotionViewer(xml_path=args.xml_path, motion_fps=args.motion_fps, record_video=True, video_path=args.video_path, camera_follow=True)

max_frames = min(args.frames, len(motion_root_pos))
for i in range(max_frames):
    root_pos = motion_root_pos[i].tolist()
    root_rot = motion_root_rot[i].tolist()
    dof_pos = motion_dof_pos[i].tolist()
    if len(dof_pos) != env.model.nv - 6:
        dof_pos = dof_pos[: env.model.nv - 6]
    env.step(root_pos, root_rot, dof_pos, rate_limit=True, follow_camera=True)

env.close()
print('Saved preview video to', args.video_path)
