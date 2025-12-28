import argparse
import os
from tqdm import tqdm
import pickle
import os
import time
import mujoco as mj
import mujoco.viewer as mjv
import imageio
from scipy.spatial.transform import Rotation as R
import numpy as np
from rich import print

class RateLimiter:
    def __init__(self, frequency, warn=False):
        self.period = 1.0 / frequency
        self.last_time = time.time()
    
    def sleep(self):
        current_time = time.time()
        elapsed = current_time - self.last_time
        sleep_time = max(0, self.period - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)
        self.last_time = time.time()

def load_robot_motion(motion_file):
    """
    Load robot motion data from a pickle file.
    """
    with open(motion_file, "rb") as f:
        motion_data = pickle.load(f)
        motion_fps = motion_data["fps"]
        motion_root_pos = motion_data["root_pos"]
        motion_root_rot = motion_data["root_rot"]#[:, [3, 0, 1, 2]]  # from xyzw to wxyz
        motion_dof_pos = motion_data["dof_pos"]
        motion_local_body_pos = motion_data["local_body_pos"]
        motion_link_body_list = motion_data["link_body_list"]
    return motion_data, motion_fps, motion_root_pos, motion_root_rot, motion_dof_pos, motion_local_body_pos, motion_link_body_list


def draw_frame(
    pos,
    mat,
    v,
    size,
    joint_name=None,
    orientation_correction=R.from_euler("xyz", [0, 0, 0]),
    pos_offset=np.array([0, 0, 0]),
):
    rgba_list = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]
    for i in range(3):
        geom = v.user_scn.geoms[v.user_scn.ngeom]
        mj.mjv_initGeom(
            geom,
            type=mj.mjtGeom.mjGEOM_ARROW,
            size=[0.01, 0.01, 0.01],
            pos=pos + pos_offset,
            mat=mat.flatten(),
            rgba=rgba_list[i],
        )
        if joint_name is not None:
            geom.label = joint_name  # 这里赋名字
        fix = orientation_correction.as_matrix()
        mj.mjv_connector(
            v.user_scn.geoms[v.user_scn.ngeom],
            type=mj.mjtGeom.mjGEOM_ARROW,
            width=0.005,
            from_=pos + pos_offset,
            to=pos + pos_offset + size * (mat @ fix)[:, i],
        )
        v.user_scn.ngeom += 1


class RobotMotionViewer:
    def __init__(
        self,
        xml_path,
        robot_base="base_link",
        camera_follow=True,
        motion_fps=30,
        transparent_robot=0,
        cam_distance=3.0,
        # video recording
        record_video=False,
        video_path=None,
        video_width=640,
        video_height=480,
    ):

        self.xml_path = xml_path
        self.model = mj.MjModel.from_xml_path(str(self.xml_path))
        self.data = mj.MjData(self.model)
        self.robot_base = robot_base
        self.viewer_cam_distance = cam_distance
        mj.mj_step(self.model, self.data)

        self.motion_fps = motion_fps
        self.rate_limiter = RateLimiter(frequency=self.motion_fps, warn=False)
        self.camera_follow = camera_follow
        self.record_video = record_video

        self.viewer = mjv.launch_passive(
            model=self.model,
            data=self.data,
#           show_left_ui=False, #Mujoco 2 don't have this arg yet
#           show_right_ui=False, #Mujoco 2 don't have this arg yet
        )

        self.viewer.opt.flags[mj.mjtVisFlag.mjVIS_TRANSPARENT] = transparent_robot

        if self.record_video:
            assert video_path is not None, "Please provide video path for recording"
            self.video_path = video_path
            video_dir = os.path.dirname(self.video_path)

            if not os.path.exists(video_dir):
                os.makedirs(video_dir)
            self.mp4_writer = imageio.get_writer(self.video_path, fps=self.motion_fps)
            print(f"Recording video to {self.video_path}")

            # Initialize renderer for video recording
            self.renderer = mj.Renderer(self.model, height=video_height, width=video_width)

    def step(
        self,
        # robot data
        root_pos,
        root_rot,
        dof_pos,
        # human data
        human_motion_data=None,
        show_human_body_name=False,
        # scale for human point visualization
        human_point_scale=0.1,
        # human pos offset add for visualization
        human_pos_offset=np.array([0.0, 0.0, 0]),
        # rate limit
        rate_limit=True,
        follow_camera=True,
    ):
        """
        by default visualize robot motion.
        also support visualize human motion by providing human_motion_data, to compare with robot motion.

        human_motion_data is a dict of {"human body name": (3d global translation, 3d global rotation)}.

        if rate_limit is True, the motion will be visualized at the same rate as the motion data.
        else, the motion will be visualized as fast as possible.
        """
        # 减去第第一帧的位置
        self.data.qpos[0] = root_pos[0] +4.809377899566405
        self.data.qpos[1] = root_pos[1] - 3.105833551937692
        self.data.qpos[2] = root_pos[2] + 0.05 #调整地面高度，202512241806
        #self.data.qpos[2] = root_pos[2] -1.0257531046377046
        self.data.qpos[3:7] = root_rot  # quat need to be scalar first! for mujoco
        self.data.qpos[7:] = dof_pos

        mj.mj_forward(self.model, self.data)

        if follow_camera:
            # Find base body ID
            base_body_id = -1
            for i in range(self.model.nbody):
                if self.model.body(i).name == self.robot_base:
                    base_body_id = i
                    break
            
            if base_body_id != -1:
                self.viewer.cam.lookat = self.data.xpos[base_body_id]
            else:
                # Fallback to body 1 (usually the root body)
                self.viewer.cam.lookat = self.data.xpos[1]
                
            self.viewer.cam.distance = self.viewer_cam_distance
            self.viewer.cam.elevation = -10  # 正面视角，轻微向下看
            # self.viewer.cam.azimuth = 180    # 正面朝向机器人

        if human_motion_data is not None:
            # Clean custom geometry
            self.viewer.user_scn.ngeom = 0
            # Draw the task targets for reference
            for human_body_name, (pos, rot) in human_motion_data.items():
                draw_frame(
                    pos,
                    R.from_quat(rot, scalar_first=True).as_matrix(),
                    self.viewer,
                    human_point_scale,
                    pos_offset=human_pos_offset,
                    joint_name=human_body_name if show_human_body_name else None,
                )

        self.viewer.sync()
        if rate_limit is True:
            self.rate_limiter.sleep()

        if self.record_video:
            # Use renderer for proper offscreen rendering
            self.renderer.update_scene(self.data, camera=self.viewer.cam)
            img = self.renderer.render()
            self.mp4_writer.append_data(img)

    def close(self):
        self.viewer.close()
        time.sleep(0.5)
        if self.record_video:
            self.mp4_writer.close()
            print(f"Video saved to {self.video_path}")


def find_base_body_name(model):
    """Find a suitable base body name from the model."""
    # Common base body names in robotics
    common_base_names = ["base_link", "base", "root", "torso", "body", "trunk"]
    
    for body_id in range(model.nbody):
        body_name = model.body(body_id).name
        if body_name in common_base_names:
            return body_name
    
    # If no common name found, return the first non-world body
    for body_id in range(1, model.nbody):  # Skip world body (id=0)
        return model.body(body_id).name
    
    return "base_link"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize robot motion from pickle file")
    
    # Required arguments
    parser.add_argument("--xml_path", type=str, required=True, 
                        help="Path to the robot XML file")
    parser.add_argument("--robot_motion_path", type=str, required=True,
                        help="Path to the robot motion pickle file")
    
    # Optional arguments
    parser.add_argument("--robot_base", type=str, default=None,
                        help="Name of the robot's base body for camera following (default: auto-detect)")
    parser.add_argument("--cam_distance", type=float, default=3.0,
                        help="Camera distance from robot (default: 3.0)")
    
    # Video recording
    parser.add_argument("--record_video", action="store_true",
                        help="Record the visualization as video")
    parser.add_argument("--video_path", type=str, default="videos/robot_motion.mp4",
                        help="Path to save the video (default: videos/robot_motion.mp4)")
    
    # Visualization options
    parser.add_argument("--transparent_robot", type=int, default=0,
                        help="Make robot transparent (0: off, 1: on)")
    parser.add_argument("--camera_follow", action="store_true",
                        help="Enable camera following the robot")
    parser.add_argument("--motion_fps", type=int, default=30,
                        help="FPS for motion playback (default: 30)")
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.xml_path):
        raise FileNotFoundError(f"XML file {args.xml_path} not found")
    
    if not os.path.exists(args.robot_motion_path):
        raise FileNotFoundError(f"Motion file {args.robot_motion_path} not found")
    
    print(f"Loading motion from: {args.robot_motion_path}")
    print(f"Loading robot model from: {args.xml_path}")
    
    # Load motion data
    motion_data, motion_fps, motion_root_pos, motion_root_rot, motion_dof_pos, motion_local_body_pos, motion_link_body_list = load_robot_motion(
        args.robot_motion_path
    )
    
    # Load model to auto-detect base body if not provided
    temp_model = mj.MjModel.from_xml_path(args.xml_path)
    if args.robot_base is None:
        args.robot_base = find_base_body_name(temp_model)
        print(f"Auto-detected base body: {args.robot_base}")
    
    # Create viewer
    env = RobotMotionViewer(
        xml_path=args.xml_path,
        robot_base=args.robot_base,
        motion_fps=args.motion_fps,
        camera_follow=args.camera_follow,
        transparent_robot=args.transparent_robot,
        cam_distance=args.cam_distance,
        record_video=args.record_video,
        video_path=args.video_path,
    )
    
    print(f"Robot base body: {args.robot_base}")
    print(f"Number of joints in model: {env.model.njnt}")
    print(f"Number of DOFs in model: {env.model.nv}")
    print(f"Number of DOFs in motion data: {motion_dof_pos.shape[1]}")
    
    # Play motion
    frame_idx = 0
    try:
        while frame_idx < len(motion_root_pos):
            # Use motion data directly (already converted to wxyz format)
            root_pos = motion_root_pos[frame_idx].tolist()
            root_rot = motion_root_rot[frame_idx].tolist()
            
            # Use DOF positions directly (assuming they match the model)
            # Note: This assumes motion_dof_pos has the same ordering as model DOFs
            # You may need to adjust this based on your specific robot
            dof_pos = motion_dof_pos[frame_idx].tolist()
            
            # Check if DOF count matches; pad or truncate to fit model
            expected = env.model.nv - 6  # subtract root pos+rot
            if len(dof_pos) != expected:
                print(f"Warning: DOF count mismatch. Model: {expected}, Motion: {len(dof_pos)}")
                if len(dof_pos) > expected:
                    dof_pos = dof_pos[:expected]
                else:
                    dof_pos = dof_pos + [0.0] * (expected - len(dof_pos))
            
            env.step(
                root_pos,
                root_rot,
                dof_pos,
                rate_limit=True,
                follow_camera=args.camera_follow,
            )
            
            frame_idx += 1
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        env.close()