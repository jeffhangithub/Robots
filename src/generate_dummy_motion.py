
import mujoco as mj
import numpy as np
import pickle
import os

def generate_dummy_motion(xml_path, output_path):
    print(f"Loading model from {xml_path}")
    model = mj.MjModel.from_xml_path(xml_path)
    
    # Motion parameters
    fps = 30
    duration_sec = 2
    num_frames = fps * duration_sec
    
    # Dimensions
    # model.nq is total coordinates (including 7 for free joint)
    # model.nv is total degrees of freedom (6 for free joint)
    # The visualization script expects:
    # root_pos: (N, 3)
    # root_rot: (N, 4)  # scalar first (w, x, y, z) based on vis script usage, wait vis script says:
    # motion_root_rot = motion_data["root_rot"]#[:, [3, 0, 1, 2]]  # from xyzw to wxyz
    # And later: self.data.qpos[3:7] = root_rot
    # MuJoCo qpos for quaternion is [w, x, y, z] (scalar first).
    
    # Let's verify what the vis script expects.
    # In vis_robot_motion.py:
    # line 35: motion_root_rot = motion_data["root_rot"] # commented out [:, [3, 0, 1, 2]]
    # line 155: self.data.qpos[3:7] = root_rot
    # So it expects w, x, y, z.
    
    # dof_pos: (N, num_joints) 
    # The script calculates model.nv - 6 for dof_pos columns.
    n_dof = model.nv - 6
    
    print(f"Generating {num_frames} frames for {n_dof} DOFs")
    
    # Generate data
    root_pos = np.zeros((num_frames, 3))
    # Move slightly in X
    root_pos[:, 0] = np.linspace(0, 1, num_frames)
    root_pos[:, 2] = 1.0 # Lift up slightly
    
    root_rot = np.zeros((num_frames, 4))
    root_rot[:, 0] = 1.0 # w=1, identity quaternion
    
    dof_pos = np.zeros((num_frames, n_dof))
    # Make some joints move
    # e.g. sin wave for all joints
    t = np.linspace(0, 2*np.pi, num_frames)
    for i in range(n_dof):
        dof_pos[:, i] = 0.5 * np.sin(t + i*0.1)

    # Local body pos (N, n_bodies, 3)
    # The script uses separate list of body names.
    # We can just extract all body names from model.
    body_names = []
    # Skip world frame (body 0)
    for i in range(1, model.nbody):
        body_names.append(model.body(i).name)
        
    local_body_pos = np.zeros((num_frames, len(body_names), 3))
    
    motion_data = {
        "fps": fps,
        "root_pos": root_pos,
        "root_rot": root_rot,
        "dof_pos": dof_pos,
        "local_body_pos": local_body_pos,
        "link_body_list": body_names
    }
    
    with open(output_path, "wb") as f:
        pickle.dump(motion_data, f)
    
    print(f"Saved dummy motion to {output_path}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Assuming running from repo root or src
    # Try to find the h1 xml
    xml_path = os.path.join(current_dir, "motion_retargeting/robots/h1_2/h1_2.xml")
    
    if not os.path.exists(xml_path):
        # Fallback to absolute path known from search
        xml_path = "/Users/jeff/Documents/文稿 - Han的MacBook Air (3159)/GitHub/Robots/src/motion_retargeting/robots/h1_2/h1_2.xml"
        
    output_path = "dummy_motion.pkl"
    generate_dummy_motion(xml_path, output_path)
