import numpy as np
import omnigibson as og
from PIL import Image
import pickle
import cv2  
from tqdm import tqdm

def main():
    cfg = {
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": "Benevolence_1_int",
        },
    }
    # Load the environment
    env = og.Environment(configs=cfg)
    cam = og.sim.viewer_camera
    camera_poses = pickle.load(open("cam_trajectory.pkl", "rb"))
    og.sim.viewer_width = 1024
    og.sim.viewer_height = 1024
    og.sim.step()
    from IPython import embed; embed()
    normal_video = cv2.VideoWriter("normal.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 24, (1024, 1024))  
    depth_video = cv2.VideoWriter("depth.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 24, (1024, 1024))  
    for pose in tqdm(camera_poses):
        cam.set_position_orientation(pose[0], pose[1])
        og.sim.step()
        obs = cam.get_obs()
        normal = ((obs["normal"] * 0.5 + 0.5) * 255).astype(np.uint8)[:, :, [1, 0, 2]]
        depth = obs["depth_linear"]
        depth = np.clip(depth, 0, 10) / 10 * 255
        depth = depth.astype(np.uint8)
        depth = np.array(Image.fromarray(depth).convert("RGB"))
        normal_video.write(normal) 
        depth_video.write(depth)
    normal_video.release() 
    depth_video.release()

if __name__ == "__main__":
    main()
