import numpy as np
import omnigibson as og
from PIL import Image


def main():
    cfg = {
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": "Rs_int",
            # "not_load_object_categories": ["ceilings"],
        },
    }
    # Load the environment
    env = og.Environment(configs=cfg)

    # og.sim.viewer_width = 2160
    # og.sim.viewer_height = 2880
    og.sim.viewer_camera.set_position_orientation(
        position=[1.72908523, -0.52173273,  1.06228891],
        orientation=[0.55076158, 0.37545521, 0.41989022, 0.61594421],
    )
    for i in range(10):
        og.sim.step()
    obs = og.sim.viewer_camera.get_obs()
    rgb = obs["rgb"]
    depth = obs["depth"]
    seg = obs["seg_instance"]
    normal = obs["normal"]
    while True:
        env.step(None)

if __name__ == "__main__":
    main()
