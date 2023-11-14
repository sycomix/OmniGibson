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

    og.sim.viewer_width = 1080
    og.sim.viewer_height = 1080
    # og.sim.viewer_camera.set_position_orientation(
    #     position=[0.4270983 , 12.22620911,  1.21585457],
    #     orientation=[-0.34374667,  0.55816492,  0.64301759, -0.39600319],
    # )
    og.sim.viewer_camera.set_position_orientation(
        position=[0.74713144, -1.19452407,  0.74987465],
        orientation=[0.61173465, 0.33341926, 0.34330987, 0.62988146],
    )
    for i in range(10):
        og.sim.step()
    from IPython import embed; embed()
    obs = og.sim.viewer_camera.get_obs()
    Image.fromarray(obs["rgb"]).save("rgb.png")
    depth = obs["depth_linear"]
    depth = depth / depth.max() * 255
    depth = depth.astype(np.uint8)
    Image.fromarray(depth).convert("RGB").save("depth.png")
    Image.fromarray(((obs["normal"] * 0.5 + 0.5) * 255).astype(np.uint8)).save("normal.png")
    seg = obs["seg_instance"]
    seg = np.rollaxis(np.array([(128 + seg) % 255, (np.sin(seg) * 255) % 255, (np.cos(seg) * 255) % 255]), 0, 3)
    seg = seg.astype(np.uint8)
    Image.fromarray(seg).save("seg.png")

    while True:
        print(og.sim.viewer_camera.get_position_orientation())
        env.step(None)

if __name__ == "__main__":
    main()
