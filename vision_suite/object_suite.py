import numpy as np
import yaml
import omnigibson as og
from omnigibson.renderer_settings.renderer_settings import RendererSettings

def main():
    # Create the environment
    cfg = yaml.load(open("vision_suite/vision_suite_objects.yaml", "r"), Loader=yaml.FullLoader)
    env = og.Environment(configs=cfg)
    env.reset()

    # Set camera to appropriate viewing pose
    og.sim.viewer_camera.set_position_orientation(
        position=np.array([-0.03396363, -0.62034724, 3.03471115]),
        orientation=np.array([0.4064837, 0.00115538, 0.00259694, 0.91365361]),
    )
    og.sim.viewer_width = 2160
    og.sim.viewer_height = 2700

    fridge = env.scene.object_registry("name", "fridge")
    cab1 = env.scene.object_registry("name", "bottom_cabinet_1")
    washer = env.scene.object_registry("name", "washer")
    fridge.set_joint_positions(np.array([1.57, 0]), drive=False)
    cab1.set_joint_positions(np.array([1, 0.2]), drive=False)
    washer.set_joint_positions(np.array([1.5]), drive=False)

    RendererSettings().set_current_renderer("Interactive (Path Tracing)")
    
    for _ in range(100000):
        env.step(None)

    # Shut down at the end
    og.shutdown()


if __name__ == "__main__":
    main()
