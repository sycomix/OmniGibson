import numpy as np
import yaml
import omnigibson as og
from omnigibson.renderer_settings.renderer_settings import RendererSettings

def main():
    # Create the environment
    cfg = yaml.load(open("vision_suite/fillable.yaml", "r"), Loader=yaml.FullLoader)
    env = og.Environment(configs=cfg)
    env.reset()

    # # Set camera to appropriate viewing pose
    # vision suite object
    # og.sim.viewer_camera.set_position_orientation(
    #     position=np.array([-0.03396363, -0.62034724, 3.03471115]),
    #     orientation=np.array([0.4064837, 0.00115538, 0.00259694, 0.91365361]),
    # )
    # vision suite object small
    # og.sim.viewer_camera.set_position_orientation(
    #     position=np.array([-0.07074663, -0.6411907, 3.33005199]),
    #     orientation=np.array([0.50899713, -0.00122137, -0.00206546, 0.86076487]),
    # )    
    # food
    og.sim.viewer_camera.set_position_orientation(
        position=np.array([-0.38490732, -0.51413021,  2.01774197]),
        orientation=np.array([0.10426396, 0.09969304, 0.68385767, 0.71521257]),
    )
    og.sim.viewer_width = 3480 # 2160
    og.sim.viewer_height = 2160 # 2700

    # fridge = env.scene.object_registry("name", "fridge")
    # cab1 = env.scene.object_registry("name", "bottom_cabinet_1")
    # washer = env.scene.object_registry("name", "washer")
    # fridge.set_joint_positions(np.array([1.57, 0]), drive=False)
    # cab1.set_joint_positions(np.array([0.2, 1]), drive=False)
    # washer.set_joint_positions(np.array([1.5]), drive=False)

    RendererSettings().set_current_renderer("Interactive (Path Tracing)")
    
    container = env.scene.object_registry("name", "teacup")
    # container.states[Filled].set_value(get_system("water"), True)
    for _ in range(100000000):
        # print(og.sim.viewer_camera.get_position_orientation())
        env.step(None)

    # Shut down at the end
    og.shutdown()


if __name__ == "__main__":
    main()
