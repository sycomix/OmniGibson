import os
import argparse
import json
import omnigibson as og
import numpy as np
import omnigibson.utils.transform_utils as T

from omnigibson.macros import gm
from omnigibson.systems import get_system
from omnigibson.object_states import Covered
from omnigibson.utils.profiling_utils import ProfilingEnv
from omnigibson.utils.constants import PrimType

parser = argparse.ArgumentParser()

parser.add_argument("-r", "--robot", type=int, default=0)
parser.add_argument("-s", "--scene", default="")
parser.add_argument("-f", "--flatcache", action='store_true')
parser.add_argument("-c", "--softbody", action='store_true')
parser.add_argument("-w", "--fluids", action='store_true')
parser.add_argument("-p", "--macro_particle_system", action='store_true')

PROFILING_FIELDS = ["Total frame time", "Physics step time", "Render step time", "Non-physics step time"]


SCENE_OFFSET = {
    "": [0, 0],
    "Rs_int": [0, 0],
    "Pomaria_0_garden": [0.3, 0],
    "grocery_store_cafe": [-3.5, 3.5],
    "house_single_floor": [0, 0],
}


def main():
    args = parser.parse_args()
    # Modify flatcache, pathtracing, GPU, and object state settings
    assert not (args.flatcache and args.softbody), "Flatcache cannot be used with softbody at the same time"
    gm.ENABLE_FLATCACHE = args.flatcache
    gm.ENABLE_HQ_RENDERING = args.fluids
    gm.ENABLE_OBJECT_STATES = True
    gm.ENABLE_TRANSITION_RULES = True
    gm.USE_GPU_DYNAMICS = True

    cfg = {}
    if args.robot > 0:
        cfg["robots"] = []
        for i in range(args.robot):
            cfg["robots"].append({
                "type": "Fetch",
                "obs_modalities": ["scan", "rgb", "depth"],
                "action_type": "continuous",
                "action_normalize": True,
                "position": [-1.3 + 0.75 * i + SCENE_OFFSET[args.scene][0], 0.5 + SCENE_OFFSET[args.scene][1], 0],
                "orientation": [0., 0., 0.7071, -0.7071]
            })

    if args.scene:
        assert args.scene in SCENE_OFFSET, f"Scene {args.scene} not found in SCENE_OFFSET"
        cfg["scene"] = {
            "type": "InteractiveTraversableScene",
            "scene_model": args.scene,
        }
    else:
        cfg["scene"] = {
            "type": "Scene",
        }

    cfg["objects"] = [{
        "type": "DatasetObject",
        "name": "table",
        "category": "breakfast_table",
        "model": "rjgmmy",
        "scale": 0.75,
        "position": [0.5 + SCENE_OFFSET[args.scene][0], -1 + SCENE_OFFSET[args.scene][1], 0.4],
        "orientation": [0., 0., 0.7071, -0.7071]
    }]
    
    if args.softbody:
        cfg["objects"].append({
            "type": "DatasetObject",
            "name": "shirt",
            "category": "t_shirt",
            "model": "kvidcx",
            "prim_type": PrimType.CLOTH,
            "scale": 0.01,
            "position": [-0.4, -1, 1.5],
            "orientation": [0.7071, 0., 0.7071, 0.],
        })
    
    cfg["objects"].extend([{
        "type": "DatasetObject",
        "name": "apple",
        "category": "apple",
        "model": "agveuv",
        "scale": 1.5,
        "position": [0.5 + SCENE_OFFSET[args.scene][0], -1 + SCENE_OFFSET[args.scene][1], 0.6],
        "abilities": {"diceable": {}} if args.macro_particle_system else {}
    },
    {
        "type": "DatasetObject",
        "name": "knife",
        "category": "table_knife",
        "model": "lrdmpf",
        "scale": 2.5,
        "position": [0.5 + SCENE_OFFSET[args.scene][0], -1 + SCENE_OFFSET[args.scene][1], 0.8],
        "orientation": T.euler2quat([-np.pi / 2, 0, 0])
    }])

    env = ProfilingEnv(configs=cfg, action_timestep=1/60., physics_timestep=1/240.)
    env.reset()

    apple = env.scene.object_registry("name", "apple")
    table = env.scene.object_registry("name", "table")
    knife = env.scene.object_registry("name", "knife")
    knife.keep_still()
    knife.set_position_orientation(
        position=apple.get_position() + np.array([-0.15, 0.0, 0.2]),
        orientation=T.euler2quat([-np.pi / 2, 0, 0]),
    )
    if args.fluids:
        table.states[Covered].set_value(get_system("stain"), True)  # TODO: water is buggy for now, temporarily use stain instead

    output, results = [], []

    # Update the simulator's viewer camera's pose so it points towards the robot
    og.sim.viewer_camera.set_position([SCENE_OFFSET[args.scene][0], -3 + SCENE_OFFSET[args.scene][1], 1])
        
    for i in range(500):
        from IPython import embed; embed()

        if args.robot:
            result = env.step(np.array([np.random.uniform(-0.3, 0.3, env.robots[i].action_dim) for i in range(args.robot)]).flatten())[4][:4]
        else:
            result = env.step(None)[4][:4]
        results.append(result)

    results = np.array(results)
    for i, title in enumerate(PROFILING_FIELDS):
        field = f"{args.scene}" if args.scene else "Empty scene"
        if args.robot:
            field += f", with {args.robot} Fetch"
        if args.softbody:
            field += ", cloth" 
        if args.fluids:
            field += ", fluids"
        if args.macro_particle_system:
            field += ", macro particles"
        if args.flatcache:
            field += ", flatcache on"
        output.append({
            "name": field,
            "unit": "fps",
            "value": np.mean(results[-300:, i]) * 1000,
            "extra": [title, title]
        })

    ret = []
    if os.path.exists("output.json"):
        with open("output.json", "r") as f:
            ret = json.load(f)
    ret.extend(output)
    with open("output.json", "w") as f:
        json.dump(ret, f)
    og.shutdown()

if __name__ == "__main__":
    main()
