import omnigibson as og
from omnigibson.macros import gm
import torch as th
from omnigibson.action_primitives.starter_semantic_action_primitives import (
    StarterSemanticActionPrimitives,
    StarterSemanticActionPrimitiveSet,
)
import numpy as np

cfg = dict()

# Define scene
cfg["scene"] = {
    "type": "InteractiveTraversableScene",
    "scene_model" : "Pomaria_1_int",
    "floor_plane_visible": True,
}

# Define robots
cfg["robots"] = [
    {
        "type": "UR5e",
        "name": "UR5e",
        "obs_modalities": ["rgb", "depth"],
        "position": [-3.056, -2.14226, 0.63268],
        "action_normalize": False,
        "controller_config":
        {
            'arm_0': {'name': 'JointController', 'use_delta_commands': False, 'command_input_limits': None}, 
            'gripper_0': {'name': 'MultiFingerGripperController', 'mode': 'binary'}
        }
    },
]

# Create the environment
env = og.Environment(cfg)

# Allow camera teleoperation
og.sim.enable_viewer_camera_teleoperation()

# Update the simulator's viewer camera's pose so it points towards the robot
og.sim.viewer_camera.set_position_orientation(
    position=th.tensor([-0.18768, -1.86944, 2.22927]),
    orientation=th.tensor([0.37, 0.39, 0.61, 0.58]),
)


# Step!
for _ in range(10000):

    action = env.action_space.sample()
    action['UR5e'] = np.concatenate(([np.pi, 0, 0, 0, 0, 0], [-1]), dtype="float32")
    # breakpoint()
    env.step(action)

og.shutdown()