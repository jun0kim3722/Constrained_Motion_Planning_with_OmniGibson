import omnigibson as og
cfg = {
    "env": {
        "action_frequency": 30,
        "physics_frequency": 60,
    },
    "scene": {
        "type": "Pomaria_1_int",
    },
    "objects": [],
    "robots": [
        {
            "type": "UR5e",
            "obs_modalities": 'all',
            "controller_config": {
                "arm_0": {
                    "name": "NullJointController",
                    "motor_type": "position",
                },
            },
        }
    ]
}

env = og.Environment(configs=cfg)
action = ...
obs, reward, terminated, truncated, info = env.step(action)