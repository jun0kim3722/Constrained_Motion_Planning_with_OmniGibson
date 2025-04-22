"""
Example script demo'ing robot control.

Options for random actions, as well as selection of robot action space
"""

import torch as th

import omnigibson as og
import omnigibson.utils.transform_utils as T
from grasp_utils.ik_solver import IKSolver
from omnigibson.utils.grasping_planning_utils import get_grasp_poses_for_object_sticky
import yaml

# # Don't use GPU dynamics and use flatcache for performance boost
# gm.USE_GPU_DYNAMICS = False
# gm.ENABLE_FLATCACHE = True

def read_yaml(file_name):
    with open(file_name, "r") as file:
        cfg = yaml.safe_load(file)
    
    return cfg

def plan_grasp(robot, ik_solver, obj):
    grasp_poses = get_grasp_poses_for_object_sticky(obj)
    
    for grasp_pos in grasp_poses:
        target_pose_homo = T.pose2mat([grasp_pos[0][0], grasp_pos[0][1]])
        joint_pos = ik_solver.solve(target_pose_homo = target_pose_homo)
        
        if joint_pos is not None:
            print("Pos", grasp_pos[0][0])
            return joint_pos
    
    return None

def main(random_selection=False, headless=False, short_exec=False, quickstart=False):
    """
    Robot control demo with selection
    Queries the user to select a robot, the controllers, a scene and a type of input (random actions or teleop)
    """
    # og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80)

    # Add the robot we want to load
    robot_name = "UR5e"
    robot0_cfg = dict()
    robot0_cfg["type"] = robot_name
    robot0_cfg["obs_modalities"] = ["rgb"]
    robot0_cfg["action_type"] = "continuous"
    robot0_cfg["action_normalize"] = True

    # Create the environment
    cfg = read_yaml("envs/kitchen_env_config.yaml")
    # robot.get_position_orientation
    robot_pos = th.tensor(cfg["robots"][0]["position"])
    robot_rot = th.tensor(cfg["robots"][0]["orientation"])
    robot_pose_world = T.pose2mat([robot_pos, robot_rot])
    env = og.Environment(configs=cfg)

    # # Choose robot controller to use
    robot = env.robots[0]
    env.scene.update_initial_state()

    # Update the simulator's viewer camera's pose so it points towards the robot
    og.sim.viewer_camera.set_position_orientation(
    position=th.tensor([-0.18768, -1.86944, 2.22927]),
    orientation=th.tensor([0.37, 0.39, 0.61, 0.58]),
    )

    # Reset environment and robot
    env.reset()
    robot.reset()

    breakpoint()

    # Other helpful user info
    print("Running demo.")
    print("Press ESC to quit")

    # Loop control until user quits
    max_steps = -1 if not short_exec else 100
    step = 0

    while step != max_steps:
        
        env.step(action=action)
        print(action)
        step += 1

    # Always shut down the environment cleanly at the end
    og.clear()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Teleoperate a robot in a BEHAVIOR scene.")

    parser.add_argument(
        "--quickstart",
        action="store_true",
        help="Whether the example should be loaded with default settings for a quick start.",
    )
    args = parser.parse_args()
    main(quickstart=args.quickstart)
