"""
Example script demo'ing robot control.

Options for random actions, as well as selection of robot action space
"""

import torch as th
import omnigibson as og
import omnigibson.utils.transform_utils as T
from benchmark.grasp_utils.kinematic import IKSolver, FKSolver
from omnigibson.utils.grasping_planning_utils import get_grasp_poses_for_object_sticky
import yaml
import numpy as np

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
            return joint_pos
    
    return None

def main(random_selection=False, headless=False, short_exec=False, quickstart=False):
    """
    Robot control demo with selection
    Queries the user to select a robot, the controllers, a scene and a type of input (random actions or teleop)
    """
    # og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80)

    # Create the environment
    cfg = read_yaml("envs/kitchen_env_config.yaml")
    env = og.Environment(configs=cfg)
    robot = env.robots[0]
    env.scene.update_initial_state()

    robot_loc = robot.get_position_orientation()
    robot_pose_world = T.pose2mat(robot_loc)
    ik_solver = IKSolver(
        robot_description_path=robot.robot_arm_descriptor_yamls[robot.default_arm],
        robot_urdf_path=robot.urdf_path,
        reset_joint_pos=robot.get_joint_positions()[robot.arm_control_idx[robot.default_arm]],
        eef_name=robot.eef_link_names[robot.default_arm],
        world2robot_homo=T.pose_inv(robot_pose_world)
    )

    # Update the simulator's viewer camera's pose so it points towards the robot
    og.sim.viewer_camera.set_position_orientation(
    position=th.tensor([-0.18768, -1.86944, 2.22927]),
    orientation=th.tensor([0.37, 0.39, 0.61, 0.58]),
    )

    # Reset environment and robot
    env.reset()
    robot.reset()

    # plan grasp
    teacup = env.scene.object_registry("name", "teacup")
    knife = env.scene.object_registry("name", "knife")
    tablespoon = env.scene.object_registry("name", "tablespoon")
    teacup_joints = plan_grasp(robot, ik_solver, teacup)
    knife_joints = plan_grasp(robot, ik_solver, knife)
    tablespoon_joints = plan_grasp(robot, ik_solver, tablespoon)

    # Other helpful user info
    print("Running demo.")
    print("Press ESC to quit")

    # Loop control until user quits
    max_steps = -1 if not short_exec else 10000
    step = 0

    while step != max_steps:
        action = env.action_space.sample()
        action['UR5e'] = np.concatenate((teacup_joints, [-1]), dtype="float32")
        env.step(action)

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
