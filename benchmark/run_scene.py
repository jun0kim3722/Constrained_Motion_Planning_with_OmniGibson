"""
Example script demo'ing robot control.

Options for random actions, as well as selection of robot action space
"""

import torch as th
import numpy as np
import yaml

import omnigibson as og
import omnigibson.utils.transform_utils as T

from grasp_utils.kinematic import IKSolver
from motion_planner import motion_plan_utils
from motion_planner.cont_planner import ArmCcontrainedPlanner

from collections import OrderedDict

# # Don't use GPU dynamics and use flatcache for performance boost
# gm.USE_GPU_DYNAMICS = False
# gm.ENABLE_FLATCACHE = True

GRASP_DIST = 0.15

def read_yaml(file_name):
    with open(file_name, "r") as file:
        cfg = yaml.safe_load(file)
    
    return cfg

def execute_controller(env, joints, is_griper_open):
    ctr = np.concatenate((joints, [1 if is_griper_open else -1]), dtype="float32")
    action = OrderedDict([('UR5e', ctr)])
    env.step(action)
    env.step(action)
    env.step(action)

def execute_motion(env, joint_path, is_griper_open):
    for joints in joint_path:
        execute_controller(env, joints, is_griper_open)

def get_pose_from_path(path_list, frame_rate=10):
    pos_list = []
    for path_idx in range(1, len(path_list)):
        start_pos = np.array(path_list[path_idx - 1])
        end_pos = np.array(path_list[path_idx])
        delta = (end_pos - start_pos) / frame_rate

        for i in range(frame_rate + 1):
            pos_list.append((start_pos + (delta * i)).tolist())
    
    return pos_list

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
    offset_joints, grasp_joints = ik_solver.get_grasp(knife)

    # motion planning
    path = None
    with motion_plan_utils.PlanningContext(env, robot, teacup) as context:
        # set planner
        griper_pos, griper_rot = context.fk_solver.get_link_poses_euler(start_joints, [robot._eef_link_names])[robot._eef_link_names]
        rot_const = griper_rot
        rot_const[-1] = None
        acp = ArmCcontrainedPlanner(context, trans_const=None, rot_const=rot_const, num_const=2, tolerance=np.deg2rad(15.0))

        adj_start_joints = start_joints
        adj_start_joints[1] -= 0.01
        path = acp.plan(robot, start_joints.tolist(), goal_joints.tolist(), context, planning_time=120.0)

        if path:
            path = get_pose_from_path(path)
            path.insert(0, start_joints)


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
