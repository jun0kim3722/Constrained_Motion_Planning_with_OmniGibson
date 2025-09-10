import omnigibson as og

import torch as th
import numpy as np
import yaml
import sys

from omnigibson.utils.grasping_planning_utils import get_grasp_poses_for_object_sticky, get_grasp_position_for_open
import omnigibson.utils.transform_utils as T

from grasp_utils.kinematic import IKSolver, FKSolver

from motion_planner import motion_plan_utils
from motion_planner.constrained_planner import ArmContrainedPlanner
from collections import OrderedDict

GRASP_DIST = 0.15

def execute_motion(robot, joint_path, is_gripper_open):
    for joints in joint_path:
        ctr = np.concatenate((joints, [1 if is_gripper_open else -1]), dtype="float32")
        robot.apply_action(ctr)
        for _ in range(50):
            og.sim.step()

def get_pose_from_path(path_list, frame_rate=10):
    pos_list = []
    for path_idx in range(1, len(path_list)):
        start_pos = np.array(path_list[path_idx - 1])
        end_pos = np.array(path_list[path_idx])
        delta = (end_pos - start_pos) / frame_rate

        for i in range(frame_rate + 1):
            pos_list.append((start_pos + (delta * i)).tolist())
    
    return pos_list

# Create the environment
# cfg_file = sys.argv[1]
# cfg_file = "test_cfg/stacking.yaml"
# cfg_file = "test_cfg/hammering.yaml"
cfg_file = "test_cfg/pouring.yaml"

with open(cfg_file, "r") as file:
    cfg = yaml.safe_load(file)
env = og.Environment(cfg)
robot = env.robots[0]
robot_loc = robot.get_position_orientation()
robot_pose_world = T.pose2mat(robot_loc)

# Update the simulator's viewer camera's pose so it points towards the robot
og.sim.viewer_camera.set_position_orientation(
    position=th.tensor([-0.18768, -1.86944, 2.22927]),
    orientation=th.tensor([0.37, 0.39, 0.61, 0.58]),
)

# # controller = StarterSemanticActionPrimitives(env, enable_head_tracking=False)
# robot = env.robots[0]
# for _, controller_options in robot._default_controller_config.items():
#     options = list(sorted(controller_options.keys()))
#     print(options)

fk_solver = FKSolver(
        robot_description_path=robot.robot_arm_descriptor_yamls[robot.default_arm],
        robot_urdf_path=robot.urdf_path,
    )

if cfg["scene"]["scene_model"] == "Pomaria_1_int":
        rm_obj = env.scene.object_registry("name", "burner_pmntxh_0")
        env.scene.remove_object(rm_obj)

space_type = sys.argv[2]
planner_name = sys.argv[3]
path = np.load(f"../test_plans/{space_type}_{planner_name}_1_{cfg['file_name']}.txt.npy")
ctr = np.concatenate((path[0], [1]), dtype="float32")
robot.apply_action(ctr)
for _ in range(200): og.sim.step()

# reset obj position
if "hammering" in cfg_file:
    hammer = env.scene.object_registry("name", "hammer")
    hammer.keep_still()
    hammer.disable_gravity()
    hammer.set_position_orientation(position=[-2.14, -1.64,  1.05], orientation=[0, -0.707, -0.707, 0])
    for _ in range(100): og.sim.step()

elif "pouring" in cfg_file:
    cup = env.scene.object_registry("name", "cup")
    cup.keep_still()
    cup.disable_gravity()
    cup.set_position_orientation(position=[-2.23, -2.005,  1.1], orientation=[0, 0, 0, 1])
    for _ in range(100): og.sim.step()

cube = env.scene.object_registry("name", "cube")
cube.keep_still()
cube.set_position_orientation(position=[-2.48, -1.84227, 1.0])
for _ in range(100): og.sim.step()


# pos = fk_solver.get_link_poses_quat(path[0], [robot._eef_link_names])[robot._eef_link_names][0] + robot_loc[0]
is_grasp_list = cfg["is_grasp_list"]
for i in range(1, cfg["task_num"]+1):
    path = np.load(f"../test_plans/{space_type}_{planner_name}_{i}_{cfg['file_name']}.txt.npy")
    # breakpoint()
    execute_motion(robot, path, is_grasp_list[i-1])


og.shutdown()