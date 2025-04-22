import omnigibson as og
from omnigibson.macros import gm
import torch as th
from omnigibson.action_primitives.starter_semantic_action_primitives import (
    StarterSemanticActionPrimitives,
    StarterSemanticActionPrimitiveSet,
)
import numpy as np

from omnigibson.utils.grasping_planning_utils import get_grasp_poses_for_object_sticky_from_arbitrary_direction, get_grasp_poses_for_object_sticky, get_grasp_position_for_open
# from omnigibson.utils.control_utils import IKSolver
from grasp_utils.ik_solver import IKSolver
from scipy.spatial.transform import Rotation as R
import omnigibson.utils.transform_utils as T

def plan_grasp(robot, ik_solver, obj):
    grasp_poses = get_grasp_poses_for_object_sticky(obj)
    
    for grasp_pos in grasp_poses:
        target_pose_homo = T.pose2mat([grasp_pos[0][0], grasp_pos[0][1]])
        joint_pos = ik_solver.solve(target_pose_homo = target_pose_homo)
        
        if joint_pos is not None:
            print("Pos", grasp_pos[0][0])
            return joint_pos
    
    return None

def execute_ik(pos, quat=None, max_iter=100):
    og.log.info("Querying joint configuration to current marker position")
    # Grab the joint positions in order to reach the desired pose target
    joint_pos = ik_solver.solve(target_pos=pos,target_quat=None,tolerance_pos=0.002,tolerance_quat=0.01,weight_pos=20.0,weight_quat=0.05,max_iterations=1000,initial_joint_pos=robot.get_joint_positions()[control_idx])

    return joint_pos

def execute_controller(ctrl_gen, env):
    for action in ctrl_gen:
        env.step(action)

cfg = dict()

# Define scene
cfg["scene"] = {
    "type": "InteractiveTraversableScene",
    "scene_model" : "Pomaria_1_int",
    "floor_plane_visible": True,
}

# Define objects
cfg["objects"] = [
    {
        "type": "DatasetObject",
        "name": "teacup",
        "category": "teacup",
        "model": "vckahe",
        "position": [-3.15968, -2.52066, 0.96],
        # "position": [-2.74713, -2.52066, 0.95],
    },
    {
        "type": "DatasetObject",
        "name": "knife",
        "category": "carving_knife",
        "model": "alekva",
        "position": [-2.34713, -2.06784, 0.95],
    },
    {
        "type": "DatasetObject",
        "name": "tablespoon",
        "category": "tablespoon",
        "model": "huudhe",
        "position": [-2.37511, -2.49872, 0.95],
    },
    # {
    #     "type": "DatasetObject",
    #     "name": "sugar jar",
    #     "category": "jar_of_sugar",
    #     "model": "pnbbfb",
    #     "position": [0, -0.5, 1.0],
    # },
]

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
robot_pos = th.tensor(cfg["robots"][0]["position"])
robot_rot = th.tensor([0,0,0,1])
robot_pose_world = T.pose2mat([robot_pos, robot_rot])
env = og.Environment(cfg)

# Allow camera teleoperation
og.sim.enable_viewer_camera_teleoperation()

# Update the simulator's viewer camera's pose so it points towards the robot
og.sim.viewer_camera.set_position_orientation(
    position=th.tensor([-0.18768, -1.86944, 2.22927]),
    orientation=th.tensor([0.37, 0.39, 0.61, 0.58]),
)

# controller = StarterSemanticActionPrimitives(env, enable_head_tracking=False)
robot = env.robots[0]
for _, controller_options in robot._default_controller_config.items():
    options = list(sorted(controller_options.keys()))
    print(options)

# Grasp of teacup
# scene = env.scene
# grasp_obj = scene.object_registry("name", "teacup")
# print("Executing controller")
# execute_controller(controller.apply_ref(StarterSemanticActionPrimitiveSet.GRASP, grasp_obj), env)
# print("Finished executing grasp")

# # Place teacup on another table
# print("Executing controller")
# table = scene.object_registry("name", "table")
# execute_controller(controller.apply_ref(StarterSemanticActionPrimitiveSet.PLACE_ON_TOP, table), env)
# print("Finished executing place")

# object position
# obj_pos = env.scene.object_registry("name", "fuck").get_position_orientation()


# set ik solver
# ik_solver = IKSolver(
#                 robot_description_path=robot.robot_arm_descriptor_yamls[robot.default_arm],
#                 robot_urdf_path=robot.urdf_path,
#                 reset_joint_pos=robot.get_joint_positions()[robot.arm_control_idx[robot.default_arm]],
#                 eef_name=robot.eef_link_names[robot.default_arm],
#             )


# rot_mat = R.from_quat([0,0,0,1]).as_matrix()  # 3x3 rotation
# robot_pose_world = np.eye(4)
# robot_pose_world[:3, :3] = rot_mat
# robot_pose_world[:3, 3] = robot_pos


control_idx = th.cat([robot.arm_control_idx[robot.default_arm]])
ik_solver = IKSolver(
        robot_description_path=robot.robot_arm_descriptor_yamls[robot.default_arm],
        robot_urdf_path=robot.urdf_path,
        reset_joint_pos=robot.get_joint_positions()[control_idx],
        eef_name=robot.eef_link_names[robot.default_arm],
        world2robot_homo=T.pose_inv(robot_pose_world)
    )

# grasp position
teacup = env.scene.object_registry("name", "teacup")
knife = env.scene.object_registry("name", "knife")
tablespoon = env.scene.object_registry("name", "tablespoon")

teacup_pos = get_grasp_poses_for_object_sticky(teacup)[0][0][0] - th.tensor([-3.056, -2.14226, 0.63268])
knife_pos = get_grasp_poses_for_object_sticky(knife)[0][0][0] - th.tensor([-3.056, -2.14226, 0.63268])
tablespoon_pos = get_grasp_poses_for_object_sticky(tablespoon)[0][0][0] - th.tensor([-3.056, -2.14226, 0.63268])

teacup_joints = plan_grasp(robot, ik_solver, teacup)
knife_joints = plan_grasp(robot, ik_solver, knife)
tablespoon_joints = plan_grasp(robot, ik_solver, tablespoon)

print("teacup_joints", teacup_joints, teacup_pos)
print("knife_joints", knife_joints, knife_pos)
print("tablespoon_joints", tablespoon_joints, tablespoon_pos)

breakpoint()
# grasp_pos = get_grasp_poses_for_object_sticky_from_arbitrary_direction(obj)

# Step!
for _ in range(10000):
    # og.sim.step()
    if _ == 200:
        breakpoint()

    #     print("teacup")
    #     obj = env.scene.object_registry("name", "teacup")
    # elif _ < 400:
    #     print("knife")
    #     obj = env.scene.object_registry("name", "knife")
    # else:
    #     print("tablespoon")
    #     obj = env.scene.object_registry("name", "tablespoon")

    # joint_pos = plan_grasp(robot, ik_solver, obj)
    # print(joint_pos)

    # if joint_pos is not None:
    #     robot.set_joint_positions(th.tensor(joint_pos), indices=control_idx, drive=True)
    #     og.sim.step()

    action = env.action_space.sample()
    # angle = np.array([0.7, -2, 2.5, -0.3, 0.7, 0]) * np.pi
    action['UR5e'] = np.concatenate(([np.pi, 0, 0, 0, 0, 0], [-1]), dtype="float32")
    env.step(action)

og.shutdown()