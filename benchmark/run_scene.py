"""
Example script demo'ing robot control.

Options for random actions, as well as selection of robot action space
"""

import torch as th
import numpy as np
import yaml

import omnigibson as og
import omnigibson.utils.transform_utils as T

from grasp_utils.kinematic import IKSolver, FKSolver
from motion_planner.motion_plan_utils import PlanningContext, ArmPlanner
from motion_planner.constrained_planner import ArmCcontrainedPlanner

from collections import OrderedDict
from omnigibson.utils.asset_utils import (
    get_all_object_categories,
    get_all_object_category_models
)

from scipy.spatial.transform import Rotation as R
from tasks.actions import ActionPlan

# # Don't use GPU dynamics and use flatcache for performance boost
# gm.USE_GPU_DYNAMICS = False
# gm.ENABLE_FLATCACHE = True
GRASP_DIST = 0.15

def read_yaml(file_name):
    with open(file_name, "r") as file:
        benchmark_cfg = yaml.safe_load(file)
    
    cfg = benchmark_cfg["config"]

    # define task
    task_cfg = benchmark_cfg["task"]
    task_name = task_cfg["task_name"]
    og.log.info(f"************ Starting {task_name} ************")

    return cfg, task_cfg["objects"], task_cfg["actions"], [benchmark_cfg["cam_pos"]["position"], benchmark_cfg["cam_pos"]["orientation"]]

def setup_objects(env, obj_cfg):
    height = obj_cfg["base_height"]
    obj_locs = []
    obj_dict = {}
    for key, options in obj_cfg["to_add"]. items():
        if isinstance(options, list):
            category = np.random.choice(options)
            model = np.random.choice(get_all_object_category_models(category))
        
        else:
            category = np.random.choice(list(options.keys()))
            model = np.random.choice(options[category])
        
        # add objects to scene
        rob_pos = env.robots[0].get_position_orientation()[0]
        random_radius_squared = np.random.uniform(0.2, 0.7)
        r = np.sqrt(random_radius_squared)
        theta = np.random.uniform(0, 2 * np.pi)
        x = r * np.cos(theta) + rob_pos[0]
        y = r * np.sin(theta) + rob_pos[1]
        obj_pos = th.tensor([x, y, height])

        obj = og.objects.DatasetObject(name=category, category=category, model=model, position=[0,0,0])
        env.scene.add_object(obj)
        obj_pos = check_region(env, obj, obj_cfg, height, obj_locs)

        obj_locs.append(obj_pos)
        obj_dict[key] = obj

        for _ in range(10):
            og.sim.step()
        
    # check_region(env, obj, obj_cfg, height)
    for _, obj in obj_dict.items():
        z = np.random.uniform(-np.pi, np.pi)
        obj.set_position_orientation(orientation=T.euler2quat(th.tensor([0,0,z])))
        # obj.set_position_orientation(orientation=[0,0,0,1])
        for _ in range(20):
            og.sim.step()

    return obj_dict

def check_region(env, obj, obj_cfg, height, obj_locs):
    # get object relation
    if obj_cfg["relation"] == "on_top":
        relation = og.object_states.on_top.OnTop
    areas = [obj_class for obj_class in env.scene.objects if obj_class.name in obj_cfg["areas"]]
    
    rob_pos = env.robots[0].get_position_orientation()[0]
    in_area = False
    while not in_area:
        # re-sample
        random_radius_squared = np.random.uniform(0.2, 0.7)
        r = np.sqrt(random_radius_squared)
        theta = np.random.uniform(0, 2 * np.pi)
        x = r * np.cos(theta) + rob_pos[0]
        y = r * np.sin(theta) + rob_pos[1]
        obj_pos = th.tensor([x, y, height])

        # check dist with other objs
        is_collision = False
        for set_obj_loc in obj_locs:
            dist = th.norm(obj_pos[:2] - set_obj_loc[:2])
            if dist < 0.3:
                is_collision = True
                break
                
        if is_collision: continue

        obj.set_position_orientation(position=obj_pos, orientation=[0,0,0,1])

        for _ in range(10):
            og.sim.step()

        # check sampled region relation
        for area in areas:
            in_area |= obj.states[relation].get_value(area)
    
    return obj_pos

def execute_controller(env, joints, is_griper_open):
    ctr = np.concatenate((joints, [1 if is_griper_open else -1]), dtype="float32")
    action = OrderedDict([('UR5e', ctr)])
    env.step(action)
    og.sim.step()

def close_gripper(env, robot):
    curr_joints = robot.get_joint_positions()[:6]
    for _ in range(20):
        execute_controller(env, curr_joints, False)

def execute_motion(env, joint_path, is_griper_open):
    for joints in joint_path:
        execute_controller(env, joints, is_griper_open)

    for _ in range(50):
        og.sim.step()

def set_robot_to(env, robot, joint_angles, is_griper_open):
    joint_angles = th.tensor(joint_angles)
    while (robot.get_joint_positions()[:6] - joint_angles > 0.01).any():
        execute_controller(env, joint_angles, is_griper_open)
        print(robot.get_joint_positions()[:6], joint_angles)

def get_pose_from_path(path_list, frame_rate=10):
    pos_list = []
    for path_idx in range(1, len(path_list)):
        start_pos = np.array(path_list[path_idx - 1])
        end_pos = np.array(path_list[path_idx])
        delta = (end_pos - start_pos) / frame_rate

        for i in range(frame_rate + 1):
            pos_list.append((start_pos + (delta * i)).tolist())
    
    return pos_list

def constrained_planning(env, robot, start_joints, goal_joints, collision_joints=None, obj=None,
                         disabled_collision_pairs_dict={}, trans_const=None, rot_const=None,
                         rot_mask=None, trans_mask=None, num_const=0, custom_fn=None, **kwarg):
    path = None
    with PlanningContext(env, robot, collision_joints, obj, disabled_collision_pairs_dict) as context:

        acp = ArmCcontrainedPlanner(context, tolerance=np.deg2rad(30.0), trans_const=trans_const, 
                                    rot_const=rot_const, custom_fn=custom_fn, rot_mask=rot_mask, 
                                    trans_mask=trans_mask, num_const=num_const)
        path = acp.plan(start_joints, goal_joints, context, planning_time=120.0)

        if path:
            path = get_pose_from_path(path)
            path.insert(0, start_joints)

    return path

def arm_planning(env, robot, start_joints, goal_joints, collision_joints=None, obj=None,
                    disabled_collision_pairs_dict={}, **kwarg):
    path = None
    with PlanningContext(env, robot, collision_joints, obj, disabled_collision_pairs_dict) as context:
        ap = ArmPlanner(context)
        path = ap.plan(start_joints, goal_joints, context)

        if path:
            path = get_pose_from_path(path)
            path.insert(0, start_joints)

    return path

def main(random_selection=False, headless=False, short_exec=False, quickstart=False):
    """
    Robot control demo with selection
    Queries the user to select a robot, the controllers, a scene and a type of input (random actions or teleop)
    """

    # Create the environment
    yaml_file = "envs/cutting_object.yaml"
    cfg, obj_cfg, action_cfg, cam_pos = read_yaml(yaml_file)
    env = og.Environment(configs=cfg)
    robot = env.robots[0]
    env.scene.update_initial_state()

    # Update the simulator's viewer camera's pose so it points towards the robot
    og.sim.viewer_camera.set_position_orientation(
        position=th.tensor(cam_pos[0]),
        orientation=th.tensor(cam_pos[1]),
    )

    obj_dict = setup_objects(env, obj_cfg)

    robot_loc = robot.get_position_orientation()
    robot_pose_world = T.pose2mat(robot_loc)
    ik_solver = IKSolver(
        robot_description_path=robot.robot_arm_descriptor_yamls[robot.default_arm],
        robot_urdf_path=robot.urdf_path,
        reset_joint_pos=robot.get_joint_positions()[robot.arm_control_idx[robot.default_arm]],
        eef_name=robot.eef_link_names[robot.default_arm],
        world2robot_homo=T.pose_inv(robot_pose_world)
    )

    fk_solver = FKSolver(
        robot_description_path=robot.robot_arm_descriptor_yamls[robot.default_arm],
        robot_urdf_path=robot.urdf_path,
    )

    # Reset environment and robot
    env.reset()
    robot.reset()
    for _ in range(100):
        og.sim.step()


    # grasp_obj = obj_dict['grasp_obj']
    # grasp_list = ik_solver.get_grasp(grasp_obj)
    # for offset_joints, grasp_joints in grasp_list:
    #     path2cutend = constrained_planning(env, robot, robot.get_joint_positions()[:6], offset_joints, custom_fn=check_region, num_const=6)

    #     # for joints in joint_path:
    #     joints = robot.arm_joint_names[robot.default_arm]
    #     breakpoint()

            # robot.set_joint_positions(joints, offset_joints)
    #     for _ in range(1000):
    #         execute_controller(env, grasp_joints, True)

    # plan task
    ap = ActionPlan(env, robot, ik_solver, fk_solver)
    find_grasp = False
    for name, action in action_cfg.items():
        print("******* NAME: ", name)
        if action[1]['rot'] == None and action[1]['trans'] == None:
            planner = arm_planning
        else:
            planner = constrained_planning

        kwargs = {
            "planner" : planner,
            "target_obj" : obj_dict[action[0]],
            "target_rot" : action[0],
            "const_dict" : action[1],
        }
        
        if find_grasp:
            if ap.find_grasps(grasp_obj, name, kwargs):
                og.log.info(f"-------------- Successfully generated {name} plan --------------")
                find_grasp = False
            else:
                raise Exception(f"{name} Planning Failed!!!!!!")

        if name == "grasp":
            find_grasp = True
            grasp_obj = obj_dict['grasp_obj']
            continue
            # ap.grasp(arm_planning, grasp_obj)

        # Plan other actions
        action_fn = getattr(ap, name, Exception(f'Action does not exist. Check {yaml_file} file!!!!'))
        if action_fn(**kwargs):
            og.log.info(f"-------------- Successfully generated {name} plan --------------")
        else:
            raise Exception(f"{name} Planning Failed!!!!!!")

        # if name == "hover":
        #     target_obj = obj_dict[action[0]]
        #     const_dict = action[1]
        #     ap.hover(constrained_planning, target_obj, const_dict)
            
        # if name == "twist":
        #     target_rot = action[0]
        #     const_dict = action[1]
        #     ap.twist(constrained_planning, target_rot, const_dict)

        # if name == "cut":
        #     target_obj = obj_dict[action[0]]
        #     const_dict = action[1]
        #     ap.cut(constrained_planning, target_obj, const_dict)


    # constrained planner successed
    print("********** plan grasp ***********")
    ap.add_grasp(arm_planning)
    
    breakpoint()
    ap.execute_actions()

    # check if task is completed
    breakpoint()


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
