"""
Example script demo'ing robot control.

Options for random actions, as well as selection of robot action space
"""

import torch as th
import numpy as np
import yaml
import os
import sys

import omnigibson as og
import omnigibson.utils.transform_utils as T

from grasp_utils.kinematic import IKSolver, FKSolver
from motion_planner.motion_plan_utils import PlanningContext, ArmPlanner
from motion_planner.constrained_planner import ArmContrainedPlanner

from collections import OrderedDict
from omnigibson.utils.asset_utils import (
    get_all_object_categories,
    get_all_object_category_models
)

from omnigibson.utils.grasping_planning_utils import get_grasp_position_for_open, grasp_position_for_open_on_prismatic_joint

from scipy.spatial.transform import Rotation as R
from tasks.actions import ActionPlan

# # Don't use GPU dynamics and use flatcache for performance boost
# gm.USE_GPU_DYNAMICS = False
# gm.ENABLE_FLATCACHE = True
GRASP_DIST = 0.15

def choose_env():
    env_dir = "envs/"
    files = [f for f in os.listdir(env_dir) if os.path.isfile(os.path.join(env_dir, f))]

    if not files:
        print(f"No files found in the {env_dir} directory.")
        return None

    print("****************Task file list****************")
    for idx, filename in enumerate(files, start=1):
        print(f"{idx}. {filename[:-5]}")

    while True:
        try:
            choice = int(input("Enter the number of the task you want to simulate: "))
            if 1 <= choice <= len(files):
                selected = files[choice - 1]
                print(f"You selected: {selected}")
                return os.path.join(env_dir, selected)
            else:
                print("Invalid number. Try again.")
        except ValueError:
            print("Please enter a valid number.")
    
def read_yaml(file_name):
    with open(file_name, "r") as file:
        benchmark_cfg = yaml.safe_load(file)
    
    cfg = benchmark_cfg["config"]

    # define task
    task_cfg = benchmark_cfg["task"]
    task_name = task_cfg["task_name"]
    print(f"************ Starting {task_name} ************")

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
    
    for _, obj in obj_dict.items():
        z = np.random.uniform(-np.pi, np.pi)
        obj.set_position_orientation(orientation=T.euler2quat(th.tensor([0,0,z])))
        for _ in range(20):
            obj.keep_still()
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

        # z = np.random.uniform(-np.pi, np.pi)
        obj.set_position_orientation(position=obj_pos, orientation=T.euler2quat(th.tensor([0,0,1])))
        for _ in range(20):
            og.sim.step()

        # check sampled region relation
        for area in areas:
            in_area |= obj.states[relation].get_value(area)
    
    return obj_pos

def get_pose_from_path(path_list, frame_rate=10):
    pos_list = []
    for path_idx in range(1, len(path_list)):
        start_pos = np.array(path_list[path_idx - 1])
        end_pos = np.array(path_list[path_idx])
        delta = (end_pos - start_pos) / frame_rate

        for i in range(frame_rate + 1):
            pos_list.append((start_pos + (delta * i)).tolist())
    
    return pos_list

def constrained_planning(env, robot, start_joints, goal_joints, num_const=0, custom_fn=None, collision_joints=None,
                         obj=None, link_name=None, disabled_collision_pairs_dict={}, tolerance=0.1, **kwarg):
    path = None
    with PlanningContext(env, robot, collision_joints, obj, link_name, disabled_collision_pairs_dict) as context:

        acp = ArmContrainedPlanner(context, tolerance=tolerance, custom_fn=custom_fn, num_const=num_const, spaceType=SPACETYPE)
        path = acp.plan(start_joints, goal_joints, context, planning_time=120.0, planner_type=PLANNERTYPE)

        if path:
            path = get_pose_from_path(path)
            path.insert(0, start_joints)

    return path

def arm_planning(env, robot, start_joints, goal_joints, collision_joints=None, obj=None,
                    link_name=None, disabled_collision_pairs_dict={}, **kwarg):
    path = None
    with PlanningContext(env, robot, collision_joints, obj, link_name, disabled_collision_pairs_dict) as context:
        ap = ArmPlanner(context)
        path = ap.plan(start_joints, goal_joints, context)

        if path:
            path = get_pose_from_path(path)
            path.insert(0, start_joints)

    return path

def main():
    """
    Contraint Motion planning demo with task selection.
    """

    # Choose the environment
    yaml_file = choose_env()
    # yaml_file = "envs/liqiuid_pouring.yaml"
    # yaml_file = "envs/object_cutting.yaml"
    # yaml_file = "envs/drawer_opening.yaml"
    # yaml_file = "envs/cabinet_opening.yaml"
    # yaml_file = "envs/stirring.yaml"

    cfg, obj_cfg, action_cfg, cam_pos = read_yaml(yaml_file)
    env = og.Environment(configs=cfg)
    robot = env.robots[0]
    env.scene.update_initial_state()

    if cfg["scene"]["scene_model"] == "Pomaria_1_int":
        rm_obj = env.scene.object_registry("name", "burner_pmntxh_0")
        env.scene.remove_object(rm_obj)

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
    for _ in range(100): og.sim.step()
    
    # plan task
    ap = ActionPlan(env, robot, ik_solver, fk_solver)
    find_grasp = False
    for name, action in action_cfg.items():
        print("******* NAME: ", name)
        if action[1] is None:
            planner = arm_planning
            action[1] = None
        else:
            planner = constrained_planning

        if not obj_dict: 
            # find name in object list
            obj_list = [obj for obj in env.scene.objects if action[0] in obj.name]
            
            min_dist = 5
            target_obj = None
            for obj in obj_list:
                obj_loc = obj.get_position_orientation()
                dist = np.linalg.norm(obj_loc[0][:2] - robot_loc[0][:2])

                if dist < min_dist:
                    min_dist = dist
                    target_obj = obj
            
            if target_obj is None:
                raise Exception(f"Can't find {action[0]}!!!!!!")
                
        elif type(action[0]) == list:
            target_obj = None
        else:
            target_obj = obj_dict[action[0]]

        kwargs = {
            "planner" : planner,
            "target_obj" : target_obj,
            "target_rot" : action[0],
            "const_dict" : action[1],
        }
        
        if find_grasp:
            if ap.find_grasps(grasp_obj, name, kwargs):
                print(f"-------------- Successfully generated {name} plan --------------")
                find_grasp = False
                continue
            else:
                raise Exception(f"{name} Planning Failed!!!!!!")

        if name == "grasp":
            find_grasp = True

            if not obj_dict: 
                # find name in object list
                obj_list = [obj for obj in env.scene.objects if action[0] in obj.name]
                
                min_dist = 5
                grasp_obj = None
                for obj in obj_list:
                    obj_loc = obj.get_position_orientation()
                    dist = np.linalg.norm(obj_loc[0][:2] - robot_loc[0][:2])

                    if dist < min_dist:
                        min_dist = dist
                        grasp_obj = obj
            else:
                grasp_obj = obj_dict[action[0]]

            continue

        # Plan other actions
        action_fn = getattr(ap, name, Exception(f'Action does not exist. Check {yaml_file} file!!!!'))
        if action_fn(**kwargs):
            print(f"-------------- Successfully generated {name} plan --------------")
        else:
            raise Exception(f"{name} Planning Failed!!!!!!")

    # constrained planner successed
    print("********** plan grasp ***********")
    ap.add_grasp(arm_planning)

    input("Press any keys to simulate motion:")    
    ap.execute_actions()
    for _ in range(200): og.sim.step()

if __name__ == "__main__":
    space_match = {"PJ": ["KPIECE1", "BKPIECE1"], "AT": ["RRTConnect", "RRTstar", "RRT", "PRM"]}
    if len(sys.argv) < 3:
        print("***********Using Project sapce with KPIECE1 planner***********")
        SPACETYPE = "PJ"
        PLANNERTYPE = "KPIECE1"

    else:
        assert sys.argv[1] in space_match.keys(), f"Please choose space type from here: \n{[*space_match.keys()]}"
        assert sys.argv[2] in space_match[sys.argv[1]], f"Please choose planner type from here: \n{space_match[sys.argv[1]]}"
        SPACETYPE = sys.argv[1]
        PLANNERTYPE = sys.argv[2]
        print(f"Planner Setting: {SPACETYPE} with {PLANNERTYPE} planner")

    main()
