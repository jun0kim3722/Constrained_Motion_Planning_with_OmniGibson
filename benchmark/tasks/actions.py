import torch as th
import numpy as np
from collections import OrderedDict
from motion_planner.motion_plan_utils import PlanningContext

import omnigibson.utils.transform_utils as T
import omnigibson as og

# **************** Helper Functions ****************
def get_constraint(const_dict, gripper_pos, gripper_rot):
    # calc constraints
    num_const = 0
    rot_mask = th.full((3,), False)
    if const_dict["rot"]:
        rot_const = gripper_rot
        if 'x' in const_dict["rot"]:
            rot_mask[0] = True
            num_const += 1

        if 'y' in const_dict["rot"]:
            rot_mask[1] = True
            num_const += 1

        if 'z' in const_dict["rot"]:
            rot_mask[2] = True
            num_const += 1
    else: 
        rot_const = None
    
    trans_mask = th.full((3,), False)
    if const_dict["trans"]:
        trans_const = gripper_pos
        if 'x' in const_dict["trans"]:
            trans_mask[0] = True
            num_const += 1

        if 'y' in const_dict["trans"]:
            trans_mask[1] = True
            num_const += 1

        if 'z' in const_dict["trans"]:
            trans_mask[2] = True
            num_const += 1
    else: 
        trans_const = None

    return {"rot_const" : rot_const, "rot_mask" : rot_mask,
            "trans_const" : trans_const, "trans_mask" : trans_mask,
            "num_const" : num_const}

def execute_motion(env, robot, joint_path, is_gripper_open):
    for joints in joint_path:
        ctr = np.concatenate((joints, [1 if is_gripper_open else -1]), dtype="float32")
        robot.apply_action(ctr)
        for _ in range(5):
            og.sim.step()

def execute_controller(env, joints, is_gripper_open):
    ctr = np.concatenate((joints, [1 if is_gripper_open else -1]), dtype="float32")
    action = OrderedDict([('UR5e', ctr)])
    env.step(action)


# **************** Action Planner ****************
class ActionPlan():
    def __init__(self, env, robot, ik_solver, fk_solver, in_hand=None):
        self.env = env
        self.robot = robot
        self.ik_solver = ik_solver
        self.fk_solver = fk_solver

        self.motion_plan = []
        self.in_hand = in_hand
        self.gripper2obj = None

        self.grasp_joints = None
        self.offset_joints = None

        self.collision_pos = robot.get_joint_positions()[:6]
        self.last_pos = robot.get_joint_positions()[:6]

    def close_gripper(self, joints):
        plan = [joints] * 20
        self.motion_plan.append([plan, False])

    def grasp(self, planner, grasp_obj):
        is_plan_success = False
        start_joints = self.robot.get_joint_positions()[:6]
        if self.grasp_joints is not None:
            self.in_hand = grasp_obj
            grasp_list = self.ik_solver.get_grasp(grasp_obj)
            for offset_joints, grasp_joints in grasp_list:
                path = planner(self.env, self.robot, start_joints, offset_joints)
                if not path: continue

                is_plan_success = True
                self.offset_joints = offset_joints
                self.last_pos = offset_joints
                self.collision_pos = grasp_joints
                self.grasp_joints = grasp_joints
                og.log.info("Grasp plan generated!!")
                break
        
        else:
            path = planner(self.env, self.robot, start_joints, self.offset_joints)
            if path is not None:
                is_plan_success = True
        
        for _ in range(20):
            path.append(self.grasp_joints)
        self.motion_plan.insert(0, [path, True])
        
        plan = [self.grasp_joints] * 20
        self.motion_plan.insert(1, [plan, False])

        offset_path = [self.offset_joints] * 20
        self.motion_plan.insert(2, [offset_path, False])
        
        if not is_plan_success:
            raise Exception("Object Grasp Planning Failed!!!!")

    def add_grasp(self, planner):
        start_joints = self.robot.get_joint_positions()[:6]
        path = planner(self.env, self.robot, start_joints, self.offset_joints)
        if path is None:
            raise Exception("Object Grasp Planning Failed!!!!")

        for _ in range(20):
            path.append(self.grasp_joints)
        self.motion_plan.insert(0, [path, True])
        
        plan = [self.grasp_joints] * 10
        self.motion_plan.insert(1, [plan, False])

        offset_path = [self.offset_joints] * 20
        self.motion_plan.insert(2, [offset_path, False])
    
    def find_grasps(self, grasp_obj, action_name, kwargs):
        action_fn = getattr(self, action_name, Exception('find_grasps: action name does not exist!!'))

        grasp_list = self.ik_solver.get_grasp(grasp_obj)
        self.in_hand = grasp_obj
        
        for offset_joints, grasp_joints in grasp_list:
            self.collision_pos = grasp_joints.clone()
            if action_fn(**kwargs, offset_joints=offset_joints):
                self.grasp_joints = grasp_joints.clone()
                self.offset_joints = offset_joints.clone()
                self.in_hand = grasp_obj

                robot_pos = self.robot.get_position_orientation()
                eef_pos = self.fk_solver.get_link_poses_quat(
                    grasp_joints, [self.robot._eef_link_names])[self.robot._eef_link_names]
                global_eef = T.pose_transform(*robot_pos, *eef_pos)

                self.obj2gripper = T.relative_pose_transform(
                    *global_eef, *grasp_obj.get_position_orientation(), 
                )

                return True
        
        return False
        
    def hover(self, planner, target_obj, const_dict, offset_joints=None, **kwargs):
        if offset_joints is None:
            start_joints = self.last_pos
        else:
            start_joints = offset_joints.clone()

        gripper_trans, gripper_quat = self.fk_solver.get_link_poses_quat(start_joints, [self.robot._eef_link_names])[self.robot._eef_link_names]
        const_arg = get_constraint(const_dict, gripper_trans.clone(), gripper_quat.clone())
        goal_trans = target_obj.get_position_orientation()[0] + th.tensor([0, 0, 0.4])
        
        goal_joints = self.ik_solver.solve_newcoord(goal_trans, gripper_quat)
        if goal_joints is None: return False

        path = planner(self.env, self.robot, start_joints, goal_joints, collision_joints=self.collision_pos,
                    obj=self.in_hand, disabled_collision_pairs_dict={}, **const_arg)
        if path is None: return False

        self.motion_plan.append([path, False])
        self.last_pos = goal_joints
        og.log.info("Hover plan generated!!")
        return True
    
    def twist(self, planner, target_rot, const_dict, offset_joints=None, **kwargs):
        if offset_joints is None:
            start_joints = self.last_pos
        else:
            start_joints = offset_joints.clone()

        gripper_trans, gripper_quat = self.fk_solver.get_link_poses_quat(start_joints, [self.robot._eef_link_names])[self.robot._eef_link_names]
        const_arg = get_constraint(const_dict, gripper_trans.clone(), gripper_quat.clone())

        for rot_axis, rot_degree in target_rot:
            twist_euler = th.zeros(3)
            twist_euler[rot_axis] = np.deg2rad(rot_degree)
            goal_quat = T.quat_multiply(T.euler2quat(twist_euler), gripper_quat)
            goal_trans = th.tensor(gripper_trans)
            goal_joints = self.ik_solver.solve_newcoord(goal_trans, goal_quat)

            if goal_joints is None:
                continue
            
            path = planner(self.env, self.robot, start_joints, goal_joints, collision_joints=self.collision_pos,
                    obj=self.in_hand, disabled_collision_pairs_dict={}, **const_arg)
            
            if path is not None:
                break

        if path is None:
            return False
        
        self.motion_plan.append([path, False])
        self.last_pos = goal_joints
        og.log.info("Twist plan generated!!")
    
    def cut(self, planner, target_obj, cutting_angle, const_dict, offset_joints=None, **kwargs):
        if offset_joints is None:
            start_joints = self.last_pos
        else:
            start_joints = offset_joints.clone()

        # find start pose
        gripper_trans, gripper_quat = self.fk_solver.get_link_poses_quat(start_joints, [self.robot._eef_link_names])[self.robot._eef_link_names]
        const_arg = get_constraint(const_dict, gripper_trans.clone(), gripper_quat.clone())

        obj_trans = target_obj.get_position_orientation()[0] + th.tensor([0, 0, 0.1])
        obj_quat = T.quat_multiply(th.tensor([0.0, 0.707, 0.0, -0.707]), T.euler2quat(th.tensor([0,0,cutting_angle])))
        # cut_quat = T.quat_multiply(obj_quat, target_obj.get_position_orientation()[1])
        # consider only z angle for target obj rotation
        goal_pos = T.pose_transform(obj_trans, obj_quat, *self.obj2gripper)
        goal_joints = self.ik_solver.solve_newcoord(*goal_pos)
        print(goal_joints)
        breakpoint()

        if goal_joints is None:
            print("cutting filed")
            breakpoint()
        
        path = planner(self.env, self.robot, start_joints, goal_joints, collision_joints=self.collision_pos,
                    obj=self.in_hand, disabled_collision_pairs_dict={}, **const_arg)

        if path is None:
            raise Exception("Cut Planning Failed!!!!")
        
        self.motion_plan.append([path, False])
        self.last_pos = goal_joints
        og.log.info("Cut plan generated!!")

    def execute_actions(self):
        og.log.info("Executing Actions!!")

        for path, is_gripper_open in self.motion_plan:
            execute_motion(self.env, self.robot, path, is_gripper_open)