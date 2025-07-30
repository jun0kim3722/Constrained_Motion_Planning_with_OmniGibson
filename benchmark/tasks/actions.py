from copy import deepcopy
import torch as th
import numpy as np
from collections import OrderedDict

from motion_planner.motion_plan_utils import PlanningContext
from motion_planner.constrained_planner import ArmCcontrainedPlanner
from motion_planner.motion_plan_utils import PlanningContext, ArmPlanner
from omnigibson.utils.grasping_planning_utils import get_grasp_position_for_open

import omnigibson.utils.transform_utils as T
import omnigibson as og

# **************** Helper Functions ****************
# def get_constraint(const_dict, gripper_pos, gripper_rot):
#     # calc constraints
#     num_const = 0
#     rot_mask = th.full((3,), False)
#     if const_dict["rot"]:
#         rot_const = gripper_rot
#         if 'x' in const_dict["rot"]:
#             rot_mask[0] = True
#             num_const += 1

#         if 'y' in const_dict["rot"]:
#             rot_mask[1] = True
#             num_const += 1

#         if 'z' in const_dict["rot"]:
#             rot_mask[2] = True
#             num_const += 1
#     else: 
#         rot_const = None
    
#     trans_mask = th.full((3,), False)
#     if const_dict["trans"]:
#         trans_const = gripper_pos
#         if 'x' in const_dict["trans"]:
#             trans_mask[0] = True
#             num_const += 1

#         if 'y' in const_dict["trans"]:
#             trans_mask[1] = True
#             num_const += 1

#         if 'z' in const_dict["trans"]:
#             trans_mask[2] = True
#             num_const += 1
#     else: 
#         trans_const = None

#     return {"rot_const" : rot_const, "rot_mask" : rot_mask,
#             "trans_const" : trans_const, "trans_mask" : trans_mask,
#             "num_const" : num_const}

def execute_motion(env, robot, joint_path, is_gripper_open):
    for joints in joint_path:
        ctr = np.concatenate((joints, [1 if is_gripper_open else -1]), dtype="float32")
        # robot.keep_still()
        robot.apply_action(ctr)
        for _ in range(10):
            og.sim.step()

def execute_controller(env, joints, is_gripper_open):
    ctr = np.concatenate((joints, [1 if is_gripper_open else -1]), dtype="float32")
    action = OrderedDict([('UR5e', ctr)])
    env.step(action)

def get_pose_from_path(path_list, frame_rate=10):
    pos_list = []
    for path_idx in range(1, len(path_list)):
        start_pos = np.array(path_list[path_idx - 1])
        end_pos = np.array(path_list[path_idx])
        delta = (end_pos - start_pos) / frame_rate

        for i in range(frame_rate + 1):
            pos_list.append((start_pos + (delta * i)).tolist())
    
    return pos_list


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

        self.collision_joints = robot.get_joint_positions()[:6]
        self.last_joint = robot.get_joint_positions()[:6]
    
    def get_constraint(self, const_dict, const_pos, const_rot):
        # calc constraints
        num_const = 0
        rot_mask = th.full((3,), False)
        if const_dict["rot"]:
            rot_const = const_rot
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
            trans_const = const_pos
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

        def const_fn(joints):
            # if type(joints) == th.Tensor:
            #     joints = joints.detach().cpu().numpy()
            
            trans, quat = self.fk_solver.get_link_poses_quat(
                    (joints), [self.robot._eef_link_names]
                )[self.robot._eef_link_names]
            
            if trans_const is not None:
                trans_diff = (trans_const - trans)[trans_mask]
            else:
                trans_diff = th.empty(0)
            
            if rot_const is not None:
                quat_diff = T.quat_distance(rot_const, quat)
                axis_diff = T.quat2axisangle(quat_diff)
                rot_diff = axis_diff[rot_mask]
            else:
                rot_diff = th.empty(0)

            return th.cat((trans_diff, rot_diff))

        return {"custom_fn" : const_fn, "num_const" : num_const}
    
    def get_line_constraint(self, start_trans, goal_trans, rot_const, rot_mask=th.full((3,), True)):
        line = start_trans - goal_trans
        line_dot = th.dot(line, line)

        def const_fn(joints):
            if type(joints) == th.Tensor:
                joints = joints.detach().cpu().numpy()

            trans, quat = self.fk_solver.get_link_poses_quat(
                    (joints), [self.robot._eef_link_names]
                )[self.robot._eef_link_names]

            vec = trans - start_trans
            t = th.dot(vec, line) / line_dot
            Q = start_trans + t * line

            quat_diff = T.quat_distance(rot_const, quat)
            axis_diff = T.quat2axisangle(quat_diff)
            rot_diff = axis_diff[rot_mask]

            return th.cat((th.norm(trans - Q).unsqueeze(0), rot_diff))

        return {"custom_fn" : const_fn, "num_const" : int(1 + sum(rot_mask))}
    
    def constrained_planning(self, env, robot, start_joints, goal_joints, collision_joints=None, obj=None,
                         disabled_collision_pairs_dict={}, trans_const=None, rot_const=None,
                         rot_mask=None, trans_mask=None, num_const=0,  **kwarg):
        path = None
        with PlanningContext(env, robot, collision_joints, obj, disabled_collision_pairs_dict) as context:
            acp = ArmCcontrainedPlanner(context, trans_const=trans_const, rot_const=rot_const,
                                        rot_mask=rot_mask, trans_mask=trans_mask, num_const=num_const, tolerance=np.deg2rad(30.0))
            path = acp.plan(start_joints, goal_joints, context, planning_time=120.0)

            if path:
                path = get_pose_from_path(path)
                path.insert(0, start_joints)

        return path

    def arm_planning(self, env, robot, start_joints, goal_joints, collision_joints=None, obj=None,
                        disabled_collision_pairs_dict={}, **kwarg):
        path = None
        with PlanningContext(env, robot, collision_joints, obj, disabled_collision_pairs_dict) as context:
            ap = ArmPlanner(context)
            path = ap.plan(start_joints, goal_joints, context)

            if path:
                path = get_pose_from_path(path)
                path.insert(0, start_joints)

        return path

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
                self.last_joint = offset_joints
                self.collision_joints = grasp_joints
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
        
        plan = [self.grasp_joints] * 5
        self.motion_plan.insert(1, [plan, False])

        offset_path = [self.offset_joints] * 10
        self.motion_plan.insert(2, [offset_path, False])
    
    def find_grasps(self, grasp_obj, action_name, kwargs):            
        action_fn = getattr(self, action_name, Exception('find_grasps: action name does not exist!!'))
        if action_name == "open_or_close":
            return action_fn(**kwargs)

        grasp_list = self.ik_solver.get_grasp(grasp_obj)
        self.in_hand = grasp_obj
        center_loc = grasp_obj.aabb_center + th.tensor([0,0,grasp_obj.aabb_extent[2]/2.0])
        for offset_joints, grasp_joints in grasp_list:
            self.collision_joints = grasp_joints.clone()
            if action_fn(**kwargs, offset_joints=offset_joints):
                self.grasp_joints = grasp_joints.clone()
                self.offset_joints = offset_joints.clone()
                self.in_hand = grasp_obj

                robot_pos = self.robot.get_position_orientation()
                eef_pos = self.fk_solver.get_link_poses_quat(
                    grasp_joints, [self.robot._eef_link_names])[self.robot._eef_link_names]
                global_eef = T.pose_transform(*robot_pos, *eef_pos)

                self.obj2gripper = T.relative_pose_transform(
                    *global_eef, 
                    center_loc, grasp_obj.get_position_orientation()[1],
                )

                return True
        
        return False
        
    def hover(self, planner, target_obj, const_dict, offset_joints=None, **kwargs):
        if offset_joints is None:
            start_joints = self.last_joint
        else:
            start_joints = offset_joints.clone()

        gripper_trans, gripper_quat = self.fk_solver.get_link_poses_quat(
            start_joints, [self.robot._eef_link_names])[self.robot._eef_link_names]
        const_arg = self.get_constraint(const_dict, gripper_trans.clone(), gripper_quat.clone())
        goal_trans = target_obj.get_position_orientation()[0] + th.tensor([0, 0, 0.4])
        
        goal_joints = self.ik_solver.solve_newcoord(goal_trans, gripper_quat)
        if goal_joints is None: return False

        path = planner(self.env, self.robot, start_joints, goal_joints, collision_joints=self.collision_joints,
                    obj=self.in_hand, disabled_collision_pairs_dict={}, **const_arg)
        if path is None: return False

        self.motion_plan.append([path, False])
        self.last_joint = goal_joints
        og.log.info("Hover plan generated!!")
        return True
    
    def twist(self, planner, target_rot, const_dict, offset_joints=None, **kwargs):
        if offset_joints is None:
            start_joints = self.last_joint
        else:
            start_joints = offset_joints.clone()

        gripper_trans, gripper_quat = self.fk_solver.get_link_poses_quat(
            start_joints, [self.robot._eef_link_names])[self.robot._eef_link_names]
        const_arg = self.get_constraint(const_dict, gripper_trans.clone(), gripper_quat.clone())

        for rot_axis, rot_degree in target_rot:
            twist_euler = th.zeros(3)
            twist_euler[rot_axis] = np.deg2rad(rot_degree)
            goal_quat = T.quat_multiply(T.euler2quat(twist_euler), gripper_quat)
            goal_trans = th.tensor(gripper_trans)
            goal_joints = self.ik_solver.solve_newcoord(goal_trans, goal_quat)

            if goal_joints is None:
                continue
            
            path = planner(self.env, self.robot, start_joints, goal_joints, collision_joints=self.collision_joints,
                    obj=self.in_hand, disabled_collision_pairs_dict={}, **const_arg)
            
            if path is not None:
                break

        if path is None:
            return False
        
        self.motion_plan.append([path, False])
        self.last_joint = goal_joints
        og.log.info("Twist plan generated!!")
        return True
    
    def cut(self, planner, target_obj, const_dict, offset_joints=None, **kwargs):
        if offset_joints is None:
            start_joints = self.last_joint
        else:
            start_joints = offset_joints.clone()

        # find start pose
        center_loc = target_obj.aabb_center + th.tensor([0,0,target_obj.aabb_extent[2]/2.0])
        robot_trans = self.robot.get_position_orientation()[0]
        direction = center_loc[:2] - robot_trans[:2]
        extra_rot = T.euler2quat(th.tensor([0,0.6,np.arctan2(direction[1], direction[0])]))

        offset = th.tensor([-0.10, 0.08, 0.1])
        obj_trans = center_loc + T.quat_apply(extra_rot, offset)
        obj_quat = T.quat_multiply(extra_rot, th.tensor([-0.5, -0.5, 0.5, 0.5])) #[ 0.1423,  0.0901, -0.6374][-3.3379e-06,  1.0157e-07, -6.9765e-02]

        # offset = th.tensor([-0.10, 0.0, 0.1])
        # obj_trans = center_loc + T.quat_apply(extra_rot, offset)
        # obj_quat = T.quat_multiply(extra_rot, th.tensor([0, -0.707, 0.707, 0]))

        ready_pos = T.pose_transform(obj_trans, obj_quat, *self.obj2gripper)
        ready_joints = self.ik_solver.solve_newcoord(*ready_pos)
        print(ready_joints)

        if ready_joints is None:
            print("cutting filed")
            return False
        
        # with PlanningContext(self.env, self.robot, self.collision_joints, self.in_hand) as context:
        #     breakpoint()
        #     context.set_arm_and_detect_collision(ready_joints, False)
        #     for _ in range(100): og.sim.step()

        path2ready = self.arm_planning(self.env, self.robot, start_joints, ready_joints,
                       collision_joints=self.collision_joints, obj=self.in_hand)
        if path2ready is None:
            print("cutting path plan failed")
            return False

        # disable collision with Knife and target_obj
        path = self.robot.eef_links[self.robot.default_arm].prim_path + "/attached_obj_"
        obj_prim_path = [path + str(i) for i in range(4)]
        disabled_collision = {target_obj.prim_path + "/base_link" : obj_prim_path}

        # get constraint
        gripper_trans, gripper_quat = self.fk_solver.get_link_poses_quat(
            ready_joints, [self.robot._eef_link_names])[self.robot._eef_link_names]
        const_arg = self.get_constraint(const_dict, gripper_trans.clone(), gripper_quat.clone())
        knife_rot = T.euler2quat(th.tensor([0,0,np.arctan2(direction[1], direction[0])]))
        # get cut joints
        for cut_height in np.arange(0.14, 0.01, -0.01):
            cut_trans = obj_trans - th.tensor([0, 0, cut_height])
            cut_pos = T.pose_transform(cut_trans, obj_quat, *self.obj2gripper)
            cut_joints = self.ik_solver.solve_newcoord(*cut_pos)

            if cut_joints is None: continue
        
            print("Planning Cut")
            path2cut = planner(self.env, self.robot, ready_joints, cut_joints, collision_joints=self.collision_joints,
                        obj=self.in_hand, disabled_collision_pairs_dict=deepcopy(disabled_collision), **const_arg)
            
            if path2cut is None:
                continue

            cut_end_trans = cut_trans + T.quat_apply(knife_rot, th.tensor([-0.2,0,0]))
            cut_end_pos = T.pose_transform(cut_end_trans, obj_quat, *self.obj2gripper)
            cut_end_joints = self.ik_solver.solve_newcoord(*cut_end_pos, initial_joint_pos=cut_joints)

            breakpoint()
            if cut_end_joints is None: continue
            
            cut_gripper_trans, cut_gripper_quat = self.fk_solver.get_link_poses_quat(cut_joints, [self.robot._eef_link_names])[self.robot._eef_link_names]
            gl_trans = cut_end_pos[0] - robot_trans
            cut_end_arg = self.get_line_constraint(cut_gripper_trans, gl_trans, cut_gripper_quat)

            print("Planning Cut End")
            path2cutend = planner(self.env, self.robot, cut_joints, cut_end_joints, collision_joints=self.collision_joints,
                        obj=self.in_hand, disabled_collision_pairs_dict=deepcopy(disabled_collision), **cut_end_arg)

            if path2cutend is not None:
                with PlanningContext(self.env, self.robot, self.collision_joints, self.in_hand) as context:
                    for j in path2cutend:
                        # check wtf is going on here. Plan looks weird sometimes.
                        context.set_arm_and_detect_collision(j, False)
                        for _ in range(200): og.sim.step()
                        breakpoint()
                break


        if path2cut is None or path2cutend is None:
            return False
            # raise Exception("Cut Planning Failed!!!!")
        
        complete_motion = path2cut + path2cutend + path2cutend[::-1] + path2cut[::-1]
        self.motion_plan.append([complete_motion, False])
        self.last_joint = ready_joints
        og.log.info("Cut plan generated!!")
        return True
    
    def open_or_close(self, planner, target_obj, const_dict, offset_joints=None, **kwargs):
        if offset_joints is None:
            start_joints = self.last_joint
        else:
            start_joints = offset_joints.clone()

        for i in range(len(target_obj._joints)-1, -1, -1):
            # find start goal joints
            joint_name = 'j_link_' + str(i)
            (
                relevant_joint,
                offset_grasp_pos,
                grasp_pos,
                goal_pos,
                _,
                required_pos_change,
            ) = get_grasp_position_for_open(self.robot, target_obj, True, relevant_joint=target_obj._joints[joint_name], offset=0.155)

            offset_joints = self.ik_solver.solve_newcoord(*offset_grasp_pos)
            if offset_joints is None: continue
            grasp_joints = self.ik_solver.solve_newcoord(*grasp_pos, initial_joint_pos=offset_joints)
            if grasp_joints is None: continue
            goal_joints = self.ik_solver.solve_newcoord(*goal_pos, initial_joint_pos=grasp_joints)
            if goal_joints is None: continue

            grasp_trans, grasp_quat = self.fk_solver.get_link_poses_quat((grasp_joints), [self.robot._eef_link_names])[self.robot._eef_link_names]
            goal_trans, goal_quat = self.fk_solver.get_link_poses_quat((goal_joints), [self.robot._eef_link_names])[self.robot._eef_link_names]
            line_const = self.get_line_constraint(grasp_trans, goal_trans, grasp_quat)

            path = planner(self.env, self.robot, offset_joints, goal_joints, collision_joints=self.collision_joints,
                        obj=None, **line_const)

            if path is not None:
                self.motion_plan.append([path, False])
                self.motion_plan.append([[path[-1]] * 10, True])

                self.offset_joints = offset_joints.clone()
                self.grasp_joints = grasp_joints.clone()
                self.last_joint = path[-1]
                target_obj._joints[joint_name].friction = 0.1 # lower friciton for easy manipulation
                return True
        
        return False

    def execute_actions(self):
        og.log.info("Executing Actions!!")
        # self.robot.control_enabled = False # This does not make any move
        # self.robot.set_joint_velocities() # try this!!!!!!

        for path, is_gripper_open in self.motion_plan:
            execute_motion(self.env, self.robot, path, is_gripper_open)