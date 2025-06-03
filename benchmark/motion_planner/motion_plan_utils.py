"""
WARNING!
The StarterSemanticActionPrimitive is a work-in-progress and is only provided as an example.
It currently only works with Fetch and Tiago with their JointControllers set to delta mode.
See provided tiago_primitives.yaml config file for an example. See examples/action_primitives for
runnable examples.
"""

import torch as th
import omnigibson as og
import omnigibson.lazy as lazy
import omnigibson.utils.transform_utils as T
from omnigibson.utils.usd_utils import GripperRigidContactAPI
from omnigibson.utils.sim_utils import prim_paths_to_rigid_prims

from omnigibson.robots import *
from grasp_utils.kinematic import FKSolver


class RobotCopy:
    """A data structure for storing information about a robot copy, used for collision checking in planning."""

    def __init__(self):
        self.prims = {}
        self.meshes = {}
        self.relative_poses = {}
        self.links_relative_poses = {}
        self.reset_pose = {
            "original": (th.tensor([0, 0, -0.5], dtype=th.float32), th.tensor([0, 0, 0, 1], dtype=th.float32)),
            "simplified": (th.tensor([0, 0, -2.0], dtype=th.float32), th.tensor([0, 0, 0, 1], dtype=th.float32)),
        }
    
    def __init__(self, og_robot):
        self.prims = {}
        self.meshes = {}
        self.relative_poses = {}
        self.links_relative_poses = {}
        # self.reset_pose = {
        #     "original": og_robot.get_position_orientation(),
        #     "simplified": og_robot.get_position_orientation(),
        # }
        self.reset_pose = {
            # "original": (th.tensor([-2.31736, -1.16763, 0.6], dtype=th.float32), th.tensor([0, 0, 0, 1], dtype=th.float32)),
            "original": (th.tensor([0, 0, -2.0], dtype=th.float32), th.tensor([0, 0, 0, 1], dtype=th.float32)),
            "simplified": (th.tensor([0, 0, -2.0], dtype=th.float32), th.tensor([0, 0, 0, 1], dtype=th.float32)),
        }
        robots_to_copy = {"original": {"robot": og_robot, "copy_path": og_robot.prim_path + "_copy"}}

        for robot_type, rc in robots_to_copy.items():
            copy_robot = None
            copy_robot_meshes = {}
            copy_robot_meshes_relative_poses = {}
            copy_robot_links_relative_poses = {}

            # Create prim under which robot meshes are nested and set position
            lazy.omni.usd.commands.CreatePrimCommand("Xform", rc["copy_path"]).do()
            copy_robot = lazy.omni.isaac.core.utils.prims.get_prim_at_path(rc["copy_path"])
            reset_pose = self.reset_pose[robot_type]
            translation = lazy.pxr.Gf.Vec3d(*reset_pose[0].tolist())
            copy_robot.GetAttribute("xformOp:translate").Set(translation)
            orientation = reset_pose[1][[3, 0, 1, 2]]
            copy_robot.GetAttribute("xformOp:orient").Set(lazy.pxr.Gf.Quatd(*orientation.tolist()))

            robot_to_copy = None
            if robot_type == "simplified":
                robot_to_copy = rc["robot"]
                self.env.scene.add_object(robot_to_copy)
            else:
                robot_to_copy = rc["robot"]

            # Copy robot meshes
            for link in robot_to_copy.links.values():
                link_name = link.prim_path.split("/")[-1]
                for mesh_name, mesh in link.collision_meshes.items():
                    split_path = mesh.prim_path.split("/")
                    # Do not copy grasping frame (this is necessary for Tiago, but should be cleaned up in the future)
                    # if "grasping_frame" in link_name:
                    #     continue

                    copy_mesh_path = rc["copy_path"] + "/" + link_name
                    copy_mesh_path += f"_{split_path[-1]}" if split_path[-1] != "collisions" else ""
                    lazy.omni.usd.commands.CopyPrimCommand(mesh.prim_path, path_to=copy_mesh_path).do()
                    copy_mesh = lazy.omni.isaac.core.utils.prims.get_prim_at_path(copy_mesh_path)

                    relative_pose = T.relative_pose_transform(
                        *mesh.get_position_orientation(), *link.get_position_orientation()
                    )

                    if link_name not in copy_robot_meshes.keys():
                        copy_robot_meshes[link_name] = {mesh_name: copy_mesh}
                        copy_robot_meshes_relative_poses[link_name] = {mesh_name: relative_pose}
                    else:
                        copy_robot_meshes[link_name][mesh_name] = copy_mesh
                        copy_robot_meshes_relative_poses[link_name][mesh_name] = relative_pose

                # breakpoint()
                copy_robot_links_relative_poses[link_name] = T.relative_pose_transform(
                    *link.get_position_orientation(), *og_robot.get_position_orientation()
                )
                # copy_robot_links_relative_poses[link_name] = og_robot.get_position_orientation()

            if robot_type == "simplified":
                self.env.scene.remove_object(robot_to_copy)

            self.prims[robot_type] = copy_robot
            self.meshes[robot_type] = copy_robot_meshes
            self.relative_poses[robot_type] = copy_robot_meshes_relative_poses
            self.links_relative_poses[robot_type] = copy_robot_links_relative_poses

        og.sim.step()


class PlanningContext(object):
    """
    A context manager that sets up a robot copy for collision checking in planning.
    """

    def __init__(self, env, robot, robot_copy, robot_copy_type="original"):
        self.env = env
        self.robot = robot
        self.robot_copy = robot_copy
        self.robot_copy_type = robot_copy_type if robot_copy_type in robot_copy.prims.keys() else "original"
        self.disabled_collision_pairs_dict = {}
        self.offset = (self.robot.get_position_orientation()[0] - self.robot_copy.reset_pose['original'][0],
                       self.robot.get_position_orientation()[1] - self.robot_copy.reset_pose['original'][1])

        self._assemble_robot_copy()
        self._construct_disabled_collision_pairs()

    def __enter__(self):
        self._assemble_robot_copy()
        self._construct_disabled_collision_pairs()
        return self

    def __exit__(self, *args):
        # breakpoint()
        # self._set_prim_pose(
        #     self.robot_copy.prims[self.robot_copy_type], self.robot_copy.reset_pose[self.robot_copy_type]
        # )

        for link_name, meshes in self.robot_copy.meshes[self.robot_copy_type].items():
            for mesh_name, copy_mesh in meshes.items():
                self._set_prim_pose(copy_mesh, self.robot_copy.reset_pose[self.robot_copy_type])


    def _assemble_robot_copy(self):
        self.fk_solver = FKSolver(
            robot_description_path=self.robot.robot_arm_descriptor_yamls[self.robot.default_arm],
            robot_urdf_path=self.robot.urdf_path,
        )

        arm_links = self.robot.arm_link_names[self.robot.default_arm]
        joint_control_idx = self.robot.arm_control_idx[self.robot.default_arm]
        joint_pos = self.robot.get_joint_positions()[joint_control_idx]
        link_poses = self.fk_solver.get_link_poses(joint_pos, arm_links)

        # Assemble robot meshes
        for link_name, meshes in self.robot_copy.meshes[self.robot_copy_type].items():
            for mesh_name, copy_mesh in meshes.items():
                # Skip grasping frame (this is necessary for Tiago, but should be cleaned up in the future)
                if "grasping_frame" in link_name:
                    continue
                # Set poses of meshes relative to the robot to construct the robot

                link_pose = (
                    link_poses[link_name]
                    if link_name in arm_links
                    else self.robot_copy.links_relative_poses[self.robot_copy_type][link_name]
                )

                mesh_copy_pose = T.pose_transform(
                    *link_pose, *self.robot_copy.relative_poses[self.robot_copy_type][link_name][mesh_name]
                )

                mesh_copy_pose = (mesh_copy_pose[0] + self.offset[0], 
                                  mesh_copy_pose[1] + self.offset[1]) # fix
                self._set_prim_pose(copy_mesh, mesh_copy_pose)

    def _set_prim_pose(self, prim, pose):
        translation = lazy.pxr.Gf.Vec3d(*pose[0].tolist())
        prim.GetAttribute("xformOp:translate").Set(translation)
        orientation = pose[1][[3, 0, 1, 2]]
        prim.GetAttribute("xformOp:orient").Set(lazy.pxr.Gf.Quatd(*orientation.tolist()))

    def _construct_disabled_collision_pairs(self):
        robot_meshes_copy = self.robot_copy.meshes[self.robot_copy_type]

        # Filter out collision pairs of meshes part of the same link
        for meshes in robot_meshes_copy.values():
            for mesh in meshes.values():
                self.disabled_collision_pairs_dict[mesh.GetPrimPath().pathString] = [
                    m.GetPrimPath().pathString for m in meshes.values()
                ]

        # Filter out all self-collisions
        if self.robot_copy_type == "simplified":
            all_meshes = [
                mesh.GetPrimPath().pathString
                for link in robot_meshes_copy.keys()
                for mesh in robot_meshes_copy[link].values()
            ]
            for link in robot_meshes_copy.keys():
                for mesh in robot_meshes_copy[link].values():
                    self.disabled_collision_pairs_dict[mesh.GetPrimPath().pathString] += all_meshes

        # Filter out collision pairs of meshes part of disabled collision pairs
        else:
            for pair in self.robot.disabled_collision_pairs:
                link_1 = pair[0]
                link_2 = pair[1]
                if link_1 in robot_meshes_copy.keys() and link_2 in robot_meshes_copy.keys():
                    for mesh in robot_meshes_copy[link_1].values():
                        self.disabled_collision_pairs_dict[mesh.GetPrimPath().pathString] += [
                            m.GetPrimPath().pathString for m in robot_meshes_copy[link_2].values()
                        ]

                    [m.GetPrimPath().pathString for m in robot_meshes_copy[link_2].values()]
                    [m.GetPrimPath().pathString for m in robot_meshes_copy[link_1].values()]

                    for mesh in robot_meshes_copy[link_2].values():
                        self.disabled_collision_pairs_dict[mesh.GetPrimPath().pathString] += [
                            m.GetPrimPath().pathString for m in robot_meshes_copy[link_1].values()
                        ]

        # Filter out colliders all robot copy meshes should ignore
        disabled_colliders = []

        # Disable original robot colliders so copy can't collide with it
        disabled_colliders += [link.prim_path for link in self.robot.links.values()]
        filter_categories = ["floors", "carpet"]
        for obj in self.env.scene.objects:
            if obj.category in filter_categories:
                disabled_colliders += [link.prim_path for link in obj.links.values()]

        # Disable object in hand
        obj_in_hand = self.robot._ag_obj_in_hand[self.robot.default_arm]
        if obj_in_hand is not None:
            disabled_colliders += [link.prim_path for link in obj_in_hand.links.values()]

        for colliders in self.disabled_collision_pairs_dict.values():
            colliders += disabled_colliders


def plan_arm_motion(robot, end_conf, context, planning_time=30.0, torso_fixed=True):
    """
    Plans an arm motion to a final joint position

    Args:
        robot (BaseRobot): Robot object to plan for
        end_conf (Iterable): Final joint position to plan to
        context (PlanningContext): Context to plan in that includes the robot copy
        planning_time (float): Time to plan for

    Returns:
        Array of arrays: Array of joint positions that the robot should navigate to
    """
    from ompl import base as ob
    from ompl import geometric as ompl_geo

    if torso_fixed:
        joint_control_idx = robot.arm_control_idx[robot.default_arm]
        dim = len(joint_control_idx)
        initial_joint_pos = robot.get_joint_positions()[joint_control_idx]
        control_idx_in_joint_pos = th.arange(dim)
    else:
        joint_control_idx = th.cat([robot.trunk_control_idx, robot.arm_control_idx[robot.default_arm]])
        dim = len(joint_control_idx)
        if "combined" in robot.robot_arm_descriptor_yamls:
            joint_combined_idx = th.cat([robot.trunk_control_idx, robot.arm_control_idx["combined"]])
            initial_joint_pos = th.tensor(robot.get_joint_positions()[joint_combined_idx])
            control_idx_in_joint_pos = th.where(th.isin(joint_combined_idx, joint_control_idx))[0]
        else:
            initial_joint_pos = th.tensor(robot.get_joint_positions()[joint_control_idx])
            control_idx_in_joint_pos = th.arange(dim)

    def state_valid_fn(q, verbose=False):
        joint_pos = initial_joint_pos
        joint_pos[control_idx_in_joint_pos] = th.tensor([q[i] for i in range(dim)])
        return not set_arm_and_detect_collision(context, joint_pos, verbose=verbose)

    # create an SE2 state space
    space = ob.RealVectorStateSpace(dim)

    # set lower and upper bounds
    bounds = ob.RealVectorBounds(dim)
    all_joints = list(robot.joints.values())
    arm_joints = [all_joints[i] for i in joint_control_idx.tolist()]
    for i, joint in enumerate(arm_joints):
        if end_conf[i] > joint.upper_limit:
            end_conf[i] = joint.upper_limit
        if end_conf[i] < joint.lower_limit:
            end_conf[i] = joint.lower_limit
        bounds.setLow(i, float(joint.lower_limit))
        bounds.setHigh(i, float(joint.upper_limit))
    space.setBounds(bounds)

    # create a simple setup object
    ss = ompl_geo.SimpleSetup(space)
    ss.setStateValidityChecker(ob.StateValidityCheckerFn(state_valid_fn))

    si = ss.getSpaceInformation()
    planner = ompl_geo.BITstar(si)
    ss.setPlanner(planner)

    start_conf = robot.get_joint_positions()[joint_control_idx]
    start = ob.State(space)
    for i in range(dim):
        start[i] = float(start_conf[i])

    goal = ob.State(space)
    for i in range(dim):
        goal[i] = float(end_conf[i])
    ss.setStartAndGoalStates(start, goal)

    # if the start pose or the goal pose collides, abort
    if not state_valid_fn(start, verbose=True) or not state_valid_fn(goal, verbose=True):
        return

    # this will automatically choose a default planner with
    # default parameters
    solved = ss.solve(planning_time)

    if solved:
        # try to shorten the path
        ss.simplifySolution()

        sol_path = ss.getSolutionPath()
        return_path = []
        for i in range(sol_path.getStateCount()):
            joint_pos = th.tensor([sol_path.getState(i)[j] for j in range(dim)], dtype=th.float32)
            return_path.append(joint_pos)
        return return_path
    return None

def set_arm_and_detect_collision(context, joint_pos, verbose=False):
    """
    Sets joint positions of the robot and detects robot collisions with the environment and itself

    Args:
        context (PlanningContext): Context to plan in that includes the robot copy
        joint_pos (Array): Joint positions to set the robot to
        verbose (bool): Whether the collision detector should output information about collisions or not. The verbose mode is too noisy in sampling so it is default to False

    Returns:
        bool: Whether the robot is in a valid state i.e. not in collision
    """
    robot_copy = context.robot_copy
    robot_copy_type = context.robot_copy_type

    arm_links = context.robot.arm_link_names[context.robot.default_arm]
    link_poses = context.fk_solver.get_link_poses(joint_pos, arm_links)

    for link in arm_links:
        pose = link_poses[link]
        if link in robot_copy.meshes[robot_copy_type].keys():
            for mesh_name, mesh in robot_copy.meshes[robot_copy_type][link].items():
                relative_pose = robot_copy.relative_poses[robot_copy_type][link][mesh_name]
                mesh_pose = T.pose_transform(*pose, *relative_pose)
                mesh_pose = (mesh_pose[0] + context.offset[0],
                             mesh_pose[1] + context.offset[1])
                translation = lazy.pxr.Gf.Vec3d(*mesh_pose[0].tolist())
                mesh.GetAttribute("xformOp:translate").Set(translation)
                orientation = mesh_pose[1][[3, 0, 1, 2]]
                mesh.GetAttribute("xformOp:orient").Set(lazy.pxr.Gf.Quatd(*orientation.tolist()))

    return detect_robot_collision(context, verbose=verbose)

def detect_robot_collision(context, verbose=False):
    """
    Detects robot collisions

    Args:
        context (PlanningContext): Context to plan in that includes the robot copy
        verbose (bool): Whether the collision detector should output information about collisions or not. The verbose mode is too noisy in sampling so it is default to False

    Returns:
        valid_hit(bool): Whether the robot is in collision
    """
    robot_copy = context.robot_copy
    robot_copy_type = context.robot_copy_type

    # Define function for checking overlap
    valid_hit = False
    mesh_path = None

    def overlap_callback(hit):
        nonlocal valid_hit
        valid_hit = hit.rigid_body not in context.disabled_collision_pairs_dict[mesh_path]

        # if verbose mode is on and overlap is detected, output a warning on the colliding object and robot mesh_path
        if valid_hit and verbose:
            print(
                f"Could not make a plan to get to the target position, colliding objects: {hit.rigid_body} and {mesh_path}",
            )

        if valid_hit:
            return not valid_hit

    for meshes in robot_copy.meshes[robot_copy_type].values():
        for mesh in meshes.values():
            if valid_hit:
                return valid_hit
            mesh_path = mesh.GetPrimPath().pathString
            mesh_id = lazy.pxr.PhysicsSchemaTools.encodeSdfPath(mesh_path)
            if mesh.GetTypeName() == "Mesh":
                print("mesh")
                og.sim.psqi.overlap_mesh(*mesh_id, reportFn=overlap_callback)
                
            else:
                print("shape")
                og.sim.psqi.overlap_shape(*mesh_id, reportFn=overlap_callback)

    return valid_hit

def detect_robot_collision_in_sim(robot, filter_objs=None, ignore_obj_in_hand=True):
    """
    Detects robot collisions with the environment, but not with itself using the ContactBodies API

    Args:
        robot (BaseRobot): Robot object to detect collisions for
        filter_objs (Array of StatefulObject or None): Objects to ignore collisions with
        ignore_obj_in_hand (bool): Whether to ignore collisions with the object in the robot's hand

    Returns:
        bool: Whether the robot is in collision
    """
    if filter_objs is None:
        filter_objs = []

    filter_categories = ["floors"]

    obj_in_hand = robot._ag_obj_in_hand[robot.default_arm]
    if obj_in_hand is not None and ignore_obj_in_hand:
        filter_objs.append(obj_in_hand)

    # Use the RigidCollisionAPI to get the things this robot is colliding with
    scene_idx = robot.scene.idx
    link_paths = set(robot.link_prim_paths)
    collision_body_paths = {
        row
        for row, _ in GripperRigidContactAPI.get_contact_pairs(scene_idx, column_prim_paths=link_paths)
        if row not in link_paths
    }

    # Convert to prim objects and filter out the necessary objects.
    rigid_prims = prim_paths_to_rigid_prims(collision_body_paths, robot.scene)
    return any(o not in filter_objs and o.category not in filter_categories for o, p in rigid_prims)
