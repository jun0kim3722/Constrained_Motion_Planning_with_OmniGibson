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
from omnigibson.utils.usd_utils import delete_or_deactivate_prim

from omnigibson.robots import *
from grasp_utils.kinematic import FKSolver

from ompl import base as ob
from ompl import geometric as omplg




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
    
    def __init__(self, og_robot, robot_joints=None, obj=None):
        self.fk_solver = FKSolver(
            robot_description_path=og_robot.robot_arm_descriptor_yamls[og_robot.default_arm],
            robot_urdf_path=og_robot.urdf_path,
        )

        self.prims = {}
        self.meshes = {}
        self.relative_poses = {}
        self.links_relative_poses = {}
        self.reset_pose = {
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
            robot_to_copy = rc["robot"]

            # Copy robot meshes
            for link in robot_to_copy.links.values():
                link_name = link.prim_path.split("/")[-1]

                for mesh_name, mesh in link.collision_meshes.items():
                    split_path = mesh.prim_path.split("/")
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

                copy_robot_links_relative_poses[link_name] = T.relative_pose_transform(
                    *link.get_position_orientation(), *og_robot.get_position_orientation()
                )

            if obj:
                from pxr import UsdPhysics

                # Attach object meshes to a specific robot copy link
                ee_link_name = og_robot.eef_link_names[og_robot.default_arm]  # Or use og_robot.eef_link_names[og_robot.default_arm]
                target_link_path = rc["copy_path"] + "/" + ee_link_name
                lazy.omni.usd.commands.CreatePrimCommand("Xform", target_link_path).do()

                if ee_link_name not in copy_robot_meshes:
                    copy_robot_meshes[ee_link_name] = {}
                    copy_robot_meshes_relative_poses[ee_link_name] = {}

                obj_idx = 0
                for link in obj.links.values():
                    for mesh_name, mesh in link.collision_meshes.items():
                        if not UsdPhysics.CollisionAPI(mesh.prim).GetCollisionEnabledAttr().Get():
                            continue  # Skip non-collision shapes

                        # Create unique name under EE link
                        copy_mesh_path = f"{target_link_path}/attached_obj_{obj_idx}"
                        lazy.omni.usd.commands.CopyPrimCommand(mesh.prim_path, path_to=copy_mesh_path).do()

                        copy_mesh = lazy.omni.isaac.core.utils.prims.get_prim_at_path(copy_mesh_path)

                        if robot_joints is not None:
                            ee_poses = self.fk_solver.get_link_poses(robot_joints, [ee_link_name])[ee_link_name]
                            offset_poses = (
                                ee_poses[0] + og_robot.get_position_orientation()[0],
                                T.quat_multiply(ee_poses[1], og_robot.get_position_orientation()[1])
                            )

                            relative_pose = T.relative_pose_transform(
                                *mesh.get_position_orientation(),
                                *offset_poses
                            )

                        else:
                            relative_pose = T.relative_pose_transform(
                                *mesh.get_position_orientation(),
                                *og_robot.links[ee_link_name].get_position_orientation()
                            )

                        copy_robot_meshes[ee_link_name][f"attached_obj_{obj_idx}"] = copy_mesh
                        copy_robot_meshes_relative_poses[ee_link_name][f"attached_obj_{obj_idx}"] = relative_pose
                        obj_idx += 1

            self.prims[robot_type] = copy_robot
            self.meshes[robot_type] = copy_robot_meshes
            self.relative_poses[robot_type] = copy_robot_meshes_relative_poses
            self.links_relative_poses[robot_type] = copy_robot_links_relative_poses

        og.sim.step()

    def remove(self):
        copy_path = self.prims["original"].GetPath().pathString
        if not delete_or_deactivate_prim(copy_path):
            og.log.error("Copy Robot Delete Failed!!!!")

class PlanningContext(object):
    """
    A context manager that sets up a robot copy for collision checking in planning.
    """

    def __init__(self, env, robot, robot_joints=None, in_hand_obj=None,
                 disabled_collision_pairs_dict={}, robot_copy_type="original"):
        self.env = env
        self.robot = robot

        self.in_hand_obj = in_hand_obj
        self.robot_joints = robot_joints
        robot_copy = RobotCopy(robot, robot_joints, in_hand_obj)
        self.robot_copy = robot_copy

        self.offset = (self.robot.get_position_orientation()[0] - self.robot_copy.reset_pose['original'][0],
                       self.robot.get_position_orientation()[1] - self.robot_copy.reset_pose['original'][1])
        
        self.robot_copy_type = robot_copy_type if robot_copy_type in robot_copy.prims.keys() else "original"
        self.disabled_collision_pairs_dict = disabled_collision_pairs_dict

        self._assemble_robot_copy()
        self._construct_disabled_collision_pairs()

        robot_prim = robot_copy.prims['original']
        self.collision_prims = self.get_collision_enabled_prims(robot_prim)

    def __enter__(self):
        self._assemble_robot_copy()
        self._construct_disabled_collision_pairs()
        return self

    def __exit__(self, *args):
        self.robot_copy.remove()
        

    def _assemble_robot_copy(self):
        self.fk_solver = FKSolver(
            robot_description_path=self.robot.robot_arm_descriptor_yamls[self.robot.default_arm],
            robot_urdf_path=self.robot.urdf_path,
        )

        arm_links = self.robot.arm_link_names[self.robot.default_arm]
        if self.robot_joints is None:
            joint_control_idx = self.robot.arm_control_idx[self.robot.default_arm]
            joint_pos = self.robot.get_joint_positions()[joint_control_idx]
            link_poses = self.fk_solver.get_link_poses(joint_pos, arm_links)
        else:
            link_poses = self.fk_solver.get_link_poses(self.robot_joints, arm_links)

        # Assemble robot meshes
        for link_name, meshes in self.robot_copy.meshes[self.robot_copy_type].items():
            for mesh_name, copy_mesh in meshes.items():
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
                                  mesh_copy_pose[1] + self.offset[1]) # fix !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

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
        if self.in_hand_obj is not None:
            # ignore collision with in-hand object
            disabled_colliders += [link.prim_path for link in self.in_hand_obj.links.values()]

            # ignore collision with griper and copy in-hand object
            obj_meshes = [link.GetPrimPath().pathString for link in robot_meshes_copy['ee_link'].values()]
            robot_meshes = []
            for robot_link in robot_meshes_copy.keys():
                if 'finger' in robot_link or 'knuckle' in robot_link:
                    for mesh in robot_meshes_copy[robot_link].values():
                        self.disabled_collision_pairs_dict[mesh.GetPrimPath().pathString] += obj_meshes
                        robot_meshes.append(mesh.GetPrimPath().pathString)
        
            for obj_mesh_path in obj_meshes:
                self.disabled_collision_pairs_dict[obj_mesh_path] += robot_meshes

        for colliders in self.disabled_collision_pairs_dict.values():
            colliders += disabled_colliders
        
    def get_collision_enabled_prims(self, root_prim):
        from pxr import Usd, UsdGeom, UsdPhysics
        stack = [root_prim]
        collision_prims = []

        while stack:
            prim = stack.pop()
            if not prim.IsValid():
                continue
            # Check if CollisionAPI is applied and enabled
            if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
                collision_prims.append(prim)
            stack.extend(list(prim.GetChildren()))

        return collision_prims

    def set_arm_and_detect_collision(self, joint_pos, verbose=False):
        """
        Sets joint positions of the robot and detects robot collisions with the environment and itself

        Args:
            joint_pos (Array): Joint positions to set the robot to
            verbose (bool): Whether the collision detector should output information about collisions or not. The verbose mode is too noisy in sampling so it is default to False

        Returns:
            bool: Whether the robot is in a valid state i.e. not in collision
        """
        robot_copy = self.robot_copy
        robot_copy_type = self.robot_copy_type

        arm_links = self.robot.arm_link_names[self.robot.default_arm]
        link_poses = self.fk_solver.get_link_poses(joint_pos, arm_links)

        for link in arm_links:
            pose = link_poses[link]
            if link in robot_copy.meshes[robot_copy_type].keys():
                for mesh_name, mesh in robot_copy.meshes[robot_copy_type][link].items():
                    relative_pose = robot_copy.relative_poses[robot_copy_type][link][mesh_name]
                    mesh_pose = T.pose_transform(*pose, *relative_pose)
                    mesh_pose = (mesh_pose[0] + self.offset[0],
                                mesh_pose[1] + self.offset[1])
                    translation = lazy.pxr.Gf.Vec3d(*mesh_pose[0].tolist())
                    mesh.GetAttribute("xformOp:translate").Set(translation)
                    orientation = mesh_pose[1][[3, 0, 1, 2]]
                    mesh.GetAttribute("xformOp:orient").Set(lazy.pxr.Gf.Quatd(*orientation.tolist()))

        return self.detect_robot_collision(verbose=verbose)

    def detect_robot_collision(self, verbose=False):
        """
        Detects robot collisions

        Args:
            verbose (bool): Whether the collision detector should output information about collisions or not. The verbose mode is too noisy in sampling so it is default to False

        Returns:
            valid_hit(bool): Whether the robot is in collision
        """
        valid_hit = False
        mesh_path = None
        def overlap_callback(hit):
            nonlocal valid_hit
            if mesh_path not in self.disabled_collision_pairs_dict:
                valid_hit = True
            else:
                valid_hit = hit.rigid_body not in self.disabled_collision_pairs_dict[mesh_path]

            if valid_hit and verbose:
                print(f"[Collision] {mesh_path} collides with {hit.rigid_body}")
            return not valid_hit

        for prim in self.collision_prims:
            if valid_hit:
                return valid_hit

            mesh_path = prim.GetPath().pathString
            mesh_id = lazy.pxr.PhysicsSchemaTools.encodeSdfPath(mesh_path)

            if prim.GetTypeName() == "Mesh":
                og.sim.psqi.overlap_mesh(*mesh_id, reportFn=overlap_callback)
            else:
                og.sim.psqi.overlap_shape(*mesh_id, reportFn=overlap_callback)

        return valid_hit

class ArmValidAll(ob.StateValidityChecker):
    def __init__(self, si, context):
        super().__init__(si)
        self.context = context
        robot = context.robot
        self.dim = len(robot.arm_control_idx[robot.default_arm])

    def isValid(self, dof_state, debug=False):
        joint_pos = th.tensor([dof_state[i] for i in range(self.dim)])
        return not self.context.set_arm_and_detect_collision(joint_pos, debug)

class ArmPlanner():
    def __init__(self, context):

        robot = context.robot
        self.context = context
        self.joint_control_idx = robot.arm_control_idx[robot.default_arm]
        self.dim = len(self.joint_control_idx)

        joint_limits = zip(robot.joint_lower_limits[:self.dim].tolist(), robot.joint_upper_limits[:self.dim].tolist())
        
        self.space_ = ob.RealVectorStateSpace(0)
        for lower_bound, upper_bound in joint_limits:
            self.space_.addDimension(lower_bound, upper_bound)

        self.si_ = ob.SpaceInformation(self.space_)


    def plan(self, goal_joints, context, planning_time=30.0):
        start_conf = context.robot.get_joint_positions()[self.joint_control_idx]
        start = ob.State(self.space_)
        for i in range(self.dim):
            start[i] = float(start_conf[i])

        goal = ob.State(self.space_)
        for i in range(self.dim):
            goal[i] = float(goal_joints[i])

        validityChecker = ArmValidAll(self.si_, context)
        self.si_.setStateValidityChecker(validityChecker)
        self.si_.setStateValidityCheckingResolution(0.0005)
        self.si_.setup()

        if not validityChecker.isValid(start, True) or not validityChecker.isValid(goal, True):
            og.log.warning("Invalid Start or Goal from ArmPlanner")
            return None

        pdef = ob.ProblemDefinition(self.si_)
        pdef.setStartAndGoalStates(start, goal)
        shortestPathObjective = ob.PathLengthOptimizationObjective(self.si_)
        pdef.setOptimizationObjective(shortestPathObjective)

        optimizingPlanner = omplg.RRTConnect(self.si_)
        optimizingPlanner.setProblemDefinition(pdef)

        optimizingPlanner.setRange(1000000)
        optimizingPlanner.setup()
        
        temp_res = optimizingPlanner.solve(planning_time)

        if temp_res.asString() == 'Exact solution':
            path = pdef.getSolutionPath()

            path_simp = omplg.PathSimplifier(self.si_)

            res = path_simp.reduceVertices(path)

            path_list = []

            for t in range(path.getStateCount()):
                state = path.getState(t)
                path_list.append([state[0], state[1], state[2], state[3], state[4], state[5]])

            return path_list
        else:
            return None
        