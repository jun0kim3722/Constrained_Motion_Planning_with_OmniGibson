"""
Set of utilities for helping to execute robot control
"""

import torch as th
import numpy as np

import json
import omnigibson.lazy as lazy
import omnigibson.utils.transform_utils as T
from scipy.spatial.transform import Rotation as R


class FKSolver:
    """
    Class for thinly wrapping Lula Forward Kinematics solver
    """

    def __init__(
        self,
        robot_description_path,
        robot_urdf_path,
    ):
        # Create robot description and kinematics
        self.robot_description = lazy.lula.load_robot(robot_description_path, robot_urdf_path)
        self.kinematics = self.robot_description.kinematics()

    def get_link_poses(
        self,
        joint_positions,
        link_names,
    ):
        """
        Given @joint_positions, get poses of the desired links (specified by @link_names)

        Args:
            joint positions (n-array): Joint positions in configuration space
            link_names (list): List of robot link names we want to specify (e.g. "gripper_link")

        Returns:
            link_poses (dict): Dictionary mapping each robot link name to its pose
        """
        link_poses = {}
        for link_name in link_names:
            pose3_lula = self.kinematics.pose(joint_positions, link_name)

            # get position
            link_position = th.tensor(pose3_lula.translation, dtype=th.float32)

            # get orientation
            rotation_lula = pose3_lula.rotation
            link_orientation = th.tensor(
                [rotation_lula.x(), rotation_lula.y(), rotation_lula.z(), rotation_lula.w()], dtype=th.float32
            )

            link_poses[link_name] = (link_position, link_orientation)
        return link_poses
    
    def get_link_poses_euler(
        self,
        joint_positions,
        link_names,
    ):
        """
        Given @joint_positions, get poses of the desired links (specified by @link_names)

        Args:
            joint positions (n-array): Joint positions in configuration space
            link_names (list): List of robot link names we want to specify (e.g. "gripper_link")

        Returns:
            link_poses (dict): Dictionary mapping each robot link name to its pose
        """
        link_poses = {}
        for link_name in link_names:
            pose3_lula = self.kinematics.pose(joint_positions, link_name)

            # get position
            link_position = th.tensor(pose3_lula.translation, dtype=th.float32)

            # get orientation
            rotation_lula = pose3_lula.rotation
            link_orientation = T.quat2euler(th.tensor([rotation_lula.x(), rotation_lula.y(), rotation_lula.z(), rotation_lula.w()]))

            link_poses[link_name] = (link_position.numpy(), link_orientation.numpy())
        return link_poses


class IKSolver:
    """
    Class for thinly wrapping Lula IK solver
    """

    def __init__(
        self,
        robot_description_path,
        robot_urdf_path,
        eef_name,
        reset_joint_pos,
        world2robot_homo,
    ):
        # Create robot description, kinematics, and config
        self.robot_description = lazy.lula.load_robot(robot_description_path, robot_urdf_path)
        self.kinematics = self.robot_description.kinematics()
        self.config = lazy.lula.CyclicCoordDescentIkConfig()
        self.eef_name = eef_name
        self.reset_joint_pos = reset_joint_pos
        self.world2robot_homo = world2robot_homo

    def solve(
        self,
        target_pose_homo,
        position_tolerance=0.01,
        orientation_tolerance=0.05,
        position_weight=1.0,
        orientation_weight=0.05,
        max_iterations=150,
        initial_joint_pos=None,
    ):
        """
        Backs out joint positions to achieve desired @target_pos and @target_quat

        Args:
            target_pose_homo (np.ndarray): [4, 4] homogeneous transformation matrix of the target pose in world frame
            position_tolerance (float): Maximum position error (L2-norm) for a successful IK solution
            orientation_tolerance (float): Maximum orientation error (per-axis L2-norm) for a successful IK solution
            position_weight (float): Weight for the relative importance of position error during CCD
            orientation_weight (float): Weight for the relative importance of position error during CCD
            max_iterations (int): Number of iterations used for each cyclic coordinate descent.
            initial_joint_pos (None or n-array): If specified, will set the initial cspace seed when solving for joint
                positions. Otherwise, will use self.reset_joint_pos

        Returns:
            ik_results (lazy.lula.CyclicCoordDescentIkResult): IK result object containing the joint positions and other information.
        """
        # convert target pose to robot base frame
        target_pose_robot = np.dot(self.world2robot_homo, target_pose_homo)
        target_pose_pos = target_pose_robot[:3, 3]
        target_pose_rot = target_pose_robot[:3, :3]
        ik_target_pose = lazy.lula.Pose3(lazy.lula.Rotation3(target_pose_rot), target_pose_pos)
        
        # Set the cspace seed and tolerance
        initial_joint_pos = self.reset_joint_pos if initial_joint_pos is None else np.array(initial_joint_pos)
        self.config.cspace_seeds = [initial_joint_pos]
        self.config.position_tolerance = position_tolerance
        self.config.orientation_tolerance = orientation_tolerance
        self.config.ccd_position_weight = position_weight
        self.config.ccd_orientation_weight = orientation_weight
        self.config.max_num_descents = max_iterations
        # Compute target joint positions
        ik_results = lazy.lula.compute_ik_ccd(self.kinematics, ik_target_pose, self.eef_name, self.config)
        if ik_results.success:
            return th.tensor(ik_results.cspace_position, dtype=th.float32)
        else:
            return None
    
    def get_grasp(self, obj, offset=0.05):
        # read grasp
        grasp_file = "/data/og_dataset/objects/" + obj.category + "/" + obj.model + "/usd/grasp_results.json"
        try:
            with open(grasp_file, 'r') as f:
                grasp_data = json.load(f)

        except:
            print("GRASP: {} does not exist!!".format(grasp_file))

        # 
        obj_loc = obj.get_position_orientation()
        center_offset = obj.get_position_orientation()[0] - obj.aabb_center + th.tensor([0, 0, obj.aabb_extent[2] / 2.0])
        obj_tran = obj_loc[0] - center_offset
        obj_rot = obj_loc[1]
        for grasp in grasp_data:
            # calc grasping location
            eef_pos = T.quat_apply(obj_loc[1], th.tensor(grasp["pos"])) + obj_tran

            grasp_quat = th.tensor(grasp['quat'])
            eef_quat = T.quat_multiply(T.quat_multiply(grasp_quat, th.tensor([0.5, 0.5, 0.5, 0.5])), obj_rot)

            # calc offset location
            off_set_pos = th.tensor(grasp["pos"]) + T.quat_apply(grasp_quat, th.tensor([-offset, 0.0, 0.0]))
            eef_offset_pos = T.quat_apply(obj_loc[1], off_set_pos) + obj_tran

            # solve IK
            target_pose_homo = T.pose2mat([eef_pos, eef_quat])
            joint_pos = self.solve(target_pose_homo = target_pose_homo)

            offset_pose_homo = T.pose2mat([eef_offset_pos, eef_quat])
            joint_offset_pos = self.solve(target_pose_homo = target_pose_homo)

            if joint_pos is not None and joint_offset_pos is not None:
                return joint_offset_pos, joint_pos
        
        return None

# def xyz2omni(rot, as_quat=True):
#     return [-np.pi/2, np.pi/2, 0]
#     # return [np.pi/2, np.pi/2, 0]
#         #  -x | y | Z   
#     # return [rot[1], rot[2], rot[0]]
#     # return [-1 * (rot[1] - np.pi/2), rot[2] - np.pi/2, rot[0]]

#     new_rot = R.from_euler("xyz", [-1 * (rot[1] - np.pi/2), rot[2], rot[0]])

#     if as_quat:
#         return new_rot.as_quat()
#     else:
#         return new_rot.as_euler("xyz")

@th.jit.script
def orientation_error(desired, current):
    """
    This function calculates a 3-dimensional orientation error vector for use in the
    impedance controller. It does this by computing the delta rotation between the
    inputs and converting that rotation to exponential coordinates (axis-angle
    representation, where the 3d vector is axis * angle).
    See https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation for more information.
    Optimized function to determine orientation error from matrices

    Args:
        desired (tensor): (..., 3, 3) where final two dims are 2d array representing target orientation matrix
        current (tensor): (..., 3, 3) where final two dims are 2d array representing current orientation matrix
    Returns:
        tensor: (..., 3) where final dim is (ax, ay, az) axis-angle representing orientation error
    """
    # Compute batch size
    batch_size = desired.numel() // 9  # Each 3x3 matrix has 9 elements

    desired_flat = desired.reshape(batch_size, 3, 3)
    current_flat = current.reshape(batch_size, 3, 3)

    rc1, rc2, rc3 = current_flat[:, :, 0], current_flat[:, :, 1], current_flat[:, :, 2]
    rd1, rd2, rd3 = desired_flat[:, :, 0], desired_flat[:, :, 1], desired_flat[:, :, 2]

    error = 0.5 * (th.linalg.cross(rc1, rd1) + th.linalg.cross(rc2, rd2) + th.linalg.cross(rc3, rd3))

    return error.reshape(desired.shape[:-2] + (3,))