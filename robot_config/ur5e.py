import sys
import os

sys.path.insert(0, os.path.join(os.getcwd(), 'assets'))

from importlib.resources import files
import torch as th

from omnigibson.macros import gm
from omnigibson.robots.manipulation_robot import GraspingPoint, ManipulationRobot
from omnigibson.utils.transform_utils import euler2quat
from omnigibson.utils.python_utils import classproperty


class UR5e(ManipulationRobot):
    """
    The Universal Robotics UR5e robot
    """

    def __init__(
        self,
        # Shared kwargs in hierarchy
        name,
        relative_prim_path=None,
        scale=None,
        visible=True,
        visual_only=False,
        self_collisions=True,
        load_config=None,
        fixed_base=True,
        # Unique to USDObject hierarchy
        abilities=None,
        # Unique to ControllableObject hierarchy
        control_freq=None,
        controller_config=None,
        action_type="continuous",
        action_normalize=True,
        reset_joint_pos=None,
        # Unique to BaseRobot
        obs_modalities=("rgb", "proprio"),
        proprio_obs="default",
        sensor_config=None,
        # Unique to ManipulationRobot
        grasping_mode="physical",
        # Unique to UR5e
        end_effector="robotiq_2f85",
        **kwargs,
    ):
        """
        Args:
            name (str): Name for the object. Names need to be unique per scene
            relative_prim_path (str): Scene-local prim path of the Prim to encapsulate or create.
            scale (None or float or 3-array): if specified, sets either the uniform (float) or x,y,z (3-array) scale
                for this object. A single number corresponds to uniform scaling along the x,y,z axes, whereas a
                3-array specifies per-axis scaling.
            visible (bool): whether to render this object or not in the stage
            visual_only (bool): Whether this object should be visual only (and not collide with any other objects)
            self_collisions (bool): Whether to enable self collisions for this object
            load_config (None or dict): If specified, should contain keyword-mapped values that are relevant for
                loading this prim at runtime.
            abilities (None or dict): If specified, manually adds specific object states to this object. It should be
                a dict in the form of {ability: {param: value}} containing object abilities and parameters to pass to
                the object state instance constructor.
            control_freq (float): control frequency (in Hz) at which to control the object. If set to be None,
                we will automatically set the control frequency to be at the render frequency by default.
            controller_config (None or dict): nested dictionary mapping controller name(s) to specific controller
                configurations for this object. This will override any default values specified by this class.
            action_type (str): one of {discrete, continuous} - what type of action space to use
            action_normalize (bool): whether to normalize inputted actions. This will override any default values
                specified by this class.
            reset_joint_pos (None or n-array): if specified, should be the joint positions that the object should
                be set to during a reset. If None (default), self._default_joint_pos will be used instead.
                Note that _default_joint_pos are hardcoded & precomputed, and thus should not be modified by the user.
                Set this value instead if you want to initialize the robot with a different rese joint position.
            obs_modalities (str or list of str): Observation modalities to use for this robot. Default is ["rgb", "proprio"].
                Valid options are "all", or a list containing any subset of omnigibson.sensors.ALL_SENSOR_MODALITIES.
                Note: If @sensor_config explicitly specifies `modalities` for a given sensor class, it will
                    override any values specified from @obs_modalities!
            proprio_obs (str or list of str): proprioception observation key(s) to use for generating proprioceptive
                observations. If str, should be exactly "default" -- this results in the default proprioception
                observations being used, as defined by self.default_proprio_obs. See self._get_proprioception_dict
                for valid key choices
            sensor_config (None or dict): nested dictionary mapping sensor class name(s) to specific sensor
                configurations for this object. This will override any default values specified by this class.
            grasping_mode (str): One of {"physical", "assisted", "sticky"}.
                If "physical", no assistive grasping will be applied (relies on contact friction + finger force).
                If "assisted", will magnetize any object touching and within the gripper's fingers.
                If "sticky", will magnetize any object touching the gripper's fingers.
            end_effector (str): type of end effector to use. One of {"gripper", "allegro", "leap_right", "leap_left", "inspire"}
            kwargs (dict): Additional keyword arguments that are used for other super() calls from subclasses, allowing
                for flexible compositions of various object subclasses (e.g.: Robot is USDObject + ControllableObject).
        """
        # store end effector information
        self.end_effector = end_effector
        if end_effector == "robotiq_2f85":
            self._model_name = "ur5e_robotiq_2f85"
            self._gripper_control_idx = 6
            self._eef_link_names = "ee_link"
            self._finger_link_names = ["left_inner_finger_pad", "right_inner_finger_pad"]
            self._finger_joint_names = ["finger_joint", "right_outer_knuckle_joint"]
            # the 6 joint angles, plut 6-dof for the gripper. technically the
            # real dof of the gripper is 2, but isaac sim or omnigibson fail to
            # properly consider over-constrained mechanisms, such as the
            # robotiq-85 gripper.
            self._default_robot_model_joint_pos = th.tensor([0.00, -2.2, 1.9, -1.383, -1.57, 0.00,
                                                             0.00, 0.00, 0.00, 0.00, 0.00, 0.00])
            self._teleop_rotation_offset = th.tensor([-1, 0, 0, 0])
            # in assisted grasp mode, objects are detecting by raycasting from
            # every start point to every end point. for now, defining the points
            # at 2 opposing corners for each finger pad. these numbers are
            # pretty close to the exact corners of the finger pads.
            self._ag_start_points = [
                # for future reference, in the local frame of the left finger pad:
                # -y -> toward the center of the grasp
                # +z -> away from the base of the gripper
                # +x -> toward the bottom of the gripper if the plane the
                # fingers are in is parallel to floor
                GraspingPoint(link_name="left_inner_finger_pad", position=th.tensor([0.010, -0.004, 0.018])),
                # GraspingPoint(link_name="left_inner_finger_pad", position=th.tensor([-0.010, -0.004, -0.018])),
            ]
            self._ag_end_points = [
                GraspingPoint(link_name="right_inner_finger_pad", position=th.tensor([-0.010, 0.004, 0.018])),
                # GraspingPoint(link_name="right_inner_finger_pad", position=th.tensor([0.010, 0.004, -0.018])),
            ]
        elif end_effector == None:
            self._model_name = "ur5e"
            self._gripper_control_idx = None
            self._eef_link_names = "ee_link"
            self._finger_link_names = []
            self._finger_joint_names = []
            # the 6 joint angles
            self._default_robot_model_joint_pos = th.tensor([0.00, -2.2, 1.9, -1.383, -1.57, 0.00])
            self._teleop_rotation_offset = th.tensor([-1, 0, 0, 0])
            self._ag_start_points = None
            self._ag_end_points = None
        else:
            raise ValueError(f"End effector {end_effector} not supported for UR5e")

        # Run super init
        super().__init__(
            relative_prim_path=relative_prim_path,
            name=name,
            scale=scale,
            visible=visible,
            fixed_base=fixed_base,
            visual_only=visual_only,
            self_collisions=self_collisions,
            load_config=load_config,
            abilities=abilities,
            control_freq=control_freq,
            controller_config=controller_config,
            action_type=action_type,
            action_normalize=action_normalize,
            reset_joint_pos=reset_joint_pos,
            obs_modalities=obs_modalities,
            proprio_obs=proprio_obs,
            sensor_config=sensor_config,
            grasping_mode=grasping_mode,
            grasping_direction=(
                "lower" if end_effector == "gripper" else "upper"
            ),  # gripper grasps in the opposite direction
            **kwargs,
        )

    @property
    def model_name(self):
        # Override based on specified UR5e variant
        return self._model_name

    @property
    def discrete_action_list(self):
        raise NotImplementedError()

    def _create_discrete_action_space(self):
        raise ValueError("UR5e does not support discrete actions!")

    @property
    def controller_order(self):
        return ["arm_{}".format(self.default_arm), "gripper_{}".format(self.default_arm)]

    @property
    def _default_controllers(self):
        controllers = super()._default_controllers
        controllers["arm_{}".format(self.default_arm)] = "JointController"
        controllers["gripper_{}".format(self.default_arm)] = "MultiFingerGripperController"
        return controllers

    @property
    def _default_joint_pos(self):
        return self._default_robot_model_joint_pos

    @property
    def finger_lengths(self):
        return {self.default_arm: 0.1}

    @property
    def arm_link_names(self):
        return {self.default_arm: ['base_link', 'shoulder_link', 'upper_arm_link', 'forearm_link',
                                   'wrist_1_link', 'wrist_2_link', 'wrist_3_link', "ee_link",
                                   'robotiq_arg2f_base_link', 'left_outer_knuckle',
                                   'left_outer_finger', 'left_inner_finger', 'left_inner_finger_pad',
                                   'left_inner_knuckle', 'right_inner_knuckle', 'right_outer_knuckle',
                                   'right_outer_finger', 'right_inner_finger', 'right_inner_finger_pad']}

    @property
    def arm_joint_names(self):
        return {self.default_arm: ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']}

    @property
    def eef_link_names(self):
        return {self.default_arm: self._eef_link_names}

    @property
    def finger_link_names(self):
        return {self.default_arm: self._finger_link_names}

    @property
    def finger_joint_names(self):
        return {self.default_arm: self._finger_joint_names}

    @property
    def usd_path(self):
        return os.path.join(gm.ASSET_PATH, "models/ur5e/ur5e_robotiq_2f85/ur5e_robotiq_2f85.usd")

    @property
    def robot_arm_descriptor_yamls(self):
        return {self.default_arm: os.path.join(gm.ASSET_PATH, "models/ur5e/ur5e_descriptor.yaml")}

    @property
    def urdf_path(self):
        return os.path.join(gm.ASSET_PATH, "models/ur5e/ur5e_robotiq_2f85.urdf")

    @property
    def curobo_path(self):
        breakpoint()
        # Only supported for normal franka now
        assert (
            self._model_name == "franka_panda"
        ), f"Only franka_panda is currently supported for curobo. Got: {self._model_name}"
        return os.path.join(gm.ASSET_PATH, f"models/franka/{self.model_name}_descriptor_curobo.yaml")

    @property
    def eef_usd_path(self):
        breakpoint()
        return {self.default_arm: os.path.join(gm.ASSET_PATH, f"models/franka/{self.model_name}_eef.usd")}

    @property
    def teleop_rotation_offset(self):
        return {self.default_arm: self._teleop_rotation_offset}

    @property
    def assisted_grasp_start_points(self):
        return {self.default_arm: self._ag_start_points}

    @property
    def assisted_grasp_end_points(self):
        return {self.default_arm: self._ag_end_points}

    @property
    def disabled_collision_pairs(self):
        return [["base_link_inertia", "shoulder_link"],
                ['base_link', 'shoulder_link'],
                ['shoulder_link', 'upper_arm_link'],
                ['upper_arm_link', 'forearm_link'],
                ['forearm_link', 'wrist_1_link'],
                ['wrist_1_link', 'wrist_2_link'],
                ['wrist_2_link', 'wrist_3_link'],
                ["robotiq_arg2f_base_link", 'wrist_3_link'],
                ["robotiq_arg2f_base_link", "left_outer_knuckle"],
                ["robotiq_arg2f_base_link", "left_inner_knuckle"],
                ["robotiq_arg2f_base_link", "right_outer_knuckle"],
                ["robotiq_arg2f_base_link", "right_inner_knuckle"],
                ["left_outer_knuckle", "left_outer_finger"],
                ["robotiq_arg2f_base_link", "left_outer_knuckle"],
                ["left_inner_finger_pad", "left_inner_finger"],
                ["left_inner_finger", "left_outer_finger"],
                ["right_outer_knuckle", "right_outer_finger"],
                ["right_inner_knuckle", "right_inner_finger"],
                ["right_inner_knuckle", "right_outer_finger"],
                ["left_inner_finger", "left_inner_knuckle"],
                ["right_inner_finger", "right_outer_finger"],
                ["right_inner_finger_pad", "right_inner_finger"]
                ]
