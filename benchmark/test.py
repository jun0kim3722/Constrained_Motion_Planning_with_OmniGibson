import omnigibson as og

import torch as th
import numpy as np
import json

from omnigibson.utils.grasping_planning_utils import get_grasp_poses_for_object_sticky, get_grasp_position_for_open
import omnigibson.utils.transform_utils as T

from grasp_utils.kinematic import IKSolver

from motion_planner import motion_plan_utils
from motion_planner.constrained_planner import ArmCcontrainedPlanner
from collections import OrderedDict

GRASP_DIST = 0.15

def execute_controller(env, joints, is_griper_open):
    ctr = np.concatenate((joints, [1 if is_griper_open else -1]), dtype="float32")
    action = OrderedDict([('UR5e', ctr)])
    env.step(action)
    # env.step(action)
    # env.step(action)

def execute_motion(env, joint_path, is_griper_open):
    for joints in joint_path:
        execute_controller(env, joints, is_griper_open)

def set_robot_to(env, robot, joint_angles, is_griper_open):
    joint_angles = th.tensor(joint_angles)
    while (robot.get_joint_positions()[:6] - joint_angles > 0.02).any():
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

cfg = dict()
# Define scene
cfg["scene"] = {
    # "type" : "Scene"
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
        "position": [-2.37511, -2.49872, 0.95],
        # "position": [-3.15968, -2.52066, 0.96],
    },
    {
        "type": "DatasetObject",
        "name": "teacup",
        "category": "teacup",
        "model": "vckahe",
        "position": [-2.37511, -2.49872, 0.95],
    },
    
    # {
    #     "type": "DatasetObject",
    #     "name": "knife",
    #     "category": "carving_knife",
    #     "model": "alekva",
    #     "position": [-2.34713, -2.06784, 0.95],
    # },
    # {
    #     "type": "DatasetObject",
    #     "name": "tablespoon",
    #     "category": "tablespoon",
    #     "model": "huudhe",
    #     "position": [-2.37511, -2.49872, 0.95],
    # },
    # {
    #     "type": "DatasetObject",
    #     "name": "sugar jar",
    #     "category": "jar_of_sugar",
    #     "model": "pnbbfb",
    #     "position": [0, -0.5, 1.0],
    # },
    # {
    #     "type": "PrimitiveObject",
    #     "name": "box",
    #     "primitive_type": "Cube",
    #     "rgba": [1.0, 0, 0, 1.0],
    #     "size": 0.20,
    #     "position": [-2.34713, -2.06784, 0.95],
    # }
]

# Define robots
cfg["robots"] = [
    {
        "type": "UR5e",
        "name": "UR5e",
        "obs_modalities": ["rgb", "depth"],
        # "position": [-3.056, -2.04226, 1.03],
        "position": [-3.056, -2.04226, 0.83],
        # "orientation": [ 0, 0, -0.7071068, 0.7071068],
        # "position": [0,0,0],
        "action_normalize": False,
        # "grasping_mode" : "assisted",
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
env.scene.show_collision_mesh = True

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
# tablespoon = env.scene.object_registry("name", "tablespoon")


# teacup_pos = get_grasp_poses_for_object_sticky(teacup)[0][0][0]# - th.tensor([-3.056, -2.14226, 0.63268])
# knife_pos = get_grasp_poses_for_object_sticky(knife)[0][0][0] - th.tensor([-3.056, -2.14226, 0.63268])
# tablespoon_pos = get_grasp_position_for_open(robot, tablespoon, True)[0][0][0] - th.tensor([-3.056, -2.14226, 0.63268])


# teacup_joints = plan_grasp(ik_solver, teacup)
# knife_joints = plan_grasp(ik_solver, knife)
# tablespoon_joints = plan_grasp(ik_solver, tablespoon)

# print("teacup_joints", teacup_joints, teacup_pos)
# print("knife_joints", knife_joints, knife_pos)
# print("tablespoon_joints", tablespoon_joints, tablespoon_pos)

start_joints = [-0.82790005,-0.9197,1.4702001,-0.4496,1.3429,-3.1407]
# goal_joints = np.array([0.4566701,-1.0224016,1.4694293,-0.44907653,1.3427804,-3.1414886])
goal_joints = np.array([0.51173443, -1.12711413,  1.44211166, -1.88738625, -1.57079507, 0.51332733]) # cup drop place


# cup_pos = teacup.get_position_orientation()
# breakpoint()
cup_pos = teacup.get_position_orientation()
# grasp_pos = cup_pos[0]
# grasp_pos[2] += GRASP_DIST

offset_joints, grasp_joints = ik_solver.get_grasp(knife)

# breakpoint()

# grasp_pos = cup_pos[0] + th.tensor([ 0.04549949, -0.0215362 ,  0.31178942])

# pick_pos = T.pose2mat([grasp_pos, T.euler2quat(th.tensor([-np.pi,  0.0, 0.0]))])
# # pick_pos = T.pose2mat([cup_pos[0], T.euler2quat(th.tensor([-2.98777613,  0.12833831, -1.56089303]))])
# start_joints = ik_solver.solve(target_pose_homo = pick_pos)



breakpoint()
set_robot_to(env, robot, offset_joints, is_griper_open=True)

# for i in range(100):
while True:
    action = env.action_space.sample()
    angle = np.array(start_joints)
    action['UR5e'] = np.concatenate((angle, [1]), dtype="float32")
    env.step(action)
    print(robot.get_joint_positions()[:6])
action['UR5e'] = np.concatenate((angle, [-1]), dtype="float32")
env.step(action)

# # for i in range(15):
# while True:
#     env.step(action)
# action['UR5e'] = np.concatenate((angle, [0]), dtype="float32")
# while True:
#     env.step(action)


# motion planning
path = None
with motion_plan_utils.PlanningContext(env, robot, teacup) as context:
    # adj_start_joints = start_joints
    # adj_start_joints[1] -= 0.01
    # while True:
    #     # angle = np.random.uniform(-np.pi, np.pi)
    #     # context.set_arm_and_detect_collision([0 ,-0.8224016,1.4694293,-0.44907653,1.3427804,-3.1414886], verbose=False)
    #     # context.set_arm_and_detect_collision(goal_joints, verbose=False)
    #     og.sim.step()

    # set planner
    griper_pos, griper_rot = context.fk_solver.get_link_poses_euler(start_joints, [robot._eef_link_names])[robot._eef_link_names]
    rot_const = griper_rot
    rot_const[-1] = None
    acp = ArmCcontrainedPlanner(context, trans_const=None, rot_const=rot_const, num_const=2, tolerance=np.deg2rad(15.0))

    adj_start_joints = start_joints
    adj_start_joints[1] -= 0.01
    path = acp.plan(robot, start_joints.tolist(), goal_joints.tolist(), context, planning_time=120.0)

    if path:
        path = get_pose_from_path(path)
        path.insert(0, start_joints)

# path = [
#     [-0.82790005,-1.0197,1.4702001,-0.4496,1.3429,-3.1407],
#     [-0.83388495,-1.0522836,1.5065811,-0.44349375,1.3589829,-3.1289241],
#     [-0.8417641,-1.086599,1.536552,-0.43698886,1.3768646,-3.1180687],
#     [-0.8508124,-1.1203152,1.5626765,-0.42998034,1.3960474,-3.1081045],
#     [-0.8610892,-1.1525712,1.586761,-0.42248696,1.4163418,-3.0988176],
#     [-0.8728719,-1.1828986,1.6097721,-0.41460252,1.4374721,-3.090084],
#     [-0.8862041,-1.211077,1.6321682,-0.40643975,1.4589659,-3.081875],
#     [-0.90050834,-1.2375357,1.6544386,-0.3979503,1.4804044,-3.0741153],
#     [-0.9144357,-1.2637037,1.6777906,-0.38865438,1.5018742,-3.0665188],
#     [-0.92557645,-1.2915704,1.7041695,-0.37761864,1.5237343,-3.0587604],
#     [-0.9318793,-1.321876,1.7335725,-0.36469156,1.545116,-3.0512161],
#     [-0.9348094,-1.3538164,1.7637508,-0.35094458,1.5647936,-3.044409],
#     [-0.9356959,-1.3870511,1.7932053,-0.3369725,1.5822717,-3.0386736],
#     [-0.9355997,-1.4210641,1.8207705,-0.32331473,1.5974523,-3.0341685],
#     [-0.9350012,-1.4553335,1.8459795,-0.31028926,1.6106507,-3.030782],
#     [-0.93381506,-1.4898244,1.8690894,-0.29779565,1.622475,-3.028275],
#     [-0.93187135,-1.5248826,1.890562,-0.28553012,1.6335789,-3.026439],
#     [-0.9291833,-1.5607065,1.9104347,-0.27340892,1.644382,-3.0251682],
#     [-0.9260805,-1.5969306,1.9282268,-0.261753,1.6549766,-3.0244002],
#     [-0.92300826,-1.6329483,1.9435723,-0.25093472,1.6653004,-3.024033],
#     [-0.9201402,-1.6686317,1.95667,-0.24097098,1.6753316,-3.0239403],
#     [-0.91727024,-1.7045897,1.9680647,-0.23154788,1.6851285,-3.0240395],
#     [-0.9135566,-1.7420108,1.9783126,-0.22223847,1.6946548,-3.0244143],
#     [-0.90710944,-1.7821953,1.9878252,-0.21270682,1.7034533,-3.025455],
#     [-0.89504504,-1.8258896,1.9967585,-0.20281012,1.7104461,-3.027831],
#     [-0.8745919,-1.8721489,2.0047443,-0.19282383,1.7140892,-3.032209],
#     [-0.8454565,-1.9181103,2.0109596,-0.18353748,1.7134672,-3.0386386],
#     [-0.8097479,-1.9617194,2.0149567,-0.1755515,1.7089447,-3.0465684],
#     [-0.76972204,-2.002093,2.0166018,-0.16923296,1.7011383,-3.0554323],
#     [-0.7270667,-2.038991,2.0158935,-0.16477513,1.6904845,-3.064873],
#     [-0.6831811,-2.072649,2.012993,-0.16210094,1.6775604,-3.074637],
#     [-0.63894814,-2.1031802,2.0081472,-0.1609097,1.6631529,-3.0844746],
#     [-0.59450823,-2.1304278,2.0017524,-0.16082162,1.6478592,-3.0942135],
#     [-0.54981345,-2.153996,1.9943026,-0.16153166,1.6320754,-3.1037936],
#     [-0.5049442,-2.1731865,1.9861981,-0.16294439,1.6161101,-3.1133335],
#     [-0.45993346,-2.187586,1.97771,-0.16507503,1.6000978,-3.123073],
#     [-0.41436625,-2.197376,1.9689869,-0.16791806,1.583788,-3.1333396],
#     [-0.3680489,-2.2027192,1.9603311,-0.17146447,1.5669018,-3.1444561],
#     [-0.32178697,-2.2035472,1.9522526,-0.17571093,1.5495846,-3.1564913],
#     [-0.27626142,-2.2004406,1.9448481,-0.18058924,1.5320323,-3.169267],
#     [-0.22850078,-2.1918824,1.937417,-0.18670557,1.512621,-3.1836708],
#     [-0.18488124,-2.1800635,1.9300072,-0.1926385,1.4936112,-3.196176],
#     [-0.14128424,-2.1658387,1.9221325,-0.19856922,1.4738089,-3.2078674],
#     [-0.09769948,-2.1494598,1.913874,-0.20436755,1.453394,-3.2186015],
#     [-0.054503746,-2.131319,1.9053813,-0.20997526,1.4327939,-3.2282836],
#     [-0.01160911,-2.1111374,1.8967794,-0.21540135,1.4122725,-3.2369025],
#     [0.03135389,-2.0888212,1.8882463,-0.22066511,1.3920192,-3.244498],
#     [0.07448143,-2.0649605,1.8798809,-0.22583438,1.3722584,-3.2511852],
#     [0.11788576,-2.0402048,1.8715929,-0.2310369,1.3531361,-3.2571614],
#     [0.1618497,-2.0146673,1.8632063,-0.23640469,1.3347343,-3.2626123],
#     [0.20630601,-1.9880562,1.854627,-0.24202372,1.3172002,-3.2676656],
#     [0.25035116,-1.9597951,1.8459486,-0.24788097,1.3007796,-3.2723248],
#     [0.29286745,-1.9293548,1.8373424,-0.25385132,1.2856671,-3.276459],
#     [0.3331672,-1.8964512,1.8288575,-0.2598579,1.2718991,-3.279896],
#     [0.3704054,-1.8606484,1.8203497,-0.26597553,1.2595487,-3.282426],
#     [0.4032681,-1.8216527,1.811664,-0.27235147,1.2490118,-3.2837982],
#     [0.4306751,-1.7802185,1.8027648,-0.2790492,1.2408781,-3.2838938],
#     [0.45292863,-1.7378601,1.7934612,-0.28599063,1.2353714,-3.282813],
#     [0.47128332,-1.6959231,1.7835487,-0.29295197,1.232251,-3.2807024],
#     [0.48735595,-1.6548654,1.7729071,-0.2997963,1.2310609,-3.2776766],
#     [0.50255513,-1.6142389,1.7613487,-0.3066322,1.231475,-3.2737503],
#     [0.5175731,-1.5733864,1.7485492,-0.3136766,1.2334479,-3.2688189],
#     [0.5324141,-1.5318118,1.7340474,-0.3211962,1.2371304,-3.2627442],
#     [0.5466485,-1.4892514,1.7173392,-0.32949132,1.2427293,-3.2554417],
#     [0.55983216,-1.445651,1.6980407,-0.33880463,1.2502755,-3.2469423],
#     [0.5717,-1.4009243,1.6759487,-0.3492535,1.2594842,-3.237356],
#     [0.5822087,-1.3547593,1.6508671,-0.3609084,1.2698816,-3.2267187],
#     [0.59181,-1.3068556,1.622424,-0.3738968,1.2810224,-3.2148917],
#     [0.6014659,-1.2574831,1.5904433,-0.3882845,1.2925595,-3.2017663],
#     [0.6123337,-1.2076333,1.55583,-0.40379256,1.3040473,-3.187765],
#     [0.6203806,-1.1571836,1.5208056,-0.41964418,1.3152571,-3.1736703],
#     [0.6049024,-1.1100295,1.4936249,-0.43223056,1.3255501,-3.1616647],
#     [0.57720846,-1.0792915,1.4801936,-0.4389368,1.332181,-3.154507],
#     [0.55070627,-1.0600363,1.4738588,-0.44260952,1.3362266,-3.1502132],
#     [0.5272907,-1.0470234,1.4707477,-0.44490054,1.3388424,-3.1473408],
#     [0.5066441,-1.037767,1.4693055,-0.44645935,1.3405817,-3.1452663],
#     [0.48826706,-1.031022,1.4688219,-0.44757983,1.3417203,-3.1436956],
#     [0.47172233,-1.0260593,1.4689299,-0.44842097,1.3424176,-3.142468],
#     [0.4566701,-1.0224016,1.4694293,-0.44907653,1.3427804,-3.1414886],
#     [0.44289634,-1.0197,1.4702001,-0.4496,1.3429,-2.8265407],
#     [0.44289634,-1.0197,1.4702001,-0.4496,1.3429,-2.8265407],
#     [0.44289634,-1.0197,1.4702001,-0.4496,1.3429,-2.5123813],
#     [0.44289634,-1.0197,1.4702001,-0.4496,1.3429,-2.1982222],
#     [0.44289634,-1.0197,1.4702001,-0.4496,1.3429,-1.8840628],
#     [0.44289634,-1.0197,1.4702001,-0.4496,1.3429,-1.5699035]
# ]

# Step!
for _ in range(10000):
    # if _ == 200:
    #     breakpoint()
    # env.step([])

    if _ == 5:
        execute_motion(env, path, False)


    # if joint_pos is not None:
    #     robot.set_joint_positions(th.tensor(joint_pos), indices=control_idx, drive=True)
    # og.sim.step()

og.shutdown()