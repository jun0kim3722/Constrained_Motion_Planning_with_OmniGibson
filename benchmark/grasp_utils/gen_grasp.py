import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from robot_arm_configuration import robot_arm_configuration
from scipy.spatial.distance import euclidean
import json
import os
    
def omni2xyz(rot, as_quat=True):
    # new_rot = R.from_euler("xyz", [rot[2], np.pi/2 - rot[0], rot[1]])
    euler = R.from_euler("xyz", rot)
    # new_rot = euler * R.from_quat([-0.5, 0.5, -0.5, -0.5])
    new_rot = euler * R.from_quat([0.5, -0.5, -0.5, -0.5])
    

    if as_quat:
        return new_rot.as_quat()
    else:
        return new_rot.as_euler("xyz")

def quaternion_angular_distance(q1, q2):
    dot_product = np.dot(q1, q2)
    return 2 * np.arccos(np.abs(dot_product))

def depth2pc(depth, K, rgb=None):
    """
    Convert depth and intrinsics to point cloud and optionally point cloud color
    :param depth: hxw depth map in m
    :param K: 3x3 Camera Matrix with intrinsics
    :returns: (Nx3 point cloud, point cloud color)
    """
    mask = np.where(depth > 0)
    x,y = mask[1], mask[0]
    
    normalized_x = (x.astype(np.float32) - K[0,2])
    normalized_y = (y.astype(np.float32) - K[1,2])

    world_x = normalized_x * depth[y, x] / K[0,0]
    world_y = normalized_y * depth[y, x] / K[1,1]
    world_z = depth[y, x]

    if rgb is not None:
        rgb = rgb[y,x,:]
        
    pc = np.vstack((world_x, world_y, world_z)).T
    return (pc, rgb)

def extract_point_clouds(depth, K, segmap=None, rgb=None, z_range=[0.2,4.0], segmap_id=0, skip_border_objects=False, margin_px=5):
        """
        Converts depth map + intrinsics to point cloud. 
        If segmap is given, also returns segmented point clouds. If rgb is given, also returns pc_colors.

        Arguments:
            depth {np.ndarray} -- HxW depth map in m
            K {np.ndarray} -- 3x3 camera Matrix

        Keyword Arguments:
            segmap {np.ndarray} -- HxW integer array that describes segeents (default: {None})
            rgb {np.ndarray} -- HxW rgb image (default: {None})
            z_range {list} -- Clip point cloud at minimum/maximum z distance (default: {[0.2,1.8]})
            segmap_id {int} -- Only return point cloud segment for the defined id (default: {0})
            skip_border_objects {bool} -- Skip segments that are at the border of the depth map to avoid artificial edges (default: {False})
            margin_px {int} -- Pixel margin of skip_border_objects (default: {5})

        Returns:
            [np.ndarray, dict[int:np.ndarray], np.ndarray] -- Full point cloud, point cloud segments, point cloud colors
        """

        if K is None:
            raise ValueError('K is required either as argument --K or from the input numpy file')
            
        # Convert to pc 
        pc_full, pc_colors = depth2pc(depth, K, rgb)

        # Threshold distance
        if pc_colors is not None:
            pc_colors = pc_colors[(pc_full[:,2] < z_range[1]) & (pc_full[:,2] > z_range[0])] 
        pc_full = pc_full[(pc_full[:,2] < z_range[1]) & (pc_full[:,2] > z_range[0])]
        
        # Extract instance point clouds from segmap and depth map
        pc_segments = {}
        if segmap is not None:
            pc_segments = {}
            obj_instances = [segmap_id] if segmap_id else np.unique(segmap[segmap>0])
            for i in obj_instances:
                if skip_border_objects and not i==segmap_id:
                    obj_i_y, obj_i_x = np.where(segmap==i)
                    if np.any(obj_i_x < margin_px) or np.any(obj_i_x > segmap.shape[1]-margin_px) or np.any(obj_i_y < margin_px) or np.any(obj_i_y > segmap.shape[0]-margin_px):
                        print('object {} not entirely in image bounds, skipping'.format(i))
                        continue
                inst_mask = segmap==i
                pc_segment,_ = depth2pc(depth*inst_mask, K)
                pc_segments[i] = pc_segment[(pc_segment[:,2] < z_range[1]) & (pc_segment[:,2] > z_range[0])] #regularize_pc_point_count(pc_segment, grasp_estimator._contact_grasp_cfg['DATA']['num_point'])

        new_pc = []
        for p in pc_segments[1]:
            pos = [p[2], -p[0], -p[1]]
            rot = R.from_quat(CAM_ROT)
            new_pos = rot.apply(pos)
            new_pc.append(new_pos + np.array(CAM_TRAN))

        return pc_full, pc_segments, pc_colors, new_pc

# def calc_grasp_pos(grasp_mat, cam_rot, cam_tran, offset, idx):
#         # calc grasp_rot
#         grasp_rot = grasp_mat[:3, :3]
#         grasp_rot = R.from_quat(R.from_matrix(grasp_mat[:3,:3]).as_quat()) #convert 3x3 into rot
#         axis_trans = np.array([
#             [0, 0, 1],  # z' -> x
#             [-1, 0, 0], # -x' -> y
#             [0, -1, 0]  # -y' -> z
#         ])
#         rota = np.array(grasp_rot.as_matrix())
#         grasp_rot = R.from_quat(R.from_matrix(np.dot(np.dot(axis_trans, rota), axis_trans.T)).as_quat())

#         # get position of camera
#         cam_rot_quat = R.from_quat(cam_rot)
 
#         # calc grasp_tran
#         grasp_tran = grasp_mat[:3, 3]
#         mat_rot = R.from_quat([-0.5, 0.5, -0.5, 0.5])
#         grasp_tran = mat_rot.apply(grasp_tran)

#         # grasp in global coord
#         target_quat = (cam_rot_quat * grasp_rot).as_quat()
#         target_pos = cam_tran + cam_rot_quat.apply(grasp_tran) + offset

#         return target_pos, target_quat

def calc_grasp_pos(grasp_mat, cam_rot, cam_tran, rot_idx, offset=0.02):
        # calc grasp_rot
        grasp_rot = grasp_mat[:3, :3]
        grasp_rot = R.from_quat(R.from_matrix(grasp_mat[:3,:3]).as_quat()) #convert 3x3 into rot
        axis_trans = np.array([
            [0, 0, 1],  # z' -> x
            [-1, 0, 0], # -x' -> y
            [0, -1, 0]  # -y' -> z
        ])
        rota = np.array(grasp_rot.as_matrix())
        grasp_rot = R.from_quat(R.from_matrix(np.dot(np.dot(axis_trans, rota), axis_trans.T)).as_quat())

        # get position of camera
        extra_rot = R.from_euler('z', -90 * rot_idx, degrees = True)
        cam_rot_quat = R.from_quat(cam_rot)

        # calc grasp_tran
        grasp_tran = grasp_mat[:3, 3]
        mat_rot = R.from_quat([-0.5, 0.5, -0.5, 0.5])
        grasp_tran = mat_rot.apply(grasp_tran) + grasp_rot.apply([-offset, 0.0, 0.0])

        # grasp in global coord
        target_quat = (extra_rot * (cam_rot_quat * grasp_rot)).as_quat()
        target_pos = extra_rot.apply(cam_tran) + extra_rot.apply(cam_rot_quat.apply(grasp_tran))

        return target_pos, target_quat

def read_grasp(file_name, index, new_pc):
    try:
        data = np.load(file_name, allow_pickle=True)
        rac = robot_arm_configuration("../assets/urdf/ur5e/meshes/collision/", np.array([-0.4, 0.0, 0.0]))

    except:
        print("GRASP: {} does not exist!!".format(file_name))
        return None
    
    grasp_data = data['pred_grasps_cam'].tolist()[1]
    scores = data['scores'].tolist()[1]
    sorted_idx = np.argsort(scores)

    grasps = []
    for i in sorted_idx[::-1]:
        grasp_mat = grasp_data[i]

        grasp_pos, grasp_quat = calc_grasp_pos(grasp_mat, CAM_ROT, CAM_TRAN, index)

        grasp_euler = R.from_quat(grasp_quat).as_euler('xyz')

        # joints = rac.inverse_kinematics(grasp_pos, grasp_euler, viz=False)
        # if joints is not None:
        #     print(grasp_euler, i)
        #     st = rac.check_collision_models(joints, pcd=new_pc)

        #     if 'y' in input("Y or N"):
        #         print("grasp added!!")
        grasps.append({"pos":grasp_pos.tolist(), "quat":grasp_quat.tolist()})

    return grasps

def filter_grasps(poses, pos_threshold=0.4, rot_threshold=0.5):
    filtered_poses = []
    if not poses:
        return filtered_poses

    filtered_poses.append(poses[0])

    for current_pose in poses[1:]:
        is_similar_to_any_filtered = False
        for filtered_pose in filtered_poses:
            pos_dist = euclidean(current_pose["pos"], filtered_pose["pos"])
            rot_dist = quaternion_angular_distance(current_pose["quat"], filtered_pose["quat"])

            if pos_dist < pos_threshold and rot_dist < rot_threshold:
                is_similar_to_any_filtered = True
                break
        if not is_similar_to_any_filtered:
            filtered_poses.append(current_pose)
    return filtered_poses

def get_grasp(obj_dir, robot_pos=[-0.4, 0.0, 0.0], viz=False):
# def get_grasp(obj_dir, robot_pos=[0.0, 0.3, 0.0], viz=True):

    total_grasps = []
    new_pc = None
    for i in range(4):
        cg_file_name = obj_dir + "/grasp_cg" + str(i) + ".npz"
        view_file_name = obj_dir + "/cg" + str(i) + ".npy"

        try:
            data = np.load(view_file_name, allow_pickle=True).tolist()
        except:
            print("CAPTURE: {} does not exist!!".format(view_file_name))
            continue

        if i == 0:
            # plt.imshow(data["rgb"])
            # plt.show()
            pc_full, pc_segments, pc_colors, new_pc = extract_point_clouds(data["depth"], data["K"], segmap=data["seg"], rgb=data["rgb"], skip_border_objects=False)

        grasps = read_grasp(cg_file_name, i, new_pc)

        total_grasps += grasps

    total_grasps = filter_grasps(total_grasps)
    print("Total Grasp Created: ", len(total_grasps))

    if viz:
        rac = robot_arm_configuration("../assets/urdf/ur5e/meshes/collision/", np.array(robot_pos))

        i = 0
        for grasp in total_grasps:
            grasp_pos = grasp['pos']
            grasp_rot = grasp['quat']
            
            # grasp_euler = omni2xyz(R.from_quat(grasp_rot).as_euler('xyz'), as_quat=False)
            
            grasp_euler = R.from_quat(grasp_rot).as_euler('xyz')


            # grasp_pos = [-0.053624151866269676, -0.0022369837388396263, 0.1283487118201101]
            # grasp_euler = R.from_quat([-0.3581,  0.4630,  0.3187,  0.7455]).as_euler('xyz')

            joints = rac.inverse_kinematics(grasp_pos, grasp_euler, viz=False)

            if joints is not None:
                print(grasp_euler, i)
                # [ 0.15408538 -0.68971402 -3.10595083] 25 didnt work
                # [0.15408538 0.68971402 3.10595083] 25
                st = rac.check_collision_models(joints, pcd=new_pc)
            else:
                st = rac.check_collision_models(np.zeros(6), pcd=new_pc)
            
            i += 1

    with open(obj_dir + "/grasp_results.json", 'w') as f:
        json.dump(total_grasps, f, indent=4)
    print("File Saved: {}\n\n".format(obj_dir + "/grasp_results.json"))

if __name__ == "__main__":
    # read_grasp("/home/j0k/Project/OmniGibson/datasets/og_dataset/objects/acetone_atomizer/krtwsl/usd/grasp_cg0.npz")

    # CAM_TRAN = [-1.2, 0.0,  2.0]
    # CAM_ROT = R.from_euler("yxz", [np.pi/2 - np.deg2rad(30), 0, 0]).as_quat()
    # get_grasp("/home/j0k/Project/OmniGibson/datasets/og_dataset/objects/acetone_atomizer/krtwsl/usd")

    # [ 0.15408538 -0.68971402 -3.10595083] 25 didnt work
    # [0.15408538 0.68971402 3.10595083] 25

    # rot = [0.15408538,  0.68971402,  3.10595083]
    # quat = R.from_euler("xyz", rot).as_quat()
    # omni_rot = xyz2omni(rot, as_quat=False)
    # og = omni2xyz(omni_rot, as_quat=False)
    # og_quat = omni2xyz(omni_rot)

    # cal1 = [-1 * (rot[1] - np.pi/2), rot[2], rot[0]]
    # cal2 = [cal1[2], np.pi/2 - cal1[0], cal1[1]]
    # breakpoint()


    CAM_TRAN = [0.0, -1.5,  1.0]
    # R.from_euler("xyz", [np.deg2rad(60), np.deg2rad(0), np.deg2rad(0)]).as_quat()
    CAM_ROT = omni2xyz([np.deg2rad(60), np.deg2rad(0), np.deg2rad(0)])

    # CAM_ROT = omni2xyz([np.deg2rad(60), np.deg2rad(30), np.deg2rad(15)])


    # CAM_TRAN = [-1.2, 0.0,  2.0]
    # CAM_ROT = omni2xyz([np.pi/2 - np.deg2rad(30), 0, 0])
    # CAM_ROT = R.from_euler("yxz", [np.pi/2 - np.deg2rad(30), 0, 0]).as_quat()

    base_dir = "/home/j0k/Project/OmniGibson/datasets/og_dataset/objects/"

    obj_list = []
    with open("../../obj_list.txt", 'r') as file:
        for l in file:
            if "\n" in l:
                l = l[:-1]
            print(l)
            obj_list.append(l)
    
    for obj_name in obj_list:
        obj_dir = base_dir + obj_name

        for item in os.listdir(obj_dir):
            # if item != "ghhbgo":
            #     continue
            item_path = os.path.join(obj_dir, item) + "/usd"
            if os.path.isdir(item_path):
                print("--------------OPENING: " + item_path +  "--------------")
                get_grasp(item_path, viz=True)