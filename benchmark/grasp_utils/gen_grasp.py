import numpy as np
from pxr import Usd, UsdGeom
import open3d as o3d

def visualize_pointcloud_with_normals_open3d(points, normals):
    """
    Visualizes a point cloud with normals using Open3D.

    Args:
        points (np.ndarray): A NumPy array of shape (N, 3) representing the point coordinates.
        normals (np.ndarray): A NumPy array of shape (N, 3) representing the normal vectors for each point.
    """
    if not isinstance(points, np.ndarray) or points.shape[1] != 3:
        raise ValueError("Points must be a NumPy array of shape (N, 3).")
    if not isinstance(normals, np.ndarray) or normals.shape != points.shape:
        raise ValueError("Normals must be a NumPy array of the same shape as points (N, 3).")

    breakpoint()
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.normals = o3d.utility.Vector3dVector(normals)

    # Visualize the point cloud with normals
    o3d.visualization.draw_geometries([point_cloud], point_show_normal=True, lookat=point_cloud.get_center())

def visualize_pointcloud_open3d(points, colors=None):
    """
    Visualizes a point cloud using Open3D.

    Args:
        points (np.ndarray): A NumPy array of shape (N, 3) representing the point coordinates.
        colors (np.ndarray, optional): A NumPy array of shape (N, 3) representing RGB colors for each point. Defaults to None (all points will be white).
    """
    if not isinstance(points, np.ndarray) or points.shape[1] != 3:
        raise ValueError("Points must be a NumPy array of shape (N, 3).")

    breakpoint()
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)

    if colors is not None:
        if not isinstance(colors, np.ndarray) or colors.shape != points.shape:
            raise ValueError("Colors must be a NumPy array of the same shape as points (N, 3).")
        point_cloud.colors = o3d.utility.Vector3dVector(colors)
    else:
        # Default color to white if no colors are provided
        default_colors = np.zeros_like(points)
        point_cloud.colors = o3d.utility.Vector3dVector(default_colors)

    o3d.visualization.draw_geometries([point_cloud])

def extract_points_from_usd(file_path):
    """Extracts 3D points from a USD or USDA file."""
    try:
        stage = Usd.Stage.Open(file_path)
        points = []
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Mesh):
                usd_points = UsdGeom.Mesh(prim).GetPointsAttr().Get()
                if usd_points:
                    points.extend(usd_points)
    
        visualize_pointcloud_open3d(np.array(points))
        return np.array(points)
    except Exception as e:
        print(f"Error extracting points: {e}")
        return None

def extract_point_clouds_from_usd(usd_file_path):
    """
    Extracts point cloud data (points, colors, normals) from all
    UsdGeom.Points prims in a USD file.

    Args:
        usd_file_path (str): The path to the .usd file.

    Returns:
        list: A list of dictionaries, where each dictionary represents a point cloud
              and contains keys like 'name', 'points' (NumPy array),
              'colors' (optional NumPy array), and 'normals' (optional NumPy array).
              Returns an empty list if no point clouds are found or the file cannot be opened.
    """
    stage = Usd.Stage.Open(usd_file_path)
    if not stage:
        print(f"Error: Could not open USD file at {usd_file_path}")
        return []

    point_clouds_data = []
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Points):
            points_geom = UsdGeom.Points(prim)

            points_attr = points_geom.GetPointsAttr()
            point_data = points_attr.Get()

            colors_attr = points_geom.GetDisplayColorAttr()
            color_data = colors_attr.Get() if colors_attr.IsValid() else None

            normals_attr = points_geom.GetNormalsAttr()
            normal_data = normals_attr.Get() if normals_attr.IsValid() else None

            if point_data is not None:
                points_np = np.array(point_data)
                colors_np = np.array(color_data) if color_data is not None else None
                normals_np = np.array(normal_data) if normal_data is not None and len(normal_data) == len(point_data) else None

                # Get the world transformation matrix
                world_transform = Gf.Matrix4d(points_geom.ComputeWorldTransform(Usd.TimeCode.Default()))
                transformed_points = np.array([world_transform.Transform(Gf.Vec3d(p)) for p in points_np])

                transformed_normals = None
                if normals_np is not None:
                    world_rotation = world_transform.ExtractRotation()
                    transformed_normals = np.array([world_rotation.Transform(Gf.Vec3f(n)).GetNormalized() for n in normals_np])

                point_clouds_data.append({
                    'name': prim.GetName(),
                    'points': transformed_points,
                    'colors': colors_np,
                    'normals': transformed_normals
                })

    visualize_pointcloud_open3d(point_clouds_data)

    return point_clouds_data


def grasp_generation():
    test_name = 'sugar_box_grasp'

    grasp_file_path = '../contact_graspnet/results/' + test_name + '.npz'
    grasp_datas = np.load(grasp_file_path, allow_pickle=True)
    grasp_score_idx = list(np.argsort(-grasp_datas["scores"].item()[1]))

    cam_file_path = 'test_data/test_scenes/7.29.14.14/test_npy/0.npy'
    # cam_file_path = 'test_data/test_scenes/7.29.14.7/test_npy/1.npy'
    # cam_file_path = 'test_data/test_scenes/7.29.13.31/test_npy/0.npy'

    cam_datas = np.load(cam_file_path, allow_pickle=True)
    cam_rot = cam_datas.item()["cam_rot"]
    cam_tran = cam_datas.item()["cam_tran"]

    scene_info = [0.56, 0.86000001, 0.1, 0.5]
    rac = robot_arm_configuration('../assets/urdf/ur5e/meshes/collision/', np.array([0.0, 0, 0]), scene_info) # point_cloud=point_cloud

    generated_grasp = []

    for grasp_idx in range(len(grasp_score_idx)):
        grasp_mat = grasp_datas["pred_grasps_cam"].item()[1][grasp_idx]
        offset = [-0.5, 0, 0]
        target_pos, target_quat = rac.calc_grasp_pos(grasp_mat, cam_rot, cam_tran, offset)
        generated_grasp.append({"target_pos":target_pos, "target_quat":target_quat})

        # init2grasp_angels = rac.grasp_verify(grasp_mat, cam_rot, cam_tran, offset=[-0.5,0,0])

        # if init2grasp_angels is not None:
        #     rac.check_collision_models(init2grasp_angels)
    np.save("../assets/urdf/ycb/004_sugar_box/grasp_dict.npy", generated_grasp)

if __name__ == "__main__":
    extract_points_from_usd("ur5e_robotiq_2f85.usd")