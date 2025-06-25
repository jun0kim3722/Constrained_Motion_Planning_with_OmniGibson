import torch as th

import omnigibson as og
from omnigibson.utils.asset_utils import (
    get_all_object_categories,
    get_all_object_category_models,
    get_og_avg_category_specs,
)
from omnigibson.utils.ui_utils import choose_from_options

import trimesh
# import open3d as o3d
import numpy as np

from omnigibson.utils.constants import semantic_class_id_to_name
import matplotlib.pyplot as plt
import omnigibson.utils.transform_utils as T
from imageio import imwrite

# def pcd_viz(points):
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points)

#     # Visualize
#     o3d.visualization.draw_geometries([pcd], window_name="OmniGibson Object Point Cloud")

# def tri_mesh_viz(trimesh_mesh):
#     mesh_o3d = o3d.geometry.TriangleMesh()
#     mesh_o3d.vertices = o3d.utility.Vector3dVector(trimesh_mesh.vertices)
#     mesh_o3d.triangles = o3d.utility.Vector3iVector(trimesh_mesh.faces)
#     mesh_o3d.compute_vertex_normals()

#     o3d.visualization.draw_geometries([mesh_o3d], window_name="Trimesh Mesh via Open3D")

# def save_pcd(obj, obj_dir):
#     from pxr import UsdGeom, Gf
#     trimesh_mesh = None
#     vertices_all = []
#     faces_all = []
#     offset = 0

#     link = obj.links['base_link']
#     for visual_geom in link.visual_meshes.values():
#         usd_prim = visual_geom.prim
#         mesh_prims = []
#         if usd_prim.IsA(UsdGeom.Mesh):
#             mesh_prims.append(UsdGeom.Mesh(usd_prim))
#         else:
#             for child in usd_prim.GetChildren():
#                 if child.IsA(UsdGeom.Mesh):
#                     mesh_prims.append(UsdGeom.Mesh(child))

#         for mesh in mesh_prims:
#             points = mesh.GetPointsAttr().Get()
#             face_vertex_counts = mesh.GetFaceVertexCountsAttr().Get()
#             face_vertex_indices = mesh.GetFaceVertexIndicesAttr().Get()

#             verts = np.array([[v[0], v[1], v[2]] for v in points])
#             vertices_all.append(verts)

#             indices = np.array(face_vertex_indices, dtype=np.int32)
#             counts = np.array(face_vertex_counts, dtype=np.int32)

#             faces = []
#             i = 0
#             for c in counts:
#                 if c == 3:
#                     faces.append([indices[i], indices[i + 1], indices[i + 2]])
#                 elif c == 4:
#                     faces.append([indices[i], indices[i + 1], indices[i + 2]])
#                     faces.append([indices[i], indices[i + 2], indices[i + 3]])
#                 i += c

#             faces = np.array(faces) + offset
#             faces_all.append(faces)
#             offset += verts.shape[0]

#     if not vertices_all:
#         raise RuntimeError("No mesh data found in visual meshes")
    
#     # Sample point cloud
#     vertices_combined = np.concatenate(vertices_all, axis=0)
#     faces_combined = np.concatenate(faces_all, axis=0)
#     trimesh_mesh = trimesh.Trimesh(vertices=vertices_combined, faces=faces_combined)
#     points, _ = trimesh.sample.sample_surface(trimesh_mesh, count=10000)

#     np.save(obj_dir+"/pcd", points)

def save_scene_imgs(cam, center_offset, obj_dir, idx, viz=False):
    cg_dict = {'K' : cam.intrinsic_matrix.numpy(),
               'cam_pos' : cam.get_position_orientation()[0].numpy(),
               'cam_rot' : cam.get_position_orientation()[1].numpy(),
               "obj_offset" : center_offset.numpy()}
    obs, seg_id = cam.get_obs()
    for modality in ["rgb", "depth", "seg_instance"]:
        img = obs[modality].cpu()

        if modality == "seg_instance":
            try:
                obj_id = [k for k, v in seg_id[modality].items() if v == 'obj'][0]
                img[img!=obj_id] = 0
                img[img==obj_id] = 1
                modality = "seg"
            except:
                print("Obj not detected!!" + obj_dir)
                return

        cg_dict[modality] = img.numpy()

        if viz:
            plt.imshow(img)
            plt.show()

    np.save(obj_dir + '/cg' + str(idx), cg_dict)

def scene_for_grasp_gen(env, cam, obj_dir):
    # Place the object so it rests on the floor
    obj = env.scene.object_registry("name", "obj")
    center_offset = obj.get_position_orientation()[0] - obj.aabb_center + th.tensor([0, 0, obj.aabb_extent[2] / 2.0])
    obj.set_position_orientation(position=center_offset)
    
    for _ in range(100):
        env.step(th.empty(0))

    for modality in ["rgb", "depth", 'depth_linear', "seg_semantic", "seg_instance"]:
        cam.add_modality(modality)

    # Rotate the object in place
    num_rotate = 4
    for img_idx in range(num_rotate):
        z_angle = 2 * np.pi * img_idx / num_rotate
        quat = T.euler2quat(th.tensor([0, 0, z_angle]))
        pos = T.quat2mat(quat) @ center_offset
        obj.set_position_orientation(position=pos, orientation=quat)
        print(img_idx, obj.get_position_orientation())

        # wait for obj to sattle
        for _ in range(100):
            env.step(th.empty(0))

        save_scene_imgs(cam, center_offset, obj_dir, img_idx, viz=False)

def main(random_selection=False, headless=True, short_exec=False):
    """
    This demo shows how to load any scaled objects from the OG object model dataset
    The user selects an object model to load
    The objects can be loaded into an empty scene or an interactive scene (OG)
    The example also shows how to use the Environment API or directly the Simulator API, loading objects and robots
    and executing actions
    """
    og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80)

    # Create and load this object into the simulator
    # obj_cfg = dict(
    #     type="DatasetObject",
    #     name="obj",
    #     category=obj_category,
    #     model=obj_model,
    #     position=[0, 0, 0.01],
    # )

    cfg = {
        "scene": {
            "type": 'Scene',
        },
        # "objects": [obj_cfg],
    }

    # Create the environment
    env = og.Environment(configs=cfg)

    # set camera
    cam = og.sim.viewer_camera
    cam.set_position_orientation(position=th.tensor([0, -2.0,  2.0]), orientation=T.euler2quat(th.tensor([np.pi/4, 0, 0])))


    # # -- Choose the object to load --
    # # Select a category to load
    # available_obj_categories = get_all_object_categories()
    # obj_category = choose_from_options(
    #     options=available_obj_categories, name="object category", random_selection=random_selection
    # )

    # # Select a model to load
    # available_obj_models = get_all_object_category_models(obj_category)
    # obj_model = choose_from_options(
    #     options=available_obj_models, name="object model", random_selection=random_selection
    # )

    # -- Choose the object to load --
    asset_dir = "/data/og_dataset/objects"
    obj_list_file = asset_dir + "/obj_list.txt"
    available_obj_categories = get_all_object_categories()
    obj_list = []
    try:
        with open(obj_list_file, 'r') as file:
            for l in file:
                if "\n" in l:
                    l = l[:-1]
                print(l)
                
                if l in available_obj_categories:
                    obj_list.append(l)
                
    except FileNotFoundError:
        print(f"Error: The file '{obj_list_file}' was not found.")

    for obj_category in obj_list:

        available_obj_models = get_all_object_category_models(obj_category)
        for obj_model in available_obj_models:
            print("\n\n\n--------------------------" + obj_category + ' ' + obj_model + "--------------------------")
            # cam.set_position_orientation(position=th.tensor([0, -1.2,  2.0]), orientation=T.euler2quat(th.tensor([np.deg2rad(30), 0, 0])))
            cam.set_position_orientation(position=th.tensor([0, -1.5,  1.0]), orientation=T.euler2quat(th.tensor([np.deg2rad(60), 0, 0])))

            # add object into scene
            obj_dir = asset_dir + '/' + obj_category + '/' + obj_model + "/usd"
            obj = og.objects.DatasetObject(name='obj', category=obj_category, model=obj_model, position=[0, 0, 0.01])
            env.scene.add_object(obj)

            for _ in range(100):
            # while True:
                env.step(th.empty(0))

            scene_for_grasp_gen(env, cam, obj_dir)

            # remove object
            env.scene.remove_object(obj)
            for _ in range(50):
                env.step(th.empty(0))

    og.clear()


if __name__ == "__main__":
    main()