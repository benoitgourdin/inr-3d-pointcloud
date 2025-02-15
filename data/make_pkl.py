import os.path
import os

os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
import mesh_to_sdf
import trimesh
import open3d as o3d
import pickle as pkl
import numpy as np


def save_pkl_file(image_path):
    # mesh_data = mesh.Mesh.from_file(str(image_path))
    points_amount = 1000000
    mesh = trimesh.load_mesh(str(image_path))
    vertices = mesh.vertices
    bbox_center = (mesh.bounds[0] + mesh.bounds[1]) / 2.0
    bbox_extents = mesh.bounds[1] - mesh.bounds[0]
    max_extent = max(bbox_extents)
    scaling_factor = 2.0 / max_extent
    normalized_vertices = (vertices-bbox_center) * scaling_factor
    mesh.vertices = normalized_vertices
    rand_coords = np.random.rand(points_amount, 3) * 2.0 - 1.0
    # rand_coords = np.random.randn(points_amount, 3)
    # rand_coords = rand_coords / np.max(np.abs(rand_coords)) * 0.5
    distances = mesh_to_sdf.mesh_to_sdf(
        mesh,
        rand_coords,
        surface_point_method='scan',
        sign_method='normal',
        bounding_radius=None,
        scan_count=100,
        scan_resolution=400,
        sample_point_count=10000000,
        normal_sample_count=11)
    save_array = np.column_stack((rand_coords, distances))
    # print(save_array)
    # print(save_array.shape)
    pickle_path = os.path.join(
        "/u/home/gob/repo/data/MedShapeNet/vertebra_pkl",
        str(image_path).split("/")[-1].split(".")[0])
    with open(pickle_path + '.pkl', 'wb') as f:
        pkl.dump(save_array, f)


def save_ply(image_path, name):
    # mesh_data = mesh.Mesh.from_file(str(image_path))
    points_amount = 1000000
    mesh = trimesh.load_mesh(str(image_path))
    vertices = mesh.vertices
    bbox_center = (mesh.bounds[0] + mesh.bounds[1]) / 2.0
    bbox_extents = mesh.bounds[1] - mesh.bounds[0]
    max_extent = max(bbox_extents)
    scaling_factor = 2.0 / max_extent
    normalized_vertices = (vertices-bbox_center) * scaling_factor
    mesh.vertices = normalized_vertices
    rand_coords = np.random.rand(points_amount, 3) * 2.0 - 1.0
    # rand_coords = np.random.randn(points_amount, 3)
    # rand_coords = rand_coords / np.max(np.abs(rand_coords)) * 0.5
    distances = mesh_to_sdf.mesh_to_sdf(
        mesh,
        rand_coords,
        surface_point_method='scan',
        sign_method='normal',
        bounding_radius=None,
        scan_count=100,
        scan_resolution=400,
        sample_point_count=10000000,
        normal_sample_count=11)
    save_array = np.column_stack((rand_coords, distances))
    # print(save_array)
    # print(save_array.shape)
    rand_coords = save_array[:, [0, 1, 2]]
    distances = save_array[:, [3]]
    negative_mask = distances < 0
    labels = negative_mask.astype(int)
    one_indices = np.where(labels == 1)[0].tolist()
    zero_indices = np.where(labels == 0)[0].tolist()
    amount_one = len(one_indices)
    zero_indices = zero_indices[:amount_one]
    print(amount_one)
    point_cloud = rand_coords[one_indices, :]
    pc_path = os.path.join("/vol/aimspace/users/gob/Dataset/MedShapeNet/",
                           f'30000p_{name}.ply')
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud[:30000, :])
    o3d.io.write_point_cloud(pc_path, pcd)
    pc_path = os.path.join("/vol/aimspace/users/gob/Dataset/MedShapeNet/",
                           f'5000p_{name}.ply')
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud[:5000, :])
    o3d.io.write_point_cloud(pc_path, pcd)


if __name__ == '__main__':
    save_ply(
        "/vol/aimspace/users/gob/Dataset/MedShapeNet/vertebra_stl/000000_vertebrae.stl",
        "000000")
    save_ply(
        "/vol/aimspace/users/gob/Dataset/MedShapeNet/vertebra_stl/000001_vertebrae.stl",
        "000001")
    save_ply(
        "/vol/aimspace/users/gob/Dataset/MedShapeNet/vertebra_stl/000002_vertebrae.stl",
        "000002")
    save_ply(
        "/vol/aimspace/users/gob/Dataset/MedShapeNet/vertebra_stl/000003_vertebrae.stl",
        "000003")
