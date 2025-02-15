import vtk
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy
import pickle as pkl
import open3d as o3d


def mesh_to_voxel_manual(filepath):
    # import
    reader = vtk.vtkSTLReader()
    reader.SetFileName(filepath)
    reader.Update()
    # vtkPolyData
    mesh_polydata = reader.GetOutput()

    # Define voxel grid dimensions
    grid_dim = 128
    max_bound = max(mesh_polydata.GetBounds())
    voxel_size = max_bound / grid_dim

    # Create an empty NumPy array for voxel grid
    voxel_grid = np.zeros((grid_dim, grid_dim, grid_dim), dtype=bool)

    # Create an OBB tree
    obb_tree = vtk.vtkOBBTree()
    obb_tree.SetDataSet(mesh_polydata)
    obb_tree.BuildLocator()

    for i in range(grid_dim):
        print(str(i) + " from " + str(grid_dim) + " layers")
        for j in range(grid_dim):
            for k in range(grid_dim):
                # Calculate voxel position
                x = i * voxel_size
                y = j * voxel_size
                z = k * voxel_size
                # Ray-casting
                point = [x, y, z]
                ray_start = point
                ray_end = [max_bound + 1, point[1], point[2]]  # Extend ray beyond mesh bounds
                intersection_points = vtk.vtkPoints()
                obb_tree.IntersectWithLine(ray_start, ray_end, intersection_points, None)
                # Count intersections
                num_intersections = intersection_points.GetNumberOfPoints()
                if num_intersections % 2 == 1:
                    voxel_grid[i, j, k] = True
                else:
                    voxel_grid[i, j, k] = False
    return voxel_grid


def mesh_to_voxel(filepath):
    # Import
    reader = vtk.vtkSTLReader()
    reader.SetFileName(filepath)
    reader.Update()
    # vtkPolyData
    mesh_polydata = reader.GetOutput()
    # Transformer to stencil
    transformer = vtk.vtkPolyDataToImageStencil()
    transformer.SetInputData(mesh_polydata)
    transformer.Update()
    image = transformer.GetOutput()
    # Transformer to image
    image_creator = vtk.vtkImageStencil()
    image_creator.SetInputData(image)
    image_creator.Update()
    img = image_creator.GetOutput()

    rows, cols, _ = img.GetDimensions()
    sc = img.GetPointData().GetScalars()
    voxel_grid = vtk_to_numpy(sc)
    voxel_grid = voxel_grid.reshape(rows, cols, -1)
    return voxel_grid


def pkl_to_pointcloud(path):
    with open(path, 'rb') as f:
        x = pkl.load(f)
    coords = x[np.where(x[:, 3] < 0)]
    coords = coords[:, [0, 1, 2]]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    o3d.io.write_point_cloud("/.../000000_vertebrae.ply", pcd)


if __name__ == '__main__':
    # Import mesh
    filename = "/.../000000_vertebrae.pkl"
    pkl_to_pointcloud(filename)
    # voxel = mesh_to_voxel(filename)
    # print(voxel)
