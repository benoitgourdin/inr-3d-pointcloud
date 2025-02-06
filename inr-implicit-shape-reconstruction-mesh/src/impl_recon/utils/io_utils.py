import sys
import os
from io import TextIOWrapper
from pathlib import Path
from typing import List, OrderedDict, Tuple

import nibabel as nib
import numpy as np
import torch
import trimesh
import mesh_to_sdf
import pickle as pkl

import platform
if platform.system() == "Linux":
    os.environ['PYOPENGL_PLATFORM'] = 'egl'


def save_nifti_file(image_data: np.ndarray, local_to_global: np.ndarray, target_filepath: Path):
    """Save the given numpy array into a given nifti file path."""
    local_to_global_ras = local_to_global.copy()
    # Convert this library's LPS+ coordinate system to nibabel's RAS+ coordinates
    local_to_global_ras[0] *= -1
    local_to_global_ras[1] *= -1
    header = nib.Nifti1Header()
    header.set_data_dtype(image_data.dtype)
    header.set_qform(local_to_global_ras, code='scanner')
    img = nib.Nifti1Image(image_data, None, header)
    nib.save(img, target_filepath)


def save_pkl_file(image_path: Path):
    if not image_path.exists():
        raise ValueError('STL file does not exist:\n{}'.format(image_path))
    # mesh_data = mesh.Mesh.from_file(str(image_path))
    points_amount = 1000000
    mesh = trimesh.load_mesh(str(image_path))
    vertices = mesh.vertices
    bbox_center = (mesh.bounds[0] + mesh.bounds[1]) / 2.0
    bbox_extents = mesh.bounds[1] - mesh.bounds[0]
    max_extent = max(bbox_extents)
    scaling_factor = 2.0 / max_extent
    normalized_vertices = (vertices - bbox_center) * scaling_factor
    mesh.vertices = normalized_vertices
    rand_coords = np.random.rand(points_amount, 3) * 2.0 - 1.0
    # rand_coords = np.random.randn(points_amount, 3)
    # rand_coords = rand_coords / np.max(np.abs(rand_coords)) * 0.5
    distances = mesh_to_sdf.mesh_to_sdf(mesh, rand_coords, surface_point_method='scan', sign_method='normal',
                                        bounding_radius=None, scan_count=100, scan_resolution=400,
                                        sample_point_count=10000000, normal_sample_count=11)
    save_array = np.column_stack((rand_coords, distances))
    # print(save_array)
    # print(save_array.shape)
    pickle_path = os.path.join("/home/mil/gourdin/inr_3d_data/data/medshapenet_vertebra_pkl_normal", str(image_path).split("/")[-1].split(".")[0])
    with open(pickle_path + '.pkl', 'wb') as f:
        pkl.dump(save_array, f)


def load_pkl_file(image_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Return the image numpy array as well as the local to global affine transformation matrix.
    In the local image coordinate system, the array indices correspond to positions of voxels'
    centers. The transformation matrix therefore translates array indices into cartesian coordinates
    of voxels' centers in the LPS+ coordinate system.
    """
    if not image_path.exists():
        raise ValueError('PKL file does not exist:\n{}'.format(image_path))
    #mesh_data = mesh.Mesh.from_file(str(image_path))
    with open(image_path, 'rb') as f:
        x = pkl.load(f)
    rand_coords = x[:, [0, 1, 2]]
    distances = x[:, [3]]
    negative_mask = distances < 0
    labels = negative_mask.astype(int)
    one_indices = np.where(labels == 1)[0].tolist()
    zero_indices = np.where(labels == 0)[0].tolist()
    amount_one = len(one_indices)
    zero_indices = zero_indices[:amount_one]
    new_coords = np.concatenate((np.array(rand_coords[zero_indices, :]), np.array(rand_coords[one_indices, :])), axis=0)
    new_labels = np.concatenate((np.array(labels[zero_indices, :]), np.array(labels[one_indices, :])), axis=0)
    return new_coords, new_labels


def load_stl_file(image_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Return the image numpy array as well as the local to global affine transformation matrix.
    In the local image coordinate system, the array indices correspond to positions of voxels'
    centers. The transformation matrix therefore translates array indices into cartesian coordinates
    of voxels' centers in the LPS+ coordinate system.
    """
    if not image_path.exists():
        raise ValueError('STL file does not exist:\n{}'.format(image_path))
    #mesh_data = mesh.Mesh.from_file(str(image_path))
    points_amount = 10000
    mesh = trimesh.load_mesh(str(image_path))
    vertices = mesh.vertices
    bbox_center = (mesh.bounds[0] + mesh.bounds[1]) / 2.0
    bbox_extents = mesh.bounds[1] - mesh.bounds[0]
    max_extent = max(bbox_extents)
    scaling_factor = 2.0 / max_extent
    normalized_vertices = (vertices - bbox_center) * scaling_factor
    mesh.vertices = normalized_vertices
    rand_coords = np.random.rand(points_amount, 3) * 2.0 - 1.0
    distances = mesh_to_sdf.mesh_to_sdf(mesh, rand_coords, surface_point_method='scan', sign_method='normal',
                                        bounding_radius=None, scan_count=100, scan_resolution=400,
                                        sample_point_count=10000000, normal_sample_count=11)
    negative_mask = distances < 0
    labels = negative_mask.astype(int)
    return rand_coords, labels


def load_casenames(filepath: Path) -> List[str]:
    """Load a text file with case names. Empty lines are ignored."""
    if not filepath.exists():
        raise ValueError('Case name file does not exist:\n{}'
                         .format(filepath))
    with open(str(filepath), 'r') as f:
        lines = [line.rstrip() for line in f if line != '\n']

    return lines


class Logger(TextIOWrapper):
    """Log both to a given file as well as stdout. NOTE: delete this object explicitly before
    creating a new one if you want to change the target logging file.
    """
    def __init__(self, filepath: Path, filemode: str):
        super().__init__(sys.__stdout__.buffer)
        self.file = open(filepath, filemode)

    def __del__(self):
        # Do not reset the default stdout so that re-assigning to a new Logger is not overwritten
        self.file.close()

    def write(self, data):
        self.file.write(data)
        sys.__stdout__.write(data)

    def flush(self):
        self.file.flush()
        sys.__stdout__.flush()


def find_single_file(source_dir: Path, pattern: str) -> Path:
    """Within a given directory, find a single file that matches a given pattern. Throw exceptions
    if the directory does not exist or contains more than one file that matches the pattern.
    """
    if not source_dir.exists():
        raise ValueError('The source directory does not exist:\n{}'.format(source_dir))

    matched_files = list(source_dir.glob(pattern))
    if len(matched_files) != 1:
        raise ValueError('The source directory has to contain exactly one file that matches the '
                         'pattern {}:\n{}'.format(pattern, source_dir))
    return matched_files[0]


class RollingCheckpointWriter:
    """Writer to write checkpoint files in a rolling fashion (automatically deleting old ones)."""
    def __init__(self, base_dir: Path, base_filename: str, max_num_checkpoints: int,
                 extension: str = 'pth'):
        if not base_dir.exists():
            raise ValueError('Target directory does not exist:\n{}'.format(base_dir))
        if max_num_checkpoints <= 0:
            raise ValueError('Max number of checkpoints must be at least one.')

        self.base_dir = base_dir
        self.base_filename = base_filename
        self.max_num_checkpoints = max_num_checkpoints
        self.extension = extension

    def write_rolling_checkpoint(self, model_state: dict, optimizer_state: dict,
                                 num_steps_trained: int, num_epochs_trained: int):
        """Write the given state into a checkpoint file. Overwrite any existing files. Delete older
        checkpoints in the directory, so that the given upper bound is not exceeded.
        """
        # First, write the checkpoint file
        state = {
            'model_state': model_state,
            'optimizer_state': optimizer_state,
            'num_steps_trained': num_steps_trained,
            'num_epochs_trained': num_epochs_trained,
        }
        target_filepath = self.base_dir / '{}_{}.{}'.format(self.base_filename, num_steps_trained,
                                                            self.extension)
        torch.save(state, target_filepath)

        # Now delete oldest files until we reach the max
        paths = list(self.base_dir.glob('{}_*.{}'.format(self.base_filename, self.extension)))
        num_files_to_delete = len(paths) - self.max_num_checkpoints
        if num_files_to_delete <= 0:
            return
        paths.sort(key=lambda pth: int(pth.stem.split('_')[-1]))
        for path in paths[:num_files_to_delete]:
            path.unlink()


def load_latest_checkpoint(base_dir: Path, base_filename: str,
                           extension: str = 'pth',
                           verbose: bool = False) -> Tuple[OrderedDict, dict, int, int]:
    """
    Load the latest checkpoint from the given directory. Throws an error if the given directory does
    not exist or has no checkpoints.

    :param base_dir: Base model directory.
    :param base_filename: The base name within checkpoint files.
    :param extension: Checkpoint extension.
    :param verbose: Whether to print loaded state file.
    :return: The model state, optimizer state, number of steps trained, as well as number of epochs
             trained.
    """
    if not base_dir.exists():
        raise ValueError('Model directory for checkpoint loading does not exist:\n{}'
                         .format(base_dir))

    paths = list(base_dir.glob('{}_*.{}'.format(base_filename, extension)))
    if not paths:
        raise ValueError('Model directory for checkpoint loading does not contain any '
                         'checkpoints:\n{}'.format(base_dir))

    paths.sort(key=lambda pth: int(pth.stem.split('_')[-1]))
    latest_path = paths[-1]
    if verbose:
        print('Loading state file:', latest_path)
    state = torch.load(latest_path)
    return state['model_state'], state['optimizer_state'], state['num_steps_trained'], \
        state['num_epochs_trained']
