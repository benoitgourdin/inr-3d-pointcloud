from io import TextIOWrapper
import torch
import sys
import open3d as o3d


class Logger(TextIOWrapper):
    """Log both to a given file as well as stdout. NOTE: delete this object explicitly before
    creating a new one if you want to change the target logging file.
    """
    def __init__(self, filepath, filemode):
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


class RollingCheckpointWriter:
    """Writer to write checkpoint files in a rolling fashion (automatically deleting old ones)."""
    def __init__(self, base_dir, base_filename, max_num_checkpoints, extension = 'pth'):
        if not base_dir.exists():
            raise ValueError('Target directory does not exist:\n{}'.format(base_dir))
        if max_num_checkpoints <= 0:
            raise ValueError('Max number of checkpoints must be at least one.')
        self.base_dir = base_dir
        self.base_filename = base_filename
        self.max_num_checkpoints = max_num_checkpoints
        self.extension = extension

    def write_rolling_checkpoint(self, model_state, optimizer_state,
                                 num_steps_trained, num_epochs_trained):
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


class PointCloudWriter:
    def write_ply_file(self, point_cloud, path):
        points = point_cloud
        if isinstance(point_cloud, torch.Tensor):
            points = point_cloud.squeeze().cpu().detach().numpy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        o3d.io.write_point_cloud(path, pcd)
