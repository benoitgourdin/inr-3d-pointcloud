import numpy as np
from torch.utils import data


def anime_read(filename):
    """
    filename: path of .anime file
    return:
        nf: number of frames in the animation
        nv: number of vertices in the mesh (mesh topology fixed through frames)
        nt: number of triangle face in the mesh
        vert_data: vertice data of the 1st frame (3D positions in x-y-z-order)
        face_data: riangle face data of the 1st frame
        offset_data: 3D offset data from the 2nd to the last frame
    """
    f = open(filename, 'rb')
    nf = np.fromfile(f, dtype=np.int32, count=1)[0]
    nv = np.fromfile(f, dtype=np.int32, count=1)[0]
    nt = np.fromfile(f, dtype=np.int32, count=1)[0]
    vert_data = np.fromfile(f, dtype=np.float32, count=nv * 3)
    face_data = np.fromfile(f, dtype=np.int32, count=nt * 3)
    offset_data = np.fromfile(f, dtype=np.float32, count=-1)
    '''check data consistency'''
    if len(offset_data) != (nf-1) * nv * 3:
        raise ("data inconsistent error!", filename)
    vert_data = vert_data.reshape((-1, 3))
    face_data = face_data.reshape((-1, 3))
    offset_data = offset_data.reshape((nf - 1, nv, 3))
    return nf, nv, nt, vert_data, face_data, offset_data


class DeformingThings4D(data.Dataset):

    def __init__(self, anime_file, points_per):
        _, _, _, vert_data, face_data, offset_data = \
            anime_read(anime_file)
        offset_data = np.concatenate([
            np.zeros(
                (1, offset_data.shape[1], offset_data.shape[2])), offset_data
        ],
                                     axis=0)
        self.registration_data = offset_data
        self.point_cloud_data = vert_data
        self.points_per = points_per

    def __len__(self):
        #******* to change ********
        return 1

    def __getitem__(self, item):
        start_idx = 5
        end_idx = 7
        registration_flow = self.registration_data[
            end_idx] - self.registration_data[start_idx]

        # registration_flow_2 = np.tile(
        #     np.array([1, 0, 0]),
        #     (self.registration_data[start_idx].shape[0], 1))
        # registration_flow = registration_flow + registration_flow_2

        moving_pc = self.point_cloud_data + self.registration_data[start_idx]
        indices = np.arange(moving_pc.shape[0])  # Create an array of indices
        np.random.shuffle(indices)
        moving_pc = moving_pc[indices]
        registration_flow = registration_flow[indices]
        result_dict = {
            'moving_pc':
                moving_pc[:int(self.points_per * moving_pc.shape[0])].astype(
                    np.float32),
            'registration_flow':
                registration_flow[:int(self.points_per
                                       * moving_pc.shape[0])].astype(np.float32
                                                                    )
        }
        return result_dict
