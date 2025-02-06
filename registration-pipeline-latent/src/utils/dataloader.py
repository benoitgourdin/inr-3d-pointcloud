from torch.utils import data
from utils.lung250M import Lung250MDataset
from utils.DeformingThings4D import DeformingThings4D
from utils.defaults import get_cfg_defaults


def create_data_loader(pc_path, points_per, phase, pair_id=0):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(
        '/u/home/gob/repo/inr-masterthesis/registration-pipeline/src/config_lung.yml'
    )
    ds = Lung250MDataset(
        cfg, points_per, phase=phase, split=phase, pair=pair_id)
    # ds = DeformingThings4D(pc_path, points_per)
    dl = data.DataLoader(ds, 1, True)
    return dl
