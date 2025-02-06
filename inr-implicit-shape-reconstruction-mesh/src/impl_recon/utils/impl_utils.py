import math
from typing import List

import numpy as np
import torch

from impl_recon.models import implicits
from impl_recon.utils import geometry_utils
from surface_distance import metrics


def sample_latents(latents: torch.Tensor, occ_net: implicits.AutoDecoder, coordinates: torch.Tensor,
                   batch_size_coords: int = 64 ** 3) -> torch.Tensor:
    """Sample given batch of latent vectors at a given spatial resolution and spacing. The spatial
    resolution must be the same for all batch examples, therefore it doesn't contain the batch
    dimension. The spacings are individual per batch example.
    WARNING: this function assumes symmetric voxel sampling (without an offset)!
    """
    # Generate target coordinates
    batch_size_volumes = latents.shape[0]
    device = latents.device

    num_coords = coordinates.size(1)
    num_batches_coords = math.ceil(num_coords / batch_size_coords)
    labels_pred = torch.empty([batch_size_volumes, num_coords, 1, 1], dtype=torch.float32)
    for i in range(batch_size_volumes):
        coordinates = coordinates.flatten(1, 3).unsqueeze(2).unsqueeze(2)  # [1, N, 1, 1, 3]
        for j in range(num_batches_coords):
            first_id = j * batch_size_coords
            last_id = min((j + 1) * batch_size_coords, num_coords)
            # coordinates = torch.Size([1, 2097152, 1, 1, 3])
            coordinates_curr = coordinates[:, first_id:last_id]
            # Since every volume is processed independently, make it look like batch size 1 here
            latents_curr = latents[i].unsqueeze(0)
            with torch.no_grad():
                labels_pred[i, first_id:last_id] = occ_net(
                    latents_curr, coordinates_curr).detach().cpu()[0]
    # labels_pred = labels_pred.reshape(batch_size_volumes)
    return labels_pred  # [B, *ST]


def eval_batch(labels_pred: np.ndarray, labels_gt: np.ndarray, spacings: np.ndarray,
               dices: List[float], asds: List[float], hd95s: List[float],
               max_distances: List[float], verbose: bool):
    """Evaluate a batch of volumetric masks."""
    if labels_pred.shape != labels_gt.shape:
        raise ValueError(f'Batch evaluation not possible: predicted shape {labels_pred.shape} '
                         f'is different from GT shape {labels_gt.shape}.')
    batch_size = labels_pred.shape[0]
    # Iterate through batch examples
    for j in range(batch_size):
        label_pred = labels_pred[j, 0]
        label_gt = labels_gt[j, 0]

        # Empty GT/prediction mess up metrics calculations...
        if label_gt.sum() == 0:
            print('Warning: empty GT occured!')
        if label_pred.sum() == 0:
            print('Warning: empty prediciton occured!')

        dice = metrics.compute_dice_coefficient(label_gt, label_pred)
        spacing = spacings[j]
        # print(labels_gt.shape)
        # print(label_pred.shape)
        # print(spacing.shape)
        # (1, 1, 128, 128, 128)
        # (128, 128, 128)
        # (3,)
        surf_distances = metrics.compute_surface_distances(label_gt, label_pred, spacing) #True
        avg_distance_gt_to_pred, avg_distance_pred_to_gt = \
            metrics.compute_average_surface_distance(surf_distances)
        asd = (avg_distance_gt_to_pred + avg_distance_pred_to_gt) / 2
        hausdorff = metrics.compute_robust_hausdorff(surf_distances, 100)
        hausdorff95 = metrics.compute_robust_hausdorff(surf_distances, 95)

        dices.append(dice)
        asds.append(asd)
        hd95s.append(hausdorff95)
        max_distances.append(hausdorff)

    if verbose:
        print(f'Batch ASD: {np.mean(asds[-batch_size:]):.2f}, '
              f'HSD: {np.mean(max_distances[-batch_size:]):.2f}, '
              f'HSD95: {np.mean(hd95s[-batch_size:]):.2f}, '
              f'DSC: {np.mean(dices[-batch_size:]):.2f}', flush=True)
