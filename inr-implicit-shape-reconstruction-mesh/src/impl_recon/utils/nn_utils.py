from typing import Optional
import numpy as np
from scipy.spatial import cKDTree as KDTree
import torch
import torch.nn.functional as F
from torch.autograd.functional import jacobian
from torch.autograd import Function


def dice_coeff(probabilities: torch.Tensor, labels: torch.Tensor, threshold: Optional[float] = None,
               reduction: Optional[str] = 'mean') -> torch.Tensor:
    """Compute a mean hard or soft dice coefficient between a batch of probabilities and target
    labels. Reduction happens over the batch dimension; if None, return dice per example.
    """
    # This factor prevents division by 0 if both prediction and GT don't have any foreground voxels
    smooth = 1e-3

    if threshold is not None:
        probabilities = probabilities.gt(threshold).float()
    # Flatten all dims except for the batch
    probabilities_flat = torch.flatten(probabilities, start_dim=1)
    labels_flat = torch.flatten(labels, start_dim=1)

    intersection = (probabilities_flat * labels_flat).sum(dim=1)
    volume_sum = probabilities_flat.sum(dim=1) + labels_flat.sum(dim=1)  # it's not the union!
    dice = (2. * intersection + smooth) / (volume_sum + smooth)
    if reduction == 'mean':
        dice = torch.mean(dice)
    elif reduction == 'sum':
        dice = torch.sum(dice)

    return dice


class ChamferDistance(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, gt_points, gen_points):
        # one direction
        gen_points_kd_tree = KDTree(gen_points)
        one_distances, one_vertex_ids = gen_points_kd_tree.query(gt_points)
        gt_to_gen_chamfer = np.mean(np.square(one_distances))
        # other direction
        gt_points_kd_tree = KDTree(gt_points)
        two_distances, two_vertex_ids = gt_points_kd_tree.query(gen_points)
        gen_to_gt_chamfer = np.mean(np.square(two_distances))
        return gt_to_gen_chamfer + gen_to_gt_chamfer


class DiceLoss(torch.nn.Module):
    """Takes logits as input."""
    def __init__(self, threshold: Optional[float] = None, reduction: Optional[str] = 'mean',
                 do_report_metric: bool = False):
        """If no threshold is given, soft dice is computed, otherwise the predicted values are
        thresholded. Reduction happens over the batch dimension; if None, return dice per example.
        If do_report_metric, report the dice score instead of the dice loss (1 - dice score).
        """
        super().__init__()

        if not do_report_metric and threshold is not None:
            raise ValueError('Dice metric should not use thresholding when used as a loss.')

        self.threshold = threshold
        self.reduction = reduction
        self.do_report_metric = do_report_metric

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probabilities = torch.sigmoid(logits)
        if self.do_report_metric:
            return dice_coeff(probabilities, target, self.threshold, self.reduction)

        return 1.0 - dice_coeff(probabilities, target, self.threshold, self.reduction)


class BCEWithDiceLoss(torch.nn.Module):
    """Weighted sum of Dice loss with binary cross-entropy."""
    def __init__(self, reduction: str, bce_weight: float = 1.0):
        super().__init__()
        self.dice = DiceLoss(None, reduction, False)
        self.bce = torch.nn.BCEWithLogitsLoss(reduction=reduction)
        self.bce_weight = bce_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.bce_weight * self.bce(logits, target) + self.dice(logits, target)


class DiVRoC(Function):
    @staticmethod
    def forward(ctx, input, grid, shape):
        device = input.device
        dtype = input.dtype
        output = -jacobian(lambda x: (F.grid_sample(x, grid) - input).pow(2).mul(0.5).sum(), torch.zeros(shape,
                                                                                                         dtype=dtype,
                                                                                                         device=device))
        ctx.save_for_backward(input, grid, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, grid, output = ctx.saved_tensors
        B, C = input.shape[:2]
        input_dims = input.shape[2:]
        output_dims = grad_output.shape[2:]
        y = jacobian(lambda x: F.grid_sample(grad_output.unsqueeze(2).view(B * C, 1, *output_dims), x).mean(),
                     grid.unsqueeze(1).repeat(1, C, *([1] * (len(input_dims) + 1))).view(B * C, *input_dims,
                                                                                         len(input_dims))).view(B, C,
                                                                                                                *input_dims,
                                                                                                                len(input_dims))
        grad_grid = (input.numel() * input.unsqueeze(-1) * y).sum(1)
        grad_input = F.grid_sample(grad_output, grid)
        return grad_input, grad_grid, None
