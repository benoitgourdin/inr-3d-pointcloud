'''Implements a generic training loop.
'''

import torch
import psutil
import open3d as o3d
import utils
import sdf_meshing
# from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
import time
import numpy as np
import os
import shutil
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree


def square_distance(src, dst):
    b, n, _ = src.shape
    _, m, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src**2, -1).view(b, n, 1)
    dist += torch.sum(dst**2, -1).view(b, 1, m)
    return dist


# class ChamferDistance(torch.nn.Module):

#     def __init__(self):
#         super().__init__()

#     def forward(self, pc1, pc2):
#         # PointPWC implementation
#         sqr_dist12 = square_distance(pc1, pc2)
#         dist1, _ = torch.topk(
#             sqr_dist12, 1, dim=-1, largest=False, sorted=False)
#         dist2, _ = torch.topk(
#             sqr_dist12, 1, dim=1, largest=False, sorted=False)
#         dist1 = dist1.squeeze(2)
#         dist2 = dist2.squeeze(1)
#         return dist1.sum(dim=1).mean() + dist2.sum(dim=1).mean()


class ChamferDistance(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, gt_points_l, gen_points_l):
        gt_points = gt_points_l.squeeze().cpu().numpy()
        gen_points = gen_points_l.squeeze().cpu().numpy()
        # one direction
        gen_points_kd_tree = KDTree(gen_points)
        one_distances, one_vertex_ids = gen_points_kd_tree.query(gt_points)
        gt_to_gen_chamfer = np.mean(np.square(one_distances))
        # other direction
        gt_points_kd_tree = KDTree(gt_points)
        two_distances, two_vertex_ids = gt_points_kd_tree.query(gen_points)
        gen_to_gt_chamfer = np.mean(np.square(two_distances))
        return gt_to_gen_chamfer + gen_to_gt_chamfer


class EarthMoversDistance(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, gt_pc, gen_pc):
        d = cdist(gt_pc.squeeze().cpu().numpy()[:5000, :],
                  gen_pc.squeeze().cpu().numpy()[:5000, :])
        assignment = linear_sum_assignment(d)
        return d[assignment].mean()


def train(wandb,
          device,
          model,
          train_dataloader,
          epochs,
          lr,
          epochs_til_summary,
          epochs_til_checkpoint,
          model_dir,
          loss_fn,
          summary_fn,
          val_dataloader=None,
          double_precision=False,
          clip_grad=False,
          use_lbfgs=False,
          loss_schedules=None):
    saved = False
    optim = torch.optim.Adam(lr=lr, params=model.parameters())
    cd = ChamferDistance()
    emd = EarthMoversDistance()
    # copy settings from Raissi et al. (2019) and here
    # https://github.com/maziarraissi/PINNs
    if use_lbfgs:
        optim = torch.optim.LBFGS(
            lr=lr,
            params=model.parameters(),
            max_iter=50000,
            max_eval=50000,
            history_size=50,
            line_search_fn='strong_wolfe')

    if os.path.exists(model_dir):
        val = input("The model directory %s exists. Overwrite? (y/n)"
                    % model_dir)
        if val == 'y':
            shutil.rmtree(model_dir)

    os.makedirs(model_dir)

    summaries_dir = os.path.join(model_dir, 'summaries')
    utils.cond_mkdir(summaries_dir)

    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    utils.cond_mkdir(checkpoints_dir)

    # writer = SummaryWriter(summaries_dir)

    total_steps = 0
    with tqdm(total=len(train_dataloader) * epochs) as pbar:
        train_losses = []
        for epoch in range(epochs):
            # while total_steps < 70000:
            if device == 'cuda':
                torch.cuda.reset_max_memory_allocated()
                free, total = torch.cuda.mem_get_info(device)
                print(f"free cuda memory: {free}, total cuda memory: {total}")
            if not epoch % epochs_til_checkpoint and total_steps:
                torch.save(
                    model.state_dict(),
                    os.path.join(checkpoints_dir,
                                 'model_epoch_%04d.pth' % epoch))
                np.savetxt(
                    os.path.join(checkpoints_dir,
                                 'train_losses_epoch_%04d.txt' % epoch),
                    np.array(train_losses))

            for step, (model_input, gt) in enumerate(train_dataloader):
                start_time = time.time()

                model_input = {
                    key: value.to(device) for key, value in model_input.items()
                }
                gt = {key: value.to(device) for key, value in gt.items()}

                if double_precision:
                    model_input = {
                        key: value.double()
                        for key, value in model_input.items()
                    }
                    gt = {key: value.double() for key, value in gt.items()}

                if use_lbfgs:

                    def closure():
                        optim.zero_grad()
                        model_output = model(model_input)
                        losses = loss_fn(model_output, gt)
                        train_loss = 0.
                        for loss_name, loss in losses.items():
                            train_loss += loss.mean()
                        train_loss.backward()
                        return train_loss

                    optim.step(closure)

                model_output = model(model_input)
                losses = loss_fn(model_output, gt)

                train_loss = 0.
                for loss_name, loss in losses.items():
                    single_loss = loss.mean()

                    if loss_schedules is not None and loss_name in loss_schedules:
                        # writer.add_scalar(
                        #     loss_name + "_weight",
                        #     loss_schedules[loss_name](total_steps),
                        #     total_steps)
                        single_loss *= loss_schedules[loss_name](total_steps)

                    # writer.add_scalar(loss_name, single_loss, total_steps)
                    train_loss += single_loss
                    # wandb.log({
                    #     f"metric/train ({loss_name})": single_loss,
                    #     "global_step": total_steps
                    # })

                train_losses.append(train_loss.item())
                # writer.add_scalar("total_train_loss", train_loss, total_steps)
                wandb.log({
                    f"metric/train (train_loss)": train_loss.item(),
                    "global_step": total_steps
                })

                # if not total_steps % steps_til_summary:
                #     torch.save(
                #         model.state_dict(),
                #         os.path.join(checkpoints_dir, 'model_current.pth'))
                #     # summary_fn(model, device, model_input, gt, model_output,
                #     #            wandb, total_steps)

                if not use_lbfgs:
                    optim.zero_grad()
                    train_loss.backward(retain_graph=False)

                    if clip_grad:
                        if isinstance(clip_grad, bool):
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), max_norm=1.)
                        else:
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), max_norm=clip_grad)

                    optim.step()

                pbar.update(1)

            if not epoch % epochs_til_summary:
                # sdf_meshing.create_mesh(
                #     device,
                #     decoder,
                #     os.path.join(model_dir, 'test'),
                #     N=100)
                tqdm.write("Step %d, Total loss %0.6f, iteration time %0.6f" %
                           (total_steps, train_loss, time.time() - start_time))

                if val_dataloader is not None:
                    print("Running validation set...")
                    model.eval()
                    with torch.no_grad():
                        val_losses = []
                        for (model_input, gt) in val_dataloader:
                            model_input = {
                                key: value.to(device)
                                for key, value in model_input.items()
                            }
                            gt = {
                                key: value.to(device)
                                for key, value in gt.items()
                            }
                            gt_pointcloud = model_input['coords'][(
                                gt['sdf'] == 0).squeeze(-1)]
                            rand_input = torch.rand(1, 1000000, 3) * 2 - 1
                            model_input = {'coords': rand_input.to(device)}
                            model_output = model(model_input)
                            _, indices = torch.topk(
                                torch.abs(model_output['model_out']),
                                30000,
                                dim=1,
                                largest=False)
                            predicted_pointcloud = model_output['model_in'][
                                torch.arange(model_output['model_in'].size(0)
                                            ).unsqueeze(-1), indices]
                            gt_pointcloud = gt_pointcloud.unsqueeze(0).float()
                            predicted_pointcloud = predicted_pointcloud.squeeze(
                                2).float()
                        # Compute Chamfer Distance
                        cd_loss = cd(gt_pointcloud, predicted_pointcloud)
                        wandb.log({
                            f"metric/val (chamfer distance)": cd_loss,
                            "global_step": total_steps
                        })
                        if epoch % 50 == 0:
                            # Compute Earth Mover's Distance
                            emd_loss = emd(gt_pointcloud, predicted_pointcloud)
                            wandb.log({
                                f"metric/val (earth mover's distance)":
                                    emd_loss,
                                "global_step":
                                    total_steps
                            })
                            if not saved:
                                gt_pointcloud_np = gt_pointcloud.squeeze().cpu(
                                ).numpy()
                                wandb.log({
                                    "Source Point Cloud":
                                        wandb.Object3D(gt_pointcloud_np)
                                })
                                saved = True
                                gt_path = os.path.join(model_dir,
                                                       'checkpoints',
                                                       f'ground_truth.ply')
                                pcd = o3d.geometry.PointCloud()
                                pcd.points = o3d.utility.Vector3dVector(
                                    gt_pointcloud_np)
                                o3d.io.write_point_cloud(gt_path, pcd)
                            predicted_pointcloud_np = predicted_pointcloud.squeeze(
                            ).cpu().numpy()
                            wandb.log({
                                "Reconstructed Point Cloud":
                                    wandb.Object3D(predicted_pointcloud_np)
                            })
                            predicted_path = os.path.join(
                                model_dir, 'checkpoints',
                                f'predicted_{epoch}.ply')
                            pcd = o3d.geometry.PointCloud()
                            pcd.points = o3d.utility.Vector3dVector(
                                predicted_pointcloud_np)
                            o3d.io.write_point_cloud(predicted_path, pcd)

                        # writer.add_scalar("val_loss", np.mean(val_losses),
                        #                   total_steps)
                    model.train()

                total_steps += 1

        torch.save(model.state_dict(),
                   os.path.join(checkpoints_dir, 'model_final.pth'))
        np.savetxt(
            os.path.join(checkpoints_dir, 'train_losses_final.txt'),
            np.array(train_losses))
    wandb.finish()


class LinearDecaySchedule():

    def __init__(self, start_val, final_val, num_steps):
        self.start_val = start_val
        self.final_val = final_val
        self.num_steps = num_steps

    def __call__(self, iter):
        return self.start_val + (self.final_val - self.start_val) * min(
            iter / self.num_steps, 1.)
