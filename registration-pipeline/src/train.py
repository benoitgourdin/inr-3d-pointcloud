import os
import sys
import time
import shutil
from pathlib import Path
from matplotlib import cm
import numpy as np
import torch
import wandb
import math

from utils import dataloader, config, loss, file_creator
from model import implicits


def train_one_epoch(ds_loader, net, optimizer, criterion, metric, device,
                    epoch, num_epochs_target, global_step, log_epoch_count,
                    logger, train_function, latents, lat_reg_lambda):
    loss_running = 0.0
    num_losses = 0
    metric_running = 0.0
    num_metrics = 0
    coords = torch.Tensor()
    lat_reg = None
    registration_pred = torch.Tensor()
    registration_gt = torch.Tensor()
    t0 = time.time()
    net.train()
    pc_writer = file_creator.PointCloudWriter()
    for batch in ds_loader:

        # latents_batch = latents[batch['caseids']].to(
        #     dtype=torch.float32, device=device)
        # lat_reg = torch.mean(torch.sum(torch.square(latents_batch), dim=1))

        registration_gt = batch['registration_flow'].to(device)
        optimizer.zero_grad()
        coords = batch['moving_pc'].to(device)

        latents_batch = 0

        registration_pred = net(latents_batch, coords)
        loss = criterion(registration_pred, registration_gt, coords.squeeze(),
                         logger)

        # if lat_reg is not None and lat_reg_lambda > 0:
        #     # Gradually build up for the first 100 epochs (follows DeepSDF)
        #     loss += min(1.0, epoch / 100) * lat_reg_lambda * lat_reg

        loss.backward()
        optimizer.step()
        # loss_running += loss.item()
        # num_losses += 1
        # metric_running += metric(registration_pred, registration_gt, coords,
        #                          logger).item()
        # num_metrics += batch['registration_flow'].shape[0]
        global_step += 1

    # if epoch % log_epoch_count == 0:
    #     loss_avg = loss_running / num_losses
    #     metric_avg = metric_running / num_metrics
    #     num_epochs_trained = epoch + 1
    #     if logger is not None:
    #         logger.log({
    #             f"loss ({logger.config.loss_function})": loss_avg,
    #             "global_step": num_epochs_trained
    #         })
    #         logger.log({
    #             f"metric/train ({train_function})": metric_avg,
    #             "global_step": num_epochs_trained
    #         })

    #         if lat_reg is not None:
    #             # Log average squared norm of latents
    #             logger.log({
    #                 "average squared norm of latents": lat_reg.item(),
    #                 "global_step": num_epochs_trained
    #             })

    #     epoch_duration = time.time() - t0
    #     print(f'[{num_epochs_trained}/{num_epochs_target}] '
    #           f'Avg loss: {loss_avg:.4f}; '
    #           f'metric: {metric_avg:.3f}; '
    #           f'global step nb. {global_step} '
    #           f'({epoch_duration:.1f}s)')

    # if epoch == 0:
    #     moving_point_cloud = coords
    #     target_point_cloud = coords + registration_gt
    #     moving_point_cloud_np = moving_point_cloud.squeeze().cpu().detach(
    #     ).numpy()
    #     target_point_cloud_np = target_point_cloud.squeeze().cpu().detach(
    #     ).numpy()
    #     rgb_colors = np.tile([0, 0, 0], (target_point_cloud_np.shape[0], 1))
    #     # Combine positions (x, y, z) with colors (r, g, b)
    #     point_cloud_1_rgb = np.hstack((target_point_cloud_np, rgb_colors))
    #     rgb_colors = np.tile([200, 200, 200],
    #                          (moving_point_cloud_np.shape[0], 1))
    #     # Combine positions (x, y, z) with colors (r, g, b)
    #     point_cloud_2_rgb = np.hstack((moving_point_cloud_np, rgb_colors))
    #     point_cloud_rgb = np.vstack((point_cloud_1_rgb, point_cloud_2_rgb))
    #     # Log to WandB as 3D object
    #     logger.log({"moving - target": wandb.Object3D(point_cloud_rgb)})

    #     pc = coords.squeeze().cpu().detach().numpy()
    #     magnitudes = np.linalg.norm(
    #         np.array(registration_gt.squeeze().cpu().detach().numpy()), axis=1)
    #     norm = (magnitudes - np.min(magnitudes)) / (
    #         np.max(magnitudes) - np.min(magnitudes))

    #     # Map values to colors using a colormap
    #     colormap = cm.viridis(norm)  # Use a colormap (e.g., viridis)
    #     rgb_colors = (colormap[:, :3] * 255).astype(
    #         int)  # Convert to RGB 0-255
    #     # Combine positions (x, y, z) with colors (r, g, b)
    #     point_cloud_rgb = np.hstack((pc[:, :3], rgb_colors))
    #     # Log to WandB as 3D object
    #     logger.log(
    #         {"target displacement magnitude": wandb.Object3D(point_cloud_rgb)})
    # if epoch % (log_epoch_count*10) == 0:
    #     moved_point_cloud = coords + registration_pred
    #     moved_point_cloud_np = moved_point_cloud.squeeze().cpu().detach(
    #     ).numpy()
    #     pc = coords.squeeze().cpu().detach().numpy()
    #     magnitudes = np.linalg.norm(
    #         np.array(registration_pred.squeeze().cpu().detach().numpy()),
    #         axis=1)
    #     norm = (magnitudes - np.min(magnitudes)) / (
    #         np.max(magnitudes) - np.min(magnitudes))
    #     # Map values to colors using a colormap
    #     colormap = cm.viridis(norm)  # Use a colormap (e.g., viridis)
    #     rgb_colors = (colormap[:, :3] * 255).astype(
    #         int)  # Convert to RGB 0-255
    #     # Combine positions (x, y, z) with colors (r, g, b)
    #     point_cloud_rgb = np.hstack((pc[:, :3], rgb_colors))
    #     # Log to WandB as 3D object
    #     logger.log({
    #         "predicted displacement magnitude": wandb.Object3D(point_cloud_rgb)
    #     })
    #     target_point_cloud = coords + registration_gt
    #     target_point_cloud_np = target_point_cloud.squeeze().cpu().detach(
    #     ).numpy()
    #     rgb_colors = np.tile([0, 0, 0], (target_point_cloud_np.shape[0], 1))
    #     # Combine positions (x, y, z) with colors (r, g, b)
    #     point_cloud_1_rgb = np.hstack((target_point_cloud_np, rgb_colors))
    #     rgb_colors = np.tile([200, 200, 200],
    #                          (moved_point_cloud_np.shape[0], 1))
    #     # Combine positions (x, y, z) with colors (r, g, b)
    #     point_cloud_2_rgb = np.hstack((moved_point_cloud_np, rgb_colors))
    #     point_cloud_rgb = np.vstack((point_cloud_1_rgb, point_cloud_2_rgb))
    #     # Log to WandB as 3D object
    #     logger.log({"moved - target": wandb.Object3D(point_cloud_rgb)})


def validate(ds_loader, net, metric, device, epoch, log_epoch_count, logger,
             val_function, tre, latents):
    metric_running = 0.0
    num_metrics = 0
    epe_3d = 0
    epe_initial = 0
    t0 = time.time()
    net.eval()
    with torch.no_grad():
        for batch in ds_loader:
            # latents_batch = latents[batch['caseids']].to(device)

            registration_gt = batch['registration_flow'].to(device)
            coords = batch['moving_pc'].to(device)

            latents_batch = 0

            registration_pred = net(latents_batch, coords)

            # metric_running += metric(registration_pred, registration_gt,
            #                          coords, logger).item()
            # tre_mean, tre_values, percentiles = tre(registration_pred,
            #                                         registration_gt, coords,
            #                                         logger)
            # num_metrics += batch['registration_flow'].shape[0]

            epe_3d = (registration_pred - registration_gt).square().sum(
                dim=2).sqrt().mean(dim=1).sum().item()
            epe_initial = registration_gt.square().sum(dim=2).sqrt().mean(
                dim=1).sum().item()
            epe_3d = epe_3d * ds_loader.dataset.norm_factor
            epe_initial = epe_initial * ds_loader.dataset.norm_factor

    #     metric_avg = metric_running / num_metrics
    #     print(f'epe initial {epe_initial}')
    # if logger is not None:
    #     logger.log({
    #         f"metric/val ({val_function})": metric_avg,
    #         "global_step": (epoch + 1)
    #     })
    #     logger.log({"metric/tre": tre_mean, "global_step": (epoch + 1)})
    #     logger.log({"metric/epe": epe_3d, "global_step": (epoch + 1)})
    #     logger.log({
    #         "metric/start_epe": epe_initial,
    #         "global_step": (epoch + 1)
    #     })
    #     for key, value in percentiles.items():
    #         logger.log({
    #             f"metric/percentiles ({key})": value,
    #             "global_step": (epoch + 1)
    #         })
    # if epoch % (log_epoch_count*10) == 0:
    #     # Visualization
    #     moving_point_cloud_np = coords.squeeze().cpu().detach().numpy()
    #     target_point_cloud = coords + registration_gt
    #     target_point_cloud_np = target_point_cloud.squeeze().cpu().detach(
    #     ).numpy()
    #     moved_point_cloud = coords + registration_pred
    #     moved_point_cloud_np = moved_point_cloud.squeeze().cpu().detach(
    #     ).numpy()
    #     norm = (tre_values - np.min(tre_values)) / (
    #         np.max(tre_values) - np.min(tre_values))
    #     # Map values to colors using a colormap
    #     colormap = cm.viridis(norm)  # Use a colormap (e.g., viridis)
    #     rgb_colors = (colormap[:, :3] * 255).astype(
    #         int)  # Convert to RGB 0-255
    #     # Combine positions (x, y, z) with colors (r, g, b)
    #     point_cloud_2_rgb = np.hstack(
    #         (moving_point_cloud_np[:, :3], rgb_colors))
    #     rgb_colors = np.tile([0, 0, 0], (target_point_cloud_np.shape[0], 1))
    #     # Combine positions (x, y, z) with colors (r, g, b)
    #     point_cloud_1_rgb = np.hstack((target_point_cloud_np, rgb_colors))
    #     # Combine positions (x, y, z) with colors (r, g, b)
    #     point_cloud_rgb = np.vstack((point_cloud_1_rgb, point_cloud_2_rgb))
    #     # Log to WandB as 3D object
    #     wandb.log({"TRE magnitude": wandb.Object3D(point_cloud_2_rgb)})
    #     wandb.log(
    #         {"TRE magnitude with target": wandb.Object3D(point_cloud_rgb)})

    #     tre_threshold = 3
    #     threshold_indices = np.where(tre_values > tre_threshold)
    #     non_threshold_indices = np.where(tre_values <= tre_threshold)
    #     rgb_colors = np.zeros((moving_point_cloud_np.shape[0], 3))
    #     rgb_colors[threshold_indices] = [255, 10, 10]
    #     rgb_colors[non_threshold_indices] = [10, 255, 10]
    #     pc1_rgb = np.hstack((moving_point_cloud_np, rgb_colors))
    #     wandb.log({
    #         f"TRE Threshold: {tre_threshold}":
    #             wandb.Object3D(np.vstack((pc1_rgb, point_cloud_1_rgb)))
    #     })

    print(f'epe initial {epe_initial}, epe 3d {epe_3d}')

    t1 = time.time()
    val_duration = t1 - t0
    # print(f'[val] metric {metric_avg:.3f} ({val_duration:.1f}s)')
    return epe_initial, epe_3d


def main():
    pairs = np.array([0, 1, 3, 4, 5, 6, 7, 9, 10, 11])
    epe_init_overall = []
    epe_overall = []

    for pair in pairs:
        print(f'Pair {pair}')
        generalized = False

        params, params_feat = config.parse_config()
        run = wandb.init(settings=wandb.Settings(_service_wait=300))
        model_dir = Path(os.path.join('/vol/aimspace/users/gob/Checkpoints/registration-pipeline', params['model_name'], run.name, str(pair)))\
            if params['model_name'] is not None else None
        # config file parameters
        num_epochs_target = params['num_epochs']
        log_epoch_count = params['log_epoch_count']
        checkpoint_epoch_count = params['checkpoint_epoch_count']
        max_num_checkpoints = params['max_num_checkpoints']
        train_function = params["train_metric"]
        val_function = params["val_metric"]
        lat_reg_lambda = params['lat_reg_lambda']
        # wandb parameters
        learning_rate = wandb.config.learning_rate
        num_points_per_example = wandb.config.sample_batch_share
        loss_function = wandb.config.loss_function
        optimizer_param = wandb.config.optimizer
        if model_dir is not None:
            if model_dir.exists():
                shutil.rmtree(model_dir)
            model_dir.mkdir(parents=True, exist_ok=True)
            sys.stdout = file_creator.Logger(model_dir / 'log.txt', 'a')
            checkpoint_writer = file_creator.RollingCheckpointWriter(
                model_dir, 'checkpoint', max_num_checkpoints, 'pth')
        else:
            print(
                'Warning: no model name provided; not writing anything to the file system.'
            )
            checkpoint_writer = None
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not torch.cuda.is_available():
            print('Warning: no GPU available; training on CPU.')

        ds_loader_train = dataloader.create_data_loader(
            params['data_path'], num_points_per_example, 'train', pair)
        ds_loader_val = dataloader.create_data_loader(params['data_path'],
                                                      num_points_per_example,
                                                      'val', pair)
        if not ds_loader_train:
            raise ValueError(
                f'Number of training examples is smaller than the batch size.')
        net = implicits.create_model(params, wandb, device)
        if torch.cuda.device_count() > 1:
            print(f'Using {torch.cuda.device_count()} GPUs for training.')
            net = torch.nn.DataParallel(net)
        net = net.to(device)
        print(net)

        if generalized:
            num_examples = 104
            lr_lats = wandb.config.latent_learning_rate
            latent_dim = params['latent_dim']
            latents_train = torch.normal(0.0, 1 / math.sqrt(latent_dim), [
                num_examples, latent_dim
            ]).to(
                torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        else:
            latents_train = torch.nn.Parameter(torch.empty(0))

        if not generalized:
            if optimizer_param == "adam":
                optimizer = torch.optim.Adam([{
                    'params': net.parameters(),
                    'lr': learning_rate
                }])
            elif optimizer_param == "sgd":
                optimizer = torch.optim.SGD([{
                    'params': net.parameters(),
                    'lr': learning_rate
                }])
            else:
                optimizer = torch.optim.Adam([{
                    'params': net.parameters(),
                    'lr': learning_rate
                }])
        else:
            if optimizer_param == "adam":
                optimizer = torch.optim.Adam([{
                    'params': net.parameters(),
                    'lr': learning_rate
                }, {
                    'params': latents_train,
                    'lr': lr_lats
                }])
            elif optimizer_param == "sgd":
                optimizer = torch.optim.SGD([{
                    'params': net.parameters(),
                    'lr': learning_rate
                }, {
                    'params': latents_train,
                    'lr': lr_lats
                }])
            else:
                optimizer = torch.optim.Adam([{
                    'params': net.parameters(),
                    'lr': learning_rate
                }, {
                    'params': latents_train,
                    'lr': lr_lats
                }])

        criterion = loss.create_loss(loss_function, device).to(device)
        train_metric = loss.create_loss(train_function, device).to(device)
        val_metric = loss.create_loss(val_function, device).to(device)
        global_step = torch.tensor(0, dtype=torch.int64)
        num_epochs_trained = 0

        batch_zero = next(iter(ds_loader_val))
        # registration_gt = batch_zero['registration_flow'].to(device)
        # coords = batch_zero['moving_pc'].to(device)

        tre = loss.create_loss("tre", device,
                               batch_zero['scaling_factor']).to(device)
        # wandb.log({
        #     "metric/start_mse":
        #         train_metric(
        #             torch.zeros_like(registration_gt), registration_gt, coords,
        #             wandb).item(),
        #     "global_step": (0)
        # })
        # wandb.log({
        #     "metric/start_mae":
        #         val_metric(
        #             torch.zeros_like(registration_gt), registration_gt, coords,
        #             wandb).item(),
        #     "global_step": (0)
        # })
        # tre_mean, tre_values, percentiles = tre(
        #     torch.zeros_like(registration_gt), registration_gt, coords, wandb)
        # wandb.log({"metric/start_tre": tre_mean, "global_step": (0)})
        # for key, value in percentiles.items():
        #     wandb.log({
        #         f"metric/start_percentiles ({key})": value,
        #         "global_step": (0)
        #     })
        epe_values = []
        for epoch in range(num_epochs_trained, num_epochs_target):
            train_one_epoch(ds_loader_train, net, optimizer, criterion,
                            train_metric, device, epoch, num_epochs_target,
                            global_step, log_epoch_count, wandb,
                            train_function, latents_train, lat_reg_lambda)
            # if epoch % log_epoch_count == 0:
            epe_initial, epe_3d = validate(ds_loader_val, net, val_metric,
                                           device, epoch, log_epoch_count,
                                           wandb, val_function, tre,
                                           latents_train)
            # if checkpoint_writer is not None and epoch % checkpoint_epoch_count == 0:
            #     checkpoint_writer.write_rolling_checkpoint(
            #         {'net': net.state_dict()}, optimizer.state_dict(),
            #         int(global_step.item()), epoch + 1)
            epe_values.append(epe_3d)
            wandb.log({
                f"metric/epe {pair}": epe_3d,
                "global_step": (epoch + 1)
            })
            wandb.log({
                f"metric/epe_init {pair}": epe_initial,
                "global_step": (epoch + 1)
            })
        epe_init_overall.append(epe_initial)
        epe_overall.append(epe_values)
    epe_overall = np.array(epe_overall)
    epe_init_overall = np.array(epe_init_overall)
    for ep in range(epe_overall.shape[1]):
        wandb.log({
            f"metric/epe": np.mean(epe_overall[:, ep]),
            "global_step": (ep)
        })
        # if checkpoint_writer is not None:
        #     checkpoint_writer.write_rolling_checkpoint(
        #         {'net': net.state_dict()}, optimizer.state_dict(),
        #         int(global_step.item()), num_epochs_target)
    wandb.log({
        f"metric/epe_init": np.mean(epe_init_overall),
        "global_step": (0)
    })
    print(f'epe_overall_mean: {np.mean(epe_overall[:, -1])}')
    print(f'epe_init_overall_mean: {np.mean(epe_init_overall)}')
    wandb.finish()


if __name__ == '__main__':
    wandb.login()
    sweep_configuration = {
        "method": "grid",
        "metric": {
            "goal": "minimize",
            "name": "loss"
        },
        "parameters": {
            "learning_rate": {
                "values": [0.5e-3, 0.5e-4]
            },
            "latent_learning_rate": {
                "values": [1.0e-3]
            },
            "sample_batch_share": {
                "values": [1.0]
            },
            "loss_function": {
                "values": ["pointpwc"]
            },
            "chamfer_w": {
                "values": [1.0, 2.0]
            },
            "smoothness_w": {
                "values": [0.5]  #2.0
            },
            "smoothness_k": {
                "values": [4, 5]  #2.0
            },
            "curvature_w": {
                "values": [0.0]  #2.0
            },
            "optimizer": {
                "values": ["adam"]
            },
            "dropout": {
                "values": [0.0]
            },
            "hidden_layer_size": {
                "values": [128]
            },
            "activation_function": {
                "values": ["relu"]
            },
            "encoding": {
                "values": [True]
            },
            "sigma": {
                "values": [0.1]
            },
        },
    }
    params, _ = config.parse_config()
    sweep_id = wandb.sweep(
        sweep=sweep_configuration, project=params['model_name'])
    wandb.agent(sweep_id, function=main)
