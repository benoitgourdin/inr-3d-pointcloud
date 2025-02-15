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
                    logger, train_function):
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
        registration_gt = batch['registration_flow'].to(device)
        optimizer.zero_grad()
        coords = batch['moving_pc'].to(device)

        latents_batch = 0

        registration_pred = net(latents_batch, coords)
        loss = criterion(registration_pred, registration_gt, coords.squeeze(),
                         logger)

        loss.backward()
        optimizer.step()
        global_step += 1

def validate(ds_loader, net, metric, device, epoch, log_epoch_count, logger,
             val_function, tre):
    metric_running = 0.0
    num_metrics = 0
    epe_3d = 0
    epe_initial = 0
    t0 = time.time()
    net.eval()
    with torch.no_grad():
        for batch in ds_loader:
            registration_gt = batch['registration_flow'].to(device)
            coords = batch['moving_pc'].to(device)

            latents_batch = 0

            registration_pred = net(latents_batch, coords)

            epe_3d = (registration_pred - registration_gt).square().sum(
                dim=2).sqrt().mean(dim=1).sum().item()
            epe_initial = registration_gt.square().sum(dim=2).sqrt().mean(
                dim=1).sum().item()
            epe_3d = epe_3d * ds_loader.dataset.norm_factor
            epe_initial = epe_initial * ds_loader.dataset.norm_factor

    print(f'epe initial {epe_initial}, epe 3d {epe_3d}')

    t1 = time.time()
    val_duration = t1 - t0
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
        model_dir = Path(os.path.join('/.../Checkpoints/registration-pipeline', params['model_name'], run.name, str(pair)))\
            if params['model_name'] is not None else None
        # config file parameters
        num_epochs_target = params['num_epochs']
        log_epoch_count = params['log_epoch_count']
        checkpoint_epoch_count = params['checkpoint_epoch_count']
        max_num_checkpoints = params['max_num_checkpoints']
        train_function = params["train_metric"]
        val_function = params["val_metric"]
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

        criterion = loss.create_loss(loss_function, device).to(device)
        train_metric = loss.create_loss(train_function, device).to(device)
        val_metric = loss.create_loss(val_function, device).to(device)
        global_step = torch.tensor(0, dtype=torch.int64)
        num_epochs_trained = 0

        batch_zero = next(iter(ds_loader_val))

        tre = loss.create_loss("tre", device,
                               batch_zero['scaling_factor']).to(device)
        epe_values = []
        for epoch in range(num_epochs_trained, num_epochs_target):
            train_one_epoch(ds_loader_train, net, optimizer, criterion,
                            train_metric, device, epoch, num_epochs_target,
                            global_step, log_epoch_count, wandb,
                            train_function)
            epe_initial, epe_3d = validate(ds_loader_val, net, val_metric,
                                           device, epoch, log_epoch_count,
                                           wandb, val_function, tre)
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
                "values": [0.5]
            },
            "smoothness_k": {
                "values": [4, 5]
            },
            "curvature_w": {
                "values": [0.0]
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
