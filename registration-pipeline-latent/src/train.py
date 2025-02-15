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
        latents_batch = latents[batch['caseids']].to(
            dtype=torch.float32, device=device)
        lat_reg = torch.mean(torch.sum(torch.square(latents_batch), dim=1))
        registration_gt = batch['registration_flow'].to(device)
        optimizer.zero_grad()
        coords = batch['moving_pc'].to(device)
        registration_pred = net(latents_batch, coords)
        loss = criterion(registration_pred, registration_gt, coords.squeeze(),
                         logger)
        if lat_reg is not None and lat_reg_lambda > 0:
            # Gradually build up for the first 100 epochs (follows DeepSDF)
            loss += min(1.0, epoch / 100) * lat_reg_lambda * lat_reg

        loss.backward()
        optimizer.step()
        global_step += 1


def validate(ds_loader, net, optimizer, criterion, metric, device, epoch,
             log_epoch_count, logger, val_function, tre, latents,
             lat_reg_lambda, latent_dim, lr_lats):
    epe_3d = []
    epe_initial = []
    net.eval()
    for batch in ds_loader:
        latent_vector = torch.normal(
            0.0,
            1 / math.sqrt(latent_dim), [1, latent_dim],
            requires_grad=True).to(
                torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        latent_vector = torch.nn.Parameter(latent_vector)
        latent_optimizer = torch.optim.Adam([{
            'params': latent_vector,
            'lr': lr_lats
        }])
        for step in range(10):  # Optimize latent space
            latent_optimizer.zero_grad()
            latent_batch = latent_vector.to(dtype=torch.float32, device=device)
            registration_gt = batch['registration_flow'].to(device)
            coords = batch['moving_pc'].to(device)
            registration_pred = net(latent_batch, coords)

            loss = criterion(registration_pred, registration_gt,
                             coords.squeeze(), logger)
            loss.backward()
            latent_optimizer.step()
        epe_3d_val = (registration_pred - registration_gt).square().sum(
            dim=2).sqrt().mean(dim=1).sum().item()
        epe_initial_val = registration_gt.square().sum(dim=2).sqrt().mean(
            dim=1).sum().item()
        epe_3d_val = epe_3d_val * ds_loader.dataset.norm_factor
        epe_initial_val = epe_initial_val * ds_loader.dataset.norm_factor
        epe_3d.append(epe_3d_val)
        epe_initial.append(epe_initial_val)
    epe_initial_np = np.array(epe_initial)
    epe_3d_np = np.array(epe_3d)
    wandb.log({f"val/epe_init": np.mean(epe_initial_np), "global_step": (0)})
    wandb.log({f"val/epe": np.mean(epe_3d_np), "global_step": (epoch)})


def test(ds_loader, net, device, criterion, latent_dim, lr_lats, logger):
    epe_3d = []
    epe_initial = []
    net.eval()
    for batch in ds_loader:
        latent_vector = torch.normal(
            0.0,
            1 / math.sqrt(latent_dim), [1, latent_dim],
            requires_grad=True).to(
                torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        latent_vector = torch.nn.Parameter(latent_vector)
        latent_optimizer = torch.optim.Adam([{
            'params': latent_vector,
            'lr': lr_lats
        }])
        for step in range(1000):  # Optimize latent space
            latent_optimizer.zero_grad()
            latent_batch = latent_vector.to(dtype=torch.float32, device=device)
            registration_gt = batch['registration_flow'].to(device)
            coords = batch['moving_pc'].to(device)
            registration_pred = net(latent_batch, coords)
            loss = criterion(registration_pred, registration_gt,
                             coords.squeeze(), logger)
            loss.backward()
            latent_optimizer.step()
        registration_pred = net(latent_vector, coords)
        epe_3d_val = (registration_pred - registration_gt).square().sum(
            dim=2).sqrt().mean(dim=1).sum().item()
        epe_initial_val = registration_gt.square().sum(dim=2).sqrt().mean(
            dim=1).sum().item()
        epe_3d_val = epe_3d_val * ds_loader.dataset.norm_factor
        epe_initial_val = epe_initial_val * ds_loader.dataset.norm_factor
        wandb.log({
            f"test/epe_pair_{batch['caseids']}": epe_initial,
            "global_step": (0)
        })
        wandb.log({
            f"test/epe_pair_{batch['caseids']}": epe_3d,
            "global_step": (1)
        })
        epe_3d.append(epe_3d_val)
        epe_initial.append(epe_initial_val)
    epe_initial_np = np.array(epe_initial)
    epe_3d_np = np.array(epe_3d)
    wandb.log({f"test/epe": np.mean(epe_initial_np), "global_step": (0)})
    wandb.log({f"test/epe": np.mean(epe_3d_np), "global_step": (1)})


def main():
    params, params_feat = config.parse_config()
    run = wandb.init(settings=wandb.Settings(_service_wait=300))
    model_dir = Path(os.path.join('/vol/aimspace/users/gob/Checkpoints/registration-pipeline', params['model_name'], run.name))\
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

    ds_loader_train = dataloader.create_data_loader(params['data_path'],
                                                    num_points_per_example,
                                                    'train')
    ds_loader_val = dataloader.create_data_loader(params['data_path'],
                                                  num_points_per_example,
                                                  'val')
    ds_loader_test = dataloader.create_data_loader(params['data_path'],
                                                   num_points_per_example,
                                                   'test')
    if not ds_loader_train:
        raise ValueError(
            f'Number of training examples is smaller than the batch size.')
    net = implicits.create_model(params, wandb, device)
    if torch.cuda.device_count() > 1:
        print(f'Using {torch.cuda.device_count()} GPUs for training.')
        net = torch.nn.DataParallel(net)
    net = net.to(device)
    print(net)

    num_examples = 104
    lr_lats = wandb.config.latent_learning_rate
    latent_dim = params['latent_dim']
    latents_train = torch.normal(
        0.0, 1 / math.sqrt(latent_dim), [num_examples, latent_dim]).to(
            torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

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

    tre = loss.create_loss("tre", device,
                           batch_zero['scaling_factor']).to(device)
    for epoch in range(num_epochs_trained, num_epochs_target):
        train_one_epoch(ds_loader_train, net, optimizer, criterion,
                        train_metric, device, epoch, num_epochs_target,
                        global_step, log_epoch_count, wandb, train_function,
                        latents_train, lat_reg_lambda)
        validate(ds_loader_val, net, optimizer, criterion, val_metric, device,
                 epoch, log_epoch_count, wandb, val_function, tre,
                 latents_train, lat_reg_lambda, latent_dim, lr_lats)
    test(ds_loader_test, net, device, criterion, latent_dim, lr_lats, wandb)
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
