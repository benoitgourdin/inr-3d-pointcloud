import math
import sys
import time
import shutil
from pathlib import Path
from typing import Optional
import numpy as np
import h5py

import torch
from torch.optim import optimizer as opt
from torch.utils import data
import wandb

from impl_recon.models import implicits
from impl_recon.utils import config_io, data_generation, io_utils, nn_utils


def create_model(params: dict, image_size: Optional[torch.Tensor], dropout_p: float,  activation: str,
                 hidden_layers: int, encoding) -> torch.nn.Module:
    task_type = params['task_type']

    net: torch.nn.Module
    if task_type == config_io.TaskType.AD:
        if image_size is None:
            raise ValueError('Image size is required for AD model creation.')
        latent_dim = params['latent_dim']
        op_num_layers = params['op_num_layers']
        op_coord_layers = params['op_coord_layers']
        net = implicits.AutoDecoder(latent_dim, len(image_size), image_size,
                                    op_num_layers, op_coord_layers, dropout_p,
                                    activation, hidden_layers, encoding)
    elif task_type == config_io.TaskType.RN:
        net = implicits.ReconNet()
    else:
        raise ValueError(f'Unknown task type {task_type}.')
    return net


def create_loss(loss_function) -> torch.nn.Module:
    if loss_function == "dice_bce":
        return nn_utils.BCEWithDiceLoss('mean', 1.0)
    elif loss_function == "dice":
        return nn_utils.DiceLoss(None, 'mean', False)
    elif loss_function == "divroc":
        return torch.nn.HuberLoss(reduction='mean', delta=1.0)


def make_coordinate_tensor(dims=(28, 28, 28), integer=False):
    """Make a coordinate tensor."""
    if integer:
        coordinate_tensor = [torch.linspace(0, dims[i], dims[i]) for i in range(3)]
    else:
        coordinate_tensor = [torch.linspace(-1, 1, dims[i]) for i in range(3)]
    coordinate_tensor = torch.meshgrid(*coordinate_tensor)
    coordinate_tensor = torch.stack(coordinate_tensor, dim=3)
    coordinate_tensor = coordinate_tensor.cuda()
    return coordinate_tensor


def train_one_epoch(task_type: config_io.TaskType, ds_loader: data.DataLoader, net: torch.nn.Module,
                    latents: torch.nn.Parameter, lat_reg_lambda: float,
                    optimizer: opt.Optimizer, criterion: torch.nn.Module, metric: torch.nn.Module,
                    device: torch.device, epoch: int, num_epochs_target: int,
                    global_step: torch.Tensor, log_epoch_count: int,
                    logger: wandb, verbose: bool):
    # labels, caseids, coords
    loss_running = 0.0
    num_losses = 0
    metric_running = 0.0
    num_metrics = 0
    lat_reg = None
    t0 = time.time()
    net.train()
    for batch in ds_loader:
        labels = batch['labels'].to(device)
        optimizer.zero_grad()
        if task_type == config_io.TaskType.AD:
            latents_batch = latents[batch['caseids']].to(dtype=torch.float32, device=device)
            coords = batch['coords'].to(dtype=torch.float32, device=device)
            labels_pred = net(latents_batch, coords)
            lat_reg = torch.mean(torch.sum(torch.square(latents_batch), dim=1))
        elif task_type == config_io.TaskType.RN:
            labels_lr = batch['labels_lr'].to(device)
            labels_pred = net(labels_lr)
        else:
            raise ValueError(f'Unknown task type {task_type}.')
        if wandb.config.loss_function == "divroc":
            gt_coords = coords.squeeze().to(dtype=torch.float32, device=device)
            pred_coords = coords.squeeze().to(dtype=torch.float32, device=device)
            input_gt = labels.squeeze().unsqueeze(0).unsqueeze(2).unsqueeze(3).unsqueeze(4).to(device)
            grid_gt = gt_coords.squeeze().unsqueeze(0).unsqueeze(2).unsqueeze(3).to(device)
            input_pred = labels_pred.squeeze().unsqueeze(0).unsqueeze(2).unsqueeze(3).unsqueeze(4).to(device)
            grid_pred = pred_coords.squeeze().unsqueeze(0).unsqueeze(2).unsqueeze(3).to(device)
            shape = [1, 1, 128, 128, 128]
            loss = criterion(nn_utils.DiVRoC.apply(input_pred, grid_pred, shape), nn_utils.DiVRoC.apply(input_gt, grid_gt, shape))
        else:
            loss = criterion(labels_pred, labels)
        if lat_reg is not None and lat_reg_lambda > 0:
            # Gradually build up for the first 100 epochs (follows DeepSDF)
            loss += min(1.0, epoch / 100) * lat_reg_lambda * lat_reg
        loss.backward()
        optimizer.step()

        loss_running += loss.item()
        num_losses += 1
        # Metric returns the sum
        metric_running += metric(labels_pred, labels).item()
        num_metrics += batch['labels'].shape[0]
        global_step += 1

    if epoch % log_epoch_count == 0:
        loss_avg = loss_running / num_losses
        metric_avg = metric_running / num_metrics
        num_epochs_trained = epoch + 1

        if logger is not None:
            logger.log({f"loss ({logger.config.loss_function})": loss_avg, "global_step": num_epochs_trained})
            logger.log({f"metric/train (dice loss)": metric_avg, "global_step": num_epochs_trained})
            if lat_reg is not None:
                # Log average squared norm of latents
                logger.log({"average squared norm of latents": lat_reg.item(), "global_step": num_epochs_trained})

        if verbose:
            epoch_duration = time.time() - t0

            print(f'[{num_epochs_trained}/{num_epochs_target}] '
                  f'Avg loss: {loss_avg:.4f}; '
                  f'metric: {metric_avg:.3f}; '
                  f'global step nb. {global_step} '
                  f'({epoch_duration:.1f}s)')


def optimize_latents(net: implicits.AutoDecoder,
                     latents_batch: torch.Tensor, labels: torch.Tensor, coords: torch.Tensor,
                     lr: float, lat_reg_lambda: float, num_iters: int,
                     device: torch.device,
                     max_num_const_train_dsc: int,
                     verbose: bool) -> None:
    """Optimize latent vectors for a single example.
    max_num_const_train_dsc: if train dice doesn't change this number of times, stop training. -1
                             means never stop early.
    """
    criterion = create_loss("dice_bce").to(device)
    optimizer_val = torch.optim.Adam([latents_batch], lr=lr)

    eval_every_x_steps = 10
    print_every_x_evals = 10
    prev_train_dsc = 0.0
    num_const_train_dsc = 0

    net.eval()

    t0 = time.time()
    for i in range(num_iters):
        labels_pred = net(latents_batch, coords)
        loss = criterion(labels_pred, labels)
        if lat_reg_lambda > 0:
            lat_reg = torch.mean(torch.sum(torch.square(latents_batch), dim=1))
            # Gradually build up regularization for the first 100 iters (follows DeepSDF)
            loss += min(1.0, i / 100) * lat_reg_lambda * lat_reg
        optimizer_val.zero_grad()
        loss.backward()
        optimizer_val.step()
        if (i + 1) % eval_every_x_steps == 0:
            dsc = nn_utils.dice_coeff(torch.sigmoid(labels_pred), labels, 0.5).item()
            if verbose and round((i + 1) / eval_every_x_steps) % print_every_x_evals == 0:
                print(f'Step {i + 1:04d}/{num_iters:04d}: loss {loss.item():.4f} DSC {dsc:.3f} '
                      f'L2^2(z): {torch.mean(torch.sum(torch.square(latents_batch), dim=1)):.2f} '
                      f'({time.time() - t0:.1f}s)')
                t0 = time.time()
            if round(dsc, 3) == round(prev_train_dsc, 3):
                num_const_train_dsc += 1
            else:
                num_const_train_dsc = 0
            if num_const_train_dsc == max_num_const_train_dsc:
                print(f'Reached stopping critertion after {i + 1} steps. '
                      f'Optimization has converged.')
                break
            prev_train_dsc = dsc


def validate(task_type: config_io.TaskType, ds_loader: data.DataLoader, net: torch.nn.Module,
             latents: torch.nn.Parameter, metric: torch.nn.Module, device: torch.device, epoch: int,
             logger: wandb, verbose: bool):
    metric_running = 0.0
    num_metrics = 0

    t0 = time.time()
    net.eval()
    with torch.no_grad():
        for batch in ds_loader:
            labels = batch['labels'].to(device)
            if task_type == config_io.TaskType.AD:
                latents_batch = latents[batch['caseids']].to(device)
                coords = batch['coords'].to(device)
                labels_pred = net(latents_batch, coords)
            elif task_type == config_io.TaskType.RN:
                labels_lr = batch['labels_lr'].to(device)
                labels_pred = net(labels_lr)
            else:
                raise ValueError(f'Unknown task type {task_type}.')

            # Metric returns the sum
            pred_array = torch.sigmoid(labels_pred).gt(0.5).squeeze(0).squeeze(1).squeeze(1).cpu().detach().numpy()
            labels_array = labels.squeeze(0).squeeze(1).squeeze(1).cpu().detach().numpy()
            indices_pred = np.where(pred_array)[0]
            indices_true = np.where(labels_array == 1)[0]
            coords_array = coords.squeeze(0).squeeze(1).squeeze(1).cpu().detach().numpy()
            metric_running += metric(coords_array[indices_pred], coords_array[indices_true]).item()

            num_metrics += batch['labels'].shape[0]

        metric_avg = metric_running / num_metrics

    if logger is not None:
        logger.log({"metric/val (Chamfer Distance)": metric_avg, "global_step": (epoch + 1)})

    if verbose:
        t1 = time.time()
        val_duration = t1 - t0
        print(f'[val] metric {metric_avg:.3f} ({val_duration:.1f}s)')


def main():
    params, config_filepath = config_io.parse_config_train()
    run = wandb.init()
    model_basedir: Path = params['model_basedir']
    model_dir = model_basedir / params['model_name'] / run.name if params['model_name'] is not None else None
    task_type = params['task_type']
    learning_rate = params['learning_rate'] * params['batch_size_train']
    lat_reg_lambda = params['lat_reg_lambda']
    num_epochs_target = params['num_epochs']
    log_epoch_count = params['log_epoch_count']
    checkpoint_epoch_count = params['checkpoint_epoch_count']
    max_num_checkpoints = params['max_num_checkpoints']
    # wandb parameters
    learning_rate = wandb.config.learning_rate
    num_points_per_example = wandb.config.sample_batch_share
    loss_function = wandb.config.loss_function
    optimizer_param = wandb.config.optimizer
    dropout_p = wandb.config.dropout
    activation = wandb.config.activation_function
    hidden_layers = wandb.config.hidden_layer_size
    encoding = None
    if wandb.config.encoding:
        encoding = {
            'sigma': 0.5,
            'frequencies': 21
        }

    checkpoint_writer: Optional[io_utils.RollingCheckpointWriter]
    if model_dir is not None:
        if model_dir.exists():
            shutil.rmtree(model_dir)
        model_dir.mkdir()
        # Write the parameters to the model folder
        config_io.write_config(config_filepath, model_dir)

        # Redirect stdout to file + stdout
        sys.stdout = io_utils.Logger(model_dir / 'log.txt', 'a')
        checkpoint_writer = io_utils.RollingCheckpointWriter(model_dir, 'checkpoint',
                                                             max_num_checkpoints, 'pth')
    else:
        print('Warning: no model name provided; not writing anything to the file system.')
        checkpoint_writer = None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print('Warning: no GPU available; training on CPU.')

    ds_loader_train = data_generation.create_data_loader(params, data_generation.PhaseType.TRAIN,
                                                         True, num_points_per_example)
    ds_loader_val = data_generation.create_data_loader(params, data_generation.PhaseType.VAL, True, -1)
    image_size = ds_loader_train.dataset.image_size \
        if isinstance(ds_loader_train.dataset, data_generation.ImplicitDataset) else None

    if not ds_loader_train:
        raise ValueError(f'Number of training examples is smaller than the batch size.')

    net = create_model(params, image_size.type(torch.DoubleTensor), dropout_p, activation, hidden_layers, encoding)
    if torch.cuda.device_count() > 1:
        print(f'Using {torch.cuda.device_count()} GPUs for training.')
        net = torch.nn.DataParallel(net)  # experimental
    net = net.to(device)
    print(net)

    if task_type == config_io.TaskType.AD:
        lr_lats = params['learning_rate_lat']
        lr_lats = wandb.config.latent_learning_rate
        latent_dim = params['latent_dim']
        num_examples_train = len(ds_loader_train.dataset)  # type: ignore[arg-type]
        # Initialization scaling follows DeepSDF
        latents_train = torch.nn.Parameter(
            torch.normal(0.0, 1 / math.sqrt(latent_dim), [num_examples_train, latent_dim],
                         device=device))
        latents_train = torch.normal(0.0, 1 / math.sqrt(128), [1, 128]).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        if optimizer_param == "adam":
            optimizer = torch.optim.Adam([
                {'params': net.parameters(), 'lr': learning_rate},
                {'params': latents_train, 'lr': lr_lats}
            ])
        elif optimizer_param == "sgd":
            optimizer = torch.optim.SGD([
                {'params': net.parameters(), 'lr': learning_rate},
                {'params': latents_train, 'lr': lr_lats}
            ])
        else:
            optimizer = torch.optim.Adam([
                {'params': net.parameters(), 'lr': learning_rate},
                {'params': latents_train, 'lr': lr_lats}
            ])
    else:
        latents_train = torch.nn.Parameter(torch.empty(0))
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    criterion = create_loss(loss_function).to(device)
    metric = nn_utils.DiceLoss(0.5, 'sum', True).to(device)
    metric_val = nn_utils.ChamferDistance()

    # This is a tensor so that it is mutable within other functions
    global_step = torch.tensor(0, dtype=torch.int64)
    num_epochs_trained = 0

    for epoch in range(num_epochs_trained, num_epochs_target):
        train_one_epoch(task_type, ds_loader_train, net, latents_train, lat_reg_lambda, optimizer,
                        criterion, metric, device, epoch, num_epochs_target, global_step,
                        log_epoch_count, wandb, True)
        if epoch % log_epoch_count == 0:
            validate(task_type, ds_loader_val, net, latents_train, metric_val, device, epoch, wandb,
                     True)

        if checkpoint_writer is not None and epoch % checkpoint_epoch_count == 0:
            checkpoint_writer.write_rolling_checkpoint(
                {'net': net.state_dict(), 'latents_train': latents_train},
                optimizer.state_dict(), int(global_step.item()), epoch + 1)

    if checkpoint_writer is not None:
        checkpoint_writer.write_rolling_checkpoint(
            {'net': net.state_dict(), 'latents_train': latents_train},
            optimizer.state_dict(), int(global_step.item()), num_epochs_target)

    wandb.finish()


if __name__ == '__main__':
    wandb.login()
    sweep_configuration = {
        "method": "grid",
        "metric": {"goal": "minimize", "name": "loss"},
        "parameters": {
            "learning_rate": {"values": [0.5e-1, 0.5e-2, 0.5e-3, 0.5e-4, 0.5e-5]},
            "latent_learning_rate": {"values": [1.0e-3]},
            "sample_batch_share": {"values": [1.0]},
            "loss_function": {"values": ["dice_bce"]},
            "optimizer": {"values": ["adam", "sgd"]},
            "dropout": {"values": [0.0, 0.5]},
            "hidden_layer_size": {"values": [128, 256]},
            "activation_function": {"values": ["sine", "relu"]},
            "encoding": {"values": [False, True]},
        },
    }
    params, config_filepath = config_io.parse_config_train()
    sweep_id = wandb.sweep(sweep=sweep_configuration, project=params['model_name'])
    wandb.agent(sweep_id, function=main, count=160)
