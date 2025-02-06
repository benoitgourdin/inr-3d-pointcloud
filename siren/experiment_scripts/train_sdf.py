'''Reproduces Sec. 4.2 in main paper and Sec. 4 in Supplement.
'''

# Enable import from parent package
import sys
import os
import wandb
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dataio, meta_modules, utils, training, loss_functions, modules

from torch.utils.data import DataLoader
import configargparse


class SDFDecoder(torch.nn.Module):

    def __init__(self, model):
        super().__init__()
        # Define the model.
        self.model = model

    def forward(self, coords):
        model_in = {'coords': coords}
        return self.model(model_in)['model_out']


def train_sdf():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run = wandb.init(settings=wandb.Settings(_service_wait=300))

    sdf_dataset = dataio.PointCloud(
        "/vol/aimspace/users/gob/Dataset/MedShapeNet/liver_sample/liver_test.xyz",
        on_surface_points=wandb.config.batch_size)
    dataloader = DataLoader(
        sdf_dataset,
        shuffle=True,
        batch_size=1,
        pin_memory=True,
        num_workers=0)
    val_dataset = dataio.PointCloud(
        "/vol/aimspace/users/gob/Dataset/MedShapeNet/liver_sample/liver_test.xyz",
        on_surface_points=30000)
    val_dataloader = DataLoader(
        val_dataset,
        shuffle=True,
        batch_size=1,
        pin_memory=True,
        num_workers=0)
    # Define the model.
    if wandb.config.activation_function == 'nerf':
        model = modules.SingleBVPNet(type='relu', mode='nerf', in_features=3)
    else:
        model = modules.SingleBVPNet(
            type=wandb.config.activation_function, in_features=3)
    model.to(device)

    # Define the loss
    loss_fn = loss_functions.sdf
    summary_fn = utils.write_sdf_summary

    root_path = os.path.join("/vol/aimspace/users/gob/Checkpoints/siren",
                             run.name)
    # utils.cond_mkdir(root_path)
    training.train(
        wandb,
        device,
        model=model,
        train_dataloader=dataloader,
        val_dataloader=val_dataloader,
        epochs=1000,
        lr=wandb.config.learning_rate,
        epochs_til_summary=2,
        epochs_til_checkpoint=100,
        model_dir=root_path,
        loss_fn=loss_fn,
        summary_fn=summary_fn,
        double_precision=False,
        clip_grad=True)


p = configargparse.ArgumentParser()
p.add(
    '-c',
    '--config_filepath',
    required=False,
    is_config_file=True,
    help='Path to config file.')

p.add_argument(
    '--logging_root', type=str, default='./logs', help='root for logging')
p.add_argument(
    '--experiment_name',
    type=str,
    required=True,
    help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.'
)

# General training options
p.add_argument('--batch_size', type=int, default=1400)
p.add_argument(
    '--lr', type=float, default=1e-4, help='learning rate. default=5e-5')
p.add_argument(
    '--num_epochs',
    type=int,
    default=10000,
    help='Number of epochs to train for.')

p.add_argument(
    '--epochs_til_ckpt',
    type=int,
    default=1,
    help='Time interval in seconds until checkpoint is saved.')
p.add_argument(
    '--steps_til_summary',
    type=int,
    default=100,
    help='Time interval in seconds until tensorboard summary is saved.')

p.add_argument(
    '--model_type',
    type=str,
    default='sine',
    help='Options are "sine" (all sine activations) and "mixed" (first layer sine, other layers tanh)'
)
p.add_argument(
    '--point_cloud_path',
    type=str,
    default='/home/sitzmann/data/point_cloud.xyz',
    help='Options are "sine" (all sine activations) and "mixed" (first layer sine, other layers tanh)'
)

p.add_argument(
    '--checkpoint_path', default=None, help='Checkpoint to trained model.')
opt = p.parse_args()

wandb.login()
sweep_configuration = {
    "method": "random",
    "metric": {
        "goal": "minimize",
        "name": "loss"
    },
    "parameters": {
        "learning_rate": {
            "values": [1e-3, 1e-4, 1e-5]
        },
        "batch_size": {
            "values": [5000, 10000, 20000, 50000]
        },
        "activation_function": {
            "values": ["relu", "sine"]
        },
    },
}
# params, _ = p.parse_config()
sweep_id = wandb.sweep(sweep=sweep_configuration, project=opt.experiment_name)
wandb.agent(sweep_id, function=train_sdf)
