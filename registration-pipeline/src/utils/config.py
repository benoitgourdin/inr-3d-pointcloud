import argparse
from pathlib import Path

import yaml

# def get_model_config_path(path_config: Dict, model_name: str, config_pattern: str) -> Path:
#     """Get the path to model config from a given model directory."""
#     model_dir = path_config['model_basedir'] / model_name
#     model_config_filepath = io_utils.find_single_file(model_dir, config_pattern)
#     return model_config_filepath


def read_yaml_config(config_file_path):
    if isinstance(config_file_path, str):
        config_file_path = Path(config_file_path)
    if not config_file_path.exists():
        raise ValueError(
            f'YAML config file does not exist:\n{config_file_path}')
    params = {}
    with open(config_file_path, 'r') as f:
        params.update(yaml.safe_load(f))
    return params


def read_config(config_file_path):
    params = read_yaml_config(config_file_path)
    return params


def parse_config():
    parser = argparse.ArgumentParser(
        description='Train a neural network.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--config_file', type=str)
    args = parser.parse_args()
    if args.config_file is None:
        raise ValueError(
            'Config must be provided via a command-line argument.')
    config = read_config(args.config_file)
    config_feat = read_config(
        "/u/home/gob/repo/inr-masterthesis/registration-pipeline/src/point-m2ae.yaml"
    )
    return config, config_feat
