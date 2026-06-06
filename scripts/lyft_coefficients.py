import argparse
import os
import json
import gc

import pytorch_lightning as pl
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from tbsim.configs.registry import get_registered_experiment_config
from tbsim.datasets.factory import datamodule_factory
from tbsim.utils.config_utils import get_experiment_config_from_file
from tbsim.utils.batch_utils import set_global_batch_type
from tbsim.utils.batch_utils import batch_utils
from tbsim.utils.trajdata_utils import (
    set_global_trajdata_batch_env,
    set_global_trajdata_batch_raster_cfg,
)

# Use CTG's own function.
# Adjust this import path if your local CTG version places it elsewhere.
from tbsim.models.diffuser_helpers import convert_state_to_state_and_action


from torch.utils.data import DataLoader, Subset
import numpy as np

def make_random_subset(dataset, num_samples, seed=42):
    num_samples = min(num_samples, len(dataset))

    rng = np.random.default_rng(seed)
    indices = rng.choice(
        len(dataset),
        size=num_samples,
        replace=False,
    )

    return Subset(dataset, indices.tolist())


def return_dict(batch):
    """
    Same conversion style as your existing goal-cache script.
    """
    batch = {
        "data_idx": batch.data_idx,
        "scene_ts": batch.scene_ts,
        "dt": batch.dt,
        "agent_name": batch.agent_name,
        "agent_type": batch.agent_type,
        "curr_agent_state": batch.curr_agent_state,
        "agent_hist": batch.agent_hist,
        "agent_hist_extent": batch.agent_hist_extent,
        "agent_hist_len": batch.agent_hist_len,
        "agent_fut": batch.agent_fut,
        "agent_fut_extent": batch.agent_fut_extent,
        "agent_fut_len": batch.agent_fut_len,
        "num_neigh": batch.num_neigh,
        "neigh_indices": batch.neigh_indices,
        "neigh_types": batch.neigh_types,
        "neigh_hist": batch.neigh_hist,
        "neigh_hist_extents": batch.neigh_hist_extents,
        "neigh_hist_len": batch.neigh_hist_len,
        "neigh_fut": batch.neigh_fut,
        "neigh_fut_extents": batch.neigh_fut_extents,
        "neigh_fut_len": batch.neigh_fut_len,
        "robot_fut": batch.robot_fut,
        "robot_fut_len": batch.robot_fut_len,
        "map_names": batch.map_names,
        "maps": batch.maps,
        "maps_resolution": batch.maps_resolution,
        "vector_maps": batch.vector_maps,
        "rasters_from_world_tf": batch.rasters_from_world_tf,
        "agents_from_world_tf": batch.agents_from_world_tf,
        "scene_ids": batch.scene_ids,
        "history_pad_dir": batch.history_pad_dir,
        "extras": batch.extras,
    }

    return batch


def move_batch_to_device(batch, device):
    moved = {}

    for k, v in batch.items():
        if torch.is_tensor(v):
            moved[k] = v.to(device, non_blocking=True)
        else:
            moved[k] = v

    return moved


def get_dt(batch, default_dt=0.1):
    if "dt" not in batch:
        return default_dt

    dt = batch["dt"]

    if torch.is_tensor(dt):
        return float(dt.flatten()[0].item())

    if isinstance(dt, (list, tuple)):
        return float(dt[0])

    return float(dt)


def get_curr_speed(batch):
    """
    CTG diffuser uses current speed when converting future states to
    state-action representation.

    After batch_utils().parse_batch(batch), this is usually available
    as curr_speed.
    """
    if "curr_speed" in batch:
        curr_speed = batch["curr_speed"]
    elif "curr_agent_speed" in batch:
        curr_speed = batch["curr_agent_speed"]
    else:
        raise KeyError(
            "Could not find current speed in parsed batch. "
            "Expected 'curr_speed' or 'curr_agent_speed'. "
            "Print batch.keys() after parse_batch() to confirm the key."
        )

    if curr_speed.ndim > 1:
        curr_speed = curr_speed.squeeze(-1)

    return curr_speed


def get_target_traj(batch):
    """
    Gets future target trajectory in the format expected by CTG conversion.

    Preferred parsed keys:
        target_positions: [B, T, 2]
        target_yaws:      [B, T, 1]
        target_availabilities: [B, T] or [B, T, 1]

    Returns:
        target_traj: [B, T, 3] = [x, y, yaw]
        avail:       [B, T]
    """

    if "target_positions" not in batch or "target_yaws" not in batch:
        raise KeyError(
            "Parsed batch does not contain 'target_positions' and 'target_yaws'. "
            "This script assumes batch_utils().parse_batch(batch) produces those keys, "
            "as used by CTG diffuser training."
        )

    target_positions = batch["target_positions"]
    target_yaws = batch["target_yaws"]

    if target_yaws.ndim == 2:
        target_yaws = target_yaws.unsqueeze(-1)

    target_traj = torch.cat(
        [target_positions, target_yaws],
        dim=-1,
    )

    if "target_availabilities" in batch:
        avail = batch["target_availabilities"]
        if avail.ndim == 3:
            avail = avail[..., 0]
        avail = avail.bool()
    else:
        avail = torch.ones(
            target_positions.shape[:2],
            dtype=torch.bool,
            device=target_positions.device,
        )

    return target_traj, avail


@torch.no_grad()
def compute_diffuser_norm_info_from_dataset(
    dataset,
    collate_fn,
    output_dir,
    batch_size=32,
    num_workers=8,
    device="cuda",
    default_dt=0.1,
    use_3std=True,
):
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device(device)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cpu"),
        drop_last=False,
        persistent_workers=False,
        collate_fn=collate_fn,
    )

    total_sum = torch.zeros(6, dtype=torch.float64, device=device)
    total_sum_sq = torch.zeros(6, dtype=torch.float64, device=device)
    total_count = torch.zeros(1, dtype=torch.float64, device=device)

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Computing diffuser norm")):
        batch = return_dict(batch)
        batch = move_batch_to_device(batch, device)

        # This is the important part reused from your existing script.
        batch = batch_utils().parse_batch(batch)

        target_traj, avail = get_target_traj(batch)
        curr_speed = get_curr_speed(batch)
        dt = get_dt(batch, default_dt=default_dt)

        # CTG function.
        # Expected output: [B, T, 6]
        # [x, y, speed, yaw, acceleration, yaw_rate]
        traj_state_action = convert_state_to_state_and_action(
            target_traj,
            curr_speed,
            dt,
        )

        valid_x = traj_state_action[avail]  # [N_valid, 6]

        if valid_x.numel() > 0:
            valid_x = valid_x.to(torch.float64)

            total_sum += valid_x.sum(dim=0)
            total_sum_sq += (valid_x ** 2).sum(dim=0)
            total_count += valid_x.shape[0]

        del batch
        del target_traj
        del traj_state_action
        del valid_x

        if device.type == "cuda" and batch_idx % 50 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    count = total_count.clamp_min(1.0)

    mean = total_sum / count
    var = total_sum_sq / count - mean ** 2
    std = torch.sqrt(torch.clamp(var, min=1e-12))

    # CTG scaling:
    # x_scaled = (x_raw + add_coeffs) / div_coeffs
    # To behave like z-score:
    # x_scaled = (x_raw - mean) / std
    add = -mean
    div = 3.0 * std if use_3std else std

    result = {
        "state_action_order": [
            "x",
            "y",
            "speed",
            "yaw",
            "acceleration",
            "yaw_rate",
        ],
        "ctg_formula": "x_scaled = (x_raw + add_coeffs) / div_coeffs",
        "add_coeffs": add.detach().cpu().tolist(),
        "div_coeffs": div.detach().cpu().tolist(),
        "mean": mean.detach().cpu().tolist(),
        "std": std.detach().cpu().tolist(),
        "use_3std": use_3std,
        "num_valid_timesteps": int(total_count.item()),
        "diffuser_norm_info": [
            add.detach().cpu().tolist(),
            div.detach().cpu().tolist(),
        ],
    }

    output_path = os.path.join(output_dir, "lyft_diffuser_norm_info.json")

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print("\nSaved normalization coefficients to:")
    print(output_path)

    print("\nPaste this into DiffuserConfig:")
    print("self.lyft_norm_info = {")
    print('    "diffuser": [')
    print(f"        {tuple(result['add_coeffs'])},")
    print(f"        {tuple(result['div_coeffs'])},")
    print("    ]")
    print("}")

    return result


def main(cfg):
    pl.seed_everything(cfg.seed)

    if cfg.env.name == "l5kit":
        set_global_batch_type("l5kit")

    elif cfg.env.name in ["nusc", "trajdata"]:
        set_global_batch_type("trajdata")

        if cfg.env.name == "nusc":
            set_global_trajdata_batch_env("nusc_trainval")
        elif cfg.env.name == "trajdata":
            set_global_trajdata_batch_env(cfg.train.trajdata_source_train[0])

        set_global_trajdata_batch_raster_cfg(cfg.env.rasterizer)

    else:
        raise NotImplementedError(f"Env {cfg.env.name} is not supported")

    datamodule = datamodule_factory(
        cls_name=cfg.train.datamodule_class,
        config=cfg,
    )
    datamodule.setup()

    base_train_dataset = datamodule.train_dataset

    if cfg.norm_num_samples > 0:
        train_dataset = make_random_subset(
            base_train_dataset,
            num_samples=cfg.norm_num_samples,
            seed=cfg.seed,
        )
    else:
        train_dataset = base_train_dataset

    # Important:
    # Get collate_fn from the original dataset, not from Subset.
    collate_fn = base_train_dataset.get_collate_fn(return_dict=False)

    compute_diffuser_norm_info_from_dataset(
        dataset=train_dataset,
        collate_fn=collate_fn,
        output_dir=cfg.output_dir,
        batch_size=cfg.norm_batch_size,
        num_workers=cfg.norm_num_workers,
        device=cfg.norm_device,
        default_dt=cfg.norm_default_dt,
        use_3std=cfg.norm_use_3std,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--name",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--norm_batch_size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--norm_num_workers",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--norm_device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
    )

    parser.add_argument(
        "--norm_default_dt",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--no_3std",
        action="store_true",
        help="Use std instead of 3*std for div_coeffs.",
   )

    parser.add_argument(
        "--norm_num_samples",
        type=int,
        default=1000000,
        help="Number of random training samples used to estimate diffuser normalization.",
    )

    args = parser.parse_args()

    if args.config_name is not None:
        default_config = get_registered_experiment_config(args.config_name)

    elif args.config_file is not None:
        default_config = get_experiment_config_from_file(
            args.config_file,
            locked=False,
        )

    else:
        raise Exception(
            "Need either --config_name or --config_file to create experiment config."
        )

    if args.name is not None:
        default_config.name = args.name

    if args.dataset_path is not None:
        default_config.train.dataset_path = args.dataset_path

    default_config.output_dir = os.path.abspath(args.output_dir)
    default_config.norm_batch_size = args.norm_batch_size
    default_config.norm_num_workers = args.norm_num_workers
    default_config.norm_device = args.norm_device
    default_config.norm_default_dt = args.norm_default_dt
    default_config.norm_use_3std = not args.no_3std
    default_config.norm_num_samples = args.norm_num_samples

    default_config.lock()

    main(default_config)