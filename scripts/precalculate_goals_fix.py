import argparse
import os
import socket
import importlib
import glob
import time
import pytorch_lightning as pl
from typing import Dict, Any, List
from tqdm import tqdm
import pickle
import gc

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from tbsim.configs.registry import get_registered_experiment_config
from tbsim.datasets.factory import datamodule_factory
from tbsim.utils.config_utils import get_experiment_config_from_file
from tbsim.utils.batch_utils import set_global_batch_type
from tbsim.utils.batch_utils import batch_utils
from tbsim.utils.trajdata_utils import (
    set_global_trajdata_batch_env,
    set_global_trajdata_batch_raster_cfg,
)
from tbsim.utils.planning_utils import load_goal_predictor

from tbsim.utils.verify_goal_utils import draw_goal_points_on_map


# ---------------------------------------------------------------------
# Indexed writer
# ---------------------------------------------------------------------

class IndexedPickleWriter:
    def __init__(self, output_path):
        self.output_path = output_path
        self.index_path = output_path + ".index.pkl"

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Important:
        # This opens a new file. Therefore, do not use the same path as old
        # partially generated cache files.
        self.f = open(output_path, "wb")
        self.index = {}
        self.num_written = 0

    def write(self, key, record):
        offset = self.f.tell()
        pickle.dump(record, self.f, protocol=pickle.HIGHEST_PROTOCOL)
        self.index[key] = offset
        self.num_written += 1

    def close(self):
        self.f.flush()
        self.f.close()

        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"Saved {self.output_path}")
        print(f"Saved {self.index_path}")
        print(f"Total records: {self.num_written}")


# ---------------------------------------------------------------------
# Load existing cache without relying on index files
# ---------------------------------------------------------------------

def read_pickle_records_without_index(path):
    """
    Reads a pickle file written by repeated pickle.dump(record, f).

    This does not require the .index.pkl file.

    If the last record is incomplete because the script crashed while writing,
    this will keep all valid records before the corrupted tail.
    """
    records = []

    with open(path, "rb") as f:
        while True:
            try:
                record = pickle.load(f)
                records.append(record)
            except EOFError:
                break
            except Exception as e:
                print(f"[WARN] Could not fully read {path}. Stopped at partial/corrupted tail.")
                print(f"[WARN] Error: {repr(e)}")
                break

    return records


def load_existing_goal_keys(output_dir):
    """
    Loads keys from all existing goal cache pickle files in output_dir.

    It ignores index files completely.

    Returns:
        existing_keys: set[str]
    """
    existing_keys = set()
    num_records = 0
    num_duplicates = 0

    cache_paths = sorted(glob.glob(os.path.join(output_dir, "goal_cache*.pkl")))

    for path in cache_paths:
        if path.endswith(".index.pkl"):
            continue

        print(f"[CACHE] Reading existing cache file: {path}")
        records = read_pickle_records_without_index(path)

        for record in records:
            key = record.get("key", None)
            if key is None:
                continue

            num_records += 1

            if key in existing_keys:
                num_duplicates += 1
            else:
                existing_keys.add(key)

    print("=" * 80)
    print(f"[CACHE] Existing unique keys loaded: {len(existing_keys)}")
    print(f"[CACHE] Total valid records found:      {num_records}")
    print(f"[CACHE] Duplicate records found:       {num_duplicates}")
    print("=" * 80)

    return existing_keys


# ---------------------------------------------------------------------
# Key creation
# ---------------------------------------------------------------------

def make_goal_key_from_batch(batch: Dict[str, Any], index: int) -> str:
    """
    Keep this exactly consistent with the key used during writing.

    Current key format:
        scene_id_scene_ts_agent_name
    """
    key = (
        batch["scene_ids"][index]
        + "_"
        + str(batch["scene_ts"][index].item())
        + "_"
        + batch["agent_name"][index]
    )
    return key


def make_goal_key(batch: Dict[str, Any], index: int) -> str:
    """
    Original generic key function.

    This is kept here, but the script currently uses make_goal_key_from_batch()
    because your already generated files used:
        scene_ids + scene_ts + agent_name
    """

    def get_value(name: str):
        value = batch.get(name, None)

        if value is None:
            return "none"

        if isinstance(value, list):
            return str(value[index])

        if isinstance(value, tuple):
            return str(value[index])

        if torch.is_tensor(value):
            return str(value[index].item())

        return str(value)

    scene_token = get_value("scene_token")
    sample_token = get_value("sample_token")
    instance_token = get_value("instance_token")

    return f"{scene_token}/{sample_token}/{instance_token}"


# ---------------------------------------------------------------------
# Move tensors to device
# ---------------------------------------------------------------------

def move_batch_to_device(batch: Dict[str, Any], device: torch.device):
    moved = {}

    for k, v in batch.items():
        if torch.is_tensor(v):
            moved[k] = v.to(device, non_blocking=True)
        else:
            moved[k] = v

    return moved


# ---------------------------------------------------------------------
# Goal target extraction
# ---------------------------------------------------------------------

def extract_goal_targets(data_batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        goals: [B, K, 3]
        avail: [B, K]

    Expected batch fields:
        target_positions:      [B, T, 2]
        target_yaws:           [B, T, 1]
        target_availabilities: [B, T]

    Optional:
        goal_indices: [K]
    """

    pos = data_batch["target_positions"]
    yaw = data_batch["target_yaws"]
    avail = data_batch["target_availabilities"]

    B, T, _ = pos.shape
    device = pos.device

    if "goal_indices" in data_batch:
        goal_indices = data_batch["goal_indices"].to(device).long()
    else:
        goal_indices = torch.linspace(
            0,
            T - 1,
            4,
            device=device,
        ).round().long()

    goal_indices = torch.clamp(goal_indices, 0, T - 1)

    goal_pos = pos[:, goal_indices, :]
    goal_yaw = yaw[:, goal_indices, :]
    goal_avail = avail[:, goal_indices]

    goals = torch.cat([goal_pos, goal_yaw], dim=-1)

    yaw = goals[..., 2]
    goals[..., 2] = torch.atan2(torch.sin(yaw), torch.cos(yaw))

    return goals, goal_avail


# ---------------------------------------------------------------------
# Goal predictor inference adapter
# ---------------------------------------------------------------------

@torch.inference_mode()
def run_goal_predictor(
    goal_predictor: torch.nn.Module,
    batch: Dict[str, Any],
    num_goal_samples: int,
    data_batch,
):
    """
    Expected output:
        goals: Tensor [B, K, goal_dim]

    Common goal_dim options:
        2 -> final x, y
        3 -> final x, y, heading
    """

    if hasattr(goal_predictor, "sample"):
        goals_out = goal_predictor(
            batch,
            num_samples=1,
            mask_drivable=True,
        )
    elif hasattr(goal_predictor, "inference"):
        goals_out = goal_predictor(
            batch,
            num_samples=1,
            mask_drivable=True,
        )
    else:
        goals_out = goal_predictor(
            batch,
            num_samples=1,
            mask_drivable=True,
        )

    goals = torch.cat(
        [
            goals_out["predictions"]["positions"],
            goals_out["predictions"]["yaws"],
        ],
        dim=-1,
    )

    # Optional debugging visualization.
    # goal_g, goal_avail = extract_goal_targets(data_batch)
    # draw_goal_points_on_map(
    #     data_batch=data_batch,
    #     pred_goals=goals,
    #     gt_goals=goal_g,
    #     batch_idx=0,
    #     sample_idx=0,
    # )

    if goals.dim() == 2:
        goals = goals[:, None, :]  # [B, 1, goal_dim]

    if goals.shape[0] == num_goal_samples:
        goals = goals.permute(1, 0, 2).contiguous()  # [B, K, goal_dim]

    return goals


# ---------------------------------------------------------------------
# Convert trajdata batch object to dict
# ---------------------------------------------------------------------

def return_dict(batch):
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
        # "llm_input_ids": batch.llm_input_ids,
        # "llm_attention_mask": batch.llm_attention_mask,
    }

    return batch


# ---------------------------------------------------------------------
# Per-rank precomputation
# ---------------------------------------------------------------------

def precompute_rank(
    rank,
    world_size,
    local_rank,
    goal_predictor,
    dataset,
    output_dir,
    device,
    existing_keys,
    num_goal_samples=20,
    clear_cache_every=10,
    batch_size=8,
    num_workers=8,
):
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    goal_predictor.to(device)
    goal_predictor.eval()

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=False,
        collate_fn=dataset.get_collate_fn(return_dict=False),
    )

    # Important:
    # Use a new filename for resumed run.
    # This prevents overwriting the old partially generated cache files.
    resume_tag = int(time.time())

    output_path = os.path.join(
        output_dir,
        f"goal_cache_resume_{resume_tag}_rank_{rank:02d}_of_{world_size:02d}.pkl",
    )

    writer = IndexedPickleWriter(output_path)

    num_skipped_existing = 0
    num_computed = 0

    try:
        iterator = tqdm(
            dataloader,
            desc=f"Rank {rank}",
            disable=(rank != 0),
        )

        for batch_idx, batch in enumerate(iterator):
            batch = return_dict(batch)
            batch = move_batch_to_device(batch, device)
            batch = batch_utils().parse_batch(batch)

            batch_keys = [
                make_goal_key_from_batch(batch, i)
                for i in range(len(batch["agent_name"]))
            ]

            # If all keys in this batch already exist, skip the full forward pass.
            if all(key in existing_keys for key in batch_keys):
                num_skipped_existing += len(batch_keys)

                if rank == 0 and batch_idx % 100 == 0:
                    iterator.set_postfix(
                        written=writer.num_written,
                        skipped=num_skipped_existing,
                    )

                continue

            goals_gpu = run_goal_predictor(
                goal_predictor=goal_predictor,
                batch=batch,
                num_goal_samples=num_goal_samples,
                data_batch=batch,
            )

            goals = goals_gpu.detach().cpu()
            batch_size_actual = goals.shape[0]

            for i in range(batch_size_actual):
                key = batch_keys[i]

                if key in existing_keys:
                    num_skipped_existing += 1
                    continue

                record = {
                    "key": key,
                    "goals": goals[i],
                    "goal_mean": goals[i].mean(dim=0),
                    "goal_std": goals[i].std(dim=0),
                }

                writer.write(key, record)

                # Update local set so duplicated keys within the same resumed run
                # are also skipped.
                existing_keys.add(key)

                num_computed += 1

            if rank == 0 and batch_idx % 100 == 0:
                iterator.set_postfix(
                    written=writer.num_written,
                    skipped=num_skipped_existing,
                )

            del goals_gpu
            del goals

            if batch_idx % clear_cache_every == 0:
                gc.collect()
                torch.cuda.empty_cache()

    finally:
        writer.close()

        print(
            f"[Rank {rank}] Done. "
            f"New records written: {writer.num_written}. "
            f"Skipped existing: {num_skipped_existing}."
        )


# ---------------------------------------------------------------------
# Distributed setup
# ---------------------------------------------------------------------

def setup_distributed():
    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


def cleanup_distributed():
    dist.barrier()
    dist.destroy_process_group()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

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

    rank, world_size, local_rank = setup_distributed()

    # Load existing keys from previous partial cache files.
    # Do this on every rank. This is simple and avoids synchronization complexity.
    existing_keys = load_existing_goal_keys(cfg.output_dir)

    # Dataset
    datamodule = datamodule_factory(
        cls_name=cfg.train.datamodule_class,
        config=cfg,
    )
    datamodule.setup()

    goal_predictor = load_goal_predictor(
        cfg=cfg,
        checkpoint_path=cfg.goal_ckpt,
    )

    precompute_rank(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        goal_predictor=goal_predictor,
        dataset=datamodule.train_dataset,
        output_dir=cfg.output_dir,
        device="cuda",
        existing_keys=existing_keys,
        num_goal_samples=cfg.num_goal_samples,
        batch_size=cfg.precompute_batch_size,
        num_workers=cfg.precompute_num_workers,
    )

    cleanup_distributed()


# ---------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--goal_ckpt",
        type=str,
        required=True,
        help="Directory to look for saved checkpoints",
    )

    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help=(
            "Optional path to a config json that overwrites the default settings. "
            "If omitted, default settings are used."
        ),
    )

    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Optional registered config name.",
    )

    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Optional experiment name override.",
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Optional dataset root path override.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory where goal cache pickle files are saved.",
    )

    parser.add_argument(
        "--num_goal_samples",
        type=int,
        default=20,
        help="Number of goal samples to generate per input.",
    )

    parser.add_argument(
        "--precompute_batch_size",
        type=int,
        default=8,
        help="Per-GPU batch size for goal precomputation.",
    )

    parser.add_argument(
        "--precompute_num_workers",
        type=int,
        default=8,
        help="Number of dataloader workers per rank.",
    )

    args = parser.parse_args()

    if args.config_name is not None:
        default_config = get_registered_experiment_config(args.config_name)
        print("args.config_name", args.config_name)
        print("default_config", default_config)

    elif args.config_file is not None:
        default_config = get_experiment_config_from_file(
            args.config_file,
            locked=False,
        )

    else:
        raise Exception(
            "Need either a config name or a json file to create experiment config"
        )

    if args.name is not None:
        default_config.name = args.name

    if args.dataset_path is not None:
        default_config.train.dataset_path = args.dataset_path

    if args.output_dir is not None:
        default_config.output_dir = os.path.abspath(args.output_dir)

    default_config.num_goal_samples = args.num_goal_samples
    default_config.goal_ckpt = args.goal_ckpt
    default_config.precompute_batch_size = args.precompute_batch_size
    default_config.precompute_num_workers = args.precompute_num_workers

    # Make rollout evaluation config consistent with the rest of the config.
    if default_config.train.rollout.enabled:
        default_config.eval.env = default_config.env.name

        assert default_config.algo.eval_class is not None, (
            f"Please set an eval_class for {default_config.algo.name}"
        )

        default_config.eval.eval_class = default_config.algo.eval_class
        default_config.eval.dataset_path = default_config.train.dataset_path

        for k in default_config.eval[default_config.eval.env]:
            default_config.eval[k] = default_config.eval[default_config.eval.env][k]

        default_config.eval.pop("nusc")
        default_config.eval.pop("l5kit")
        # default_config.eval.pop("trajdata")

    default_config.lock()
    main(default_config)