import argparse
import os
import socket
import importlib
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
from tbsim.utils.trajdata_utils import set_global_trajdata_batch_env, set_global_trajdata_batch_raster_cfg
from tbsim.utils.planning_utils import load_goal_predictor

from tbsim.utils.verify_goal_utils import draw_goal_points_on_map

class IndexedPickleWriter:
    def __init__(self, output_path):
        self.output_path = output_path
        self.index_path = output_path + ".index.pkl"

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

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


def make_goal_key(batch: Dict[str, Any], index: int) -> str:
    """
    Creates a stable key for saving goal samples.

    Prefer nuScenes identifiers:
        scene_token
        sample_token
        instance_token

    Modify this function according to your dataset batch.
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
# Goal predictor inference adapter
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


@torch.inference_mode()
def run_goal_predictor(
    goal_predictor: torch.nn.Module,
    batch: Dict[str, Any],
    num_goal_samples: int,
    data_batch
):
    """
    This is the main function you may need to adapt.

    Expected output:
        goals: Tensor [B, K, goal_dim]

    Common goal_dim options:
        2 -> final x, y
        3 -> final x, y, heading
        4+ -> extra attributes
    """

    # -----------------------------------------------------------------
    # Case 1: your model has a sample() function
    # -----------------------------------------------------------------
    if hasattr(goal_predictor, "sample"):
        goals_out = goal_predictor(
            batch,
            num_samples=1,
            mask_drivable=True,
        )

    # -----------------------------------------------------------------
    # Case 2: your model has an inference() function
    # -----------------------------------------------------------------
    elif hasattr(goal_predictor, "inference"):
        goals_out = goal_predictor(
            batch,
            num_samples=1,
            mask_drivable=True,
        )

    # -----------------------------------------------------------------
    # Case 3: your model forward() supports num_samples
    # -----------------------------------------------------------------
    else:
        goals_out = goal_predictor(
            batch,
            num_samples=1,
            mask_drivable=True,
        )

    # Some models return dictionaries.
    goals = torch.cat([goals_out['predictions']["positions"], goals_out['predictions']["yaws"]], dim=-1)

    # goal_g, goal_avail = extract_goal_targets(data_batch)
    # draw_goal_points_on_map(
    #             data_batch=data_batch,
    #             pred_goals=goals,
    #             gt_goals=goal_g,
    #             batch_idx=0,
    #             sample_idx=0,
    #             # heat_map=goals_out["goal_heatmap"]
    #         )

    # Normalize shapes.
    # Accept:
    #   [B, K, goal_dim]
    #   [K, B, goal_dim]
    #   [B, goal_dim]
    if goals.dim() == 2:
        goals = goals[:, None, :]  # [B, 1, goal_dim]

    if goals.shape[0] == num_goal_samples:
        goals = goals.permute(1, 0, 2).contiguous()  # [B, K, goal_dim]

    return goals


def return_dict(batch):
    batch = {
            'data_idx' : batch.data_idx,
            'scene_ts' : batch.scene_ts,
            'dt' : batch.dt,
            'agent_name' : batch.agent_name,
            'agent_type' : batch.agent_type,
            'curr_agent_state' : batch.curr_agent_state,
            'agent_hist' : batch.agent_hist,
            'agent_hist_extent' : batch.agent_hist_extent,
            'agent_hist_len' : batch.agent_hist_len,
            'agent_fut' : batch.agent_fut,
            'agent_fut_extent' : batch.agent_fut_extent,
            'agent_fut_len' : batch.agent_fut_len,
            'num_neigh' : batch.num_neigh,
            'neigh_indices' : batch.neigh_indices,
            'neigh_types' : batch.neigh_types,
            'neigh_hist' : batch.neigh_hist,
            'neigh_hist_extents' : batch.neigh_hist_extents,
            'neigh_hist_len' : batch.neigh_hist_len,
            'neigh_fut' : batch.neigh_fut,
            'neigh_fut_extents' : batch.neigh_fut_extents,
            'neigh_fut_len' : batch.neigh_fut_len,
            'robot_fut' : batch.robot_fut,
            'robot_fut_len' : batch.robot_fut_len,
            'map_names' : batch.map_names,
            'maps' : batch.maps,
            'maps_resolution' : batch.maps_resolution,
            'vector_maps' : batch.vector_maps,
            'rasters_from_world_tf' : batch.rasters_from_world_tf,
            'agents_from_world_tf' : batch.agents_from_world_tf,
            'scene_ids' : batch.scene_ids,
            'history_pad_dir' : batch.history_pad_dir,
            'extras' : batch.extras,
            # 'llm_input_ids': batch.llm_input_ids,
            # 'llm_attention_mask': batch.llm_attention_mask,
        }

    return batch


def precompute_rank(
    rank,
    world_size,
    local_rank,
    goal_predictor,
    dataset,
    output_dir,
    device,
    num_goal_samples=20,
    clear_cache_every = 10
):
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)

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
        batch_size=8,          # per-GPU batch size
        sampler=sampler,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
        persistent_workers=False,
        collate_fn=dataset.get_collate_fn(return_dict=False),
    )

    output_path = os.path.join(
        output_dir,
        f"goal_cache_rank_{rank:02d}_of_{world_size:02d}.pkl",
    )
    writer = IndexedPickleWriter(output_path)

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

            goals_gpu = run_goal_predictor(
                goal_predictor=goal_predictor,
                batch=batch,
                num_goal_samples=num_goal_samples,
                data_batch=batch
            )

            # Move only this batch to CPU.
            goals = goals_gpu.detach().cpu()  # [B, K, goal_dim]

            batch_size = goals.shape[0]

            for i in range(batch_size):
                key = batch['scene_ids'][i] + '_' + str(batch['scene_ts'][i].item()) + '_' + batch['agent_name'][i]

                record = {
                    "key": key,
                    "goals": goals[i],  # [K, goal_dim]
                    "goal_mean": goals[i].mean(dim=0),
                    "goal_std": goals[i].std(dim=0),
                }

                writer.write(key, record)

            del goals_gpu
            del goals

            if batch_idx % clear_cache_every == 0:
                gc.collect()
                torch.cuda.empty_cache()
    finally:
        writer.close()


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


def main(cfg):
    pl.seed_everything(cfg.seed)

    if cfg.env.name == "l5kit":
        set_global_batch_type("l5kit")
    elif cfg.env.name in ["nusc", "trajdata"]:
        set_global_batch_type("trajdata")
        if cfg.env.name == "nusc":
            set_global_trajdata_batch_env("nusc_trainval")
        elif cfg.env.name == "trajdata":
            # assumes all used trajdata datasets use share same map layers
            set_global_trajdata_batch_env(cfg.train.trajdata_source_train[0])
        set_global_trajdata_batch_raster_cfg(cfg.env.rasterizer) # determines cfg for rasterizing agents
    else:
        raise NotImplementedError("Env {} is not supported".format(cfg.env.name))

    rank, world_size, local_rank = setup_distributed()

    # Dataset
    datamodule = datamodule_factory(
        cls_name=cfg.train.datamodule_class, config=cfg
    )
    datamodule.setup()

    goal_predictor = load_goal_predictor(
        cfg=cfg,
        checkpoint_path=cfg.goal_ckpt
    )

    precompute_rank(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        goal_predictor=goal_predictor,
        dataset=datamodule.train_dataset,
        output_dir=cfg.output_dir,
        device='cuda',
        num_goal_samples=cfg.num_goal_samples
    )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--goal_ckpt",
        type=str,
        required=True,
        help="Directory to look for saved checkpoints"
    )

    # External config file that overwrites default config
    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="(optional) path to a config json that will be used to override the default settings. \
            If omitted, default settings are used. This is the preferred way to run experiments.",
    )

    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="(optional) create experiment config from a preregistered name (see configs/registry.py)",
    )
    # Experiment Name (for tensorboard, saving models, etc.)
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="(optional) if provided, override the experiment name defined in the config",
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="(optional) if provided, override the dataset root path",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Root directory of training output (checkpoints, visualization, tensorboard log, etc.)",
    )

    parser.add_argument(
        "--num_goal_samples",
        type=int,
        default=20,
        help="Number of goal samples to generate per input.",
    )

    args = parser.parse_args()

    if args.config_name is not None:
        default_config = get_registered_experiment_config(args.config_name)
        print('args.config_name', args.config_name)
        print('default_config', default_config)
    elif args.config_file is not None:
        # Update default config with external json file
        default_config = get_experiment_config_from_file(args.config_file, locked=False)
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

    # make rollout evaluation config consistent with the rest of the config
    if default_config.train.rollout.enabled:
        default_config.eval.env = default_config.env.name
        assert default_config.algo.eval_class is not None, \
            "Please set an eval_class for {}".format(default_config.algo.name)
        default_config.eval.eval_class = default_config.algo.eval_class
        default_config.eval.dataset_path = default_config.train.dataset_path
        for k in default_config.eval[default_config.eval.env]:  # copy env-specific config to the global-level
            default_config.eval[k] = default_config.eval[default_config.eval.env][k]
        default_config.eval.pop("nusc")
        default_config.eval.pop("l5kit")
        # default_config.eval.pop("trajdata")

    default_config.lock()  # Make config read-only
    main(default_config)
