import argparse
import sys
import os
import socket
import importlib

import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from tbsim.utils.learning_rate import BatchLearningRateDecay
from tbsim.utils.log_utils import PrintLogger
import tbsim.utils.train_utils as TrainUtils
from tbsim.utils.env_utils import RolloutCallback
from tbsim.configs.registry import get_registered_experiment_config
from tbsim.datasets.factory import datamodule_factory
from tbsim.utils.config_utils import get_experiment_config_from_file
from tbsim.utils.batch_utils import set_global_batch_type
from tbsim.utils.trajdata_utils import set_global_trajdata_batch_env, set_global_trajdata_batch_raster_cfg
from tbsim.algos.factory import algo_factory
from tbsim.utils.planning_utils import load_goal_predictor


os.environ["WANDB_DISABLE_CODE"] = "True"
def main(cfg, auto_remove_exp_dir=False, debug=False, load_checkpoint=False, load_policy_checkpoint=False):
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

    print("\n============= New Training Run with Config =============")
    # print(cfg)
    # print("")
    root_dir, log_dir, ckpt_dir, video_dir, version_key = TrainUtils.get_exp_dir(
        exp_name=cfg.name,
        output_dir=cfg.root_dir,
        save_checkpoints=cfg.train.save.enabled,
        auto_remove_exp_dir=auto_remove_exp_dir
    )

    # Save experiment config to the training dir
    cfg.dump(os.path.join(root_dir, version_key, "config.json"))

    if cfg.train.logging.terminal_output_to_txt and not debug:
        # log stdout and stderr to a text file
        logger = PrintLogger(os.path.join(log_dir, "log.txt"))
        sys.stdout = logger
        sys.stderr = logger

    train_callbacks = []

    # Training Parallelism
    cfg.train.parallel_strategy = "ddp"
    cfg.devices.num_gpus = 4
    # # cfg.train.parallel_strategy = "auto"
    # # cfg.devices.num_gpus = 1
    assert cfg.train.parallel_strategy in [
        "ddp",
        "ddp_spawn",
        None,
        "auto"
    ]  # TODO: look into other strategies
    if not cfg.devices.num_gpus > 1:
        # Override strategy when training on a single GPU
        with cfg.train.unlocked():
            cfg.train.parallel_strategy = "auto"
    if cfg.train.parallel_strategy in ["ddp_spawn" ,"ddp"]:
        with cfg.train.training.unlocked():
            cfg.train.training.batch_size = int(
                cfg.train.training.batch_size / cfg.devices.num_gpus
            )
        with cfg.train.validation.unlocked():
            cfg.train.validation.batch_size = int(
                cfg.train.validation.batch_size / cfg.devices.num_gpus
            )

    if cfg.train.parallel_strategy == "ddp":
        cfg.train.parallel_strategy = "ddp_find_unused_parameters_true"

    # Dataset
    datamodule = datamodule_factory(
        cls_name=cfg.train.datamodule_class, config=cfg
    )
    datamodule.setup()

    # Environment for close-loop evaluation # Commented out by Dimuthu.
    if cfg.train.rollout.enabled:
        # Run rollout at regular intervals
        rollout_callback = RolloutCallback(
            exp_config=cfg,
            every_n_steps=cfg.train.rollout.every_n_steps,
            warm_start_n_steps=cfg.train.rollout.warm_start_n_steps,
            verbose=True,
            save_video=cfg.train.rollout.save_video,
            video_dir=video_dir
        )
        train_callbacks.append(rollout_callback)

    if load_checkpoint:
        policy_composers = importlib.import_module("tbsim.evaluation.policy_composers")
        composer_class = getattr(policy_composers, 'Diffuser')
        composer = composer_class(cfg, 'cuda')
        model, exp_config = composer.get_policy(policy_wrapper=False)
        model.train()
    else:
        model = algo_factory(
            config=cfg,
            modality_shapes={"image": (34, 224, 224)},
            data_cache_dir=cfg.train.trajdata_cache_location
        )

    if load_policy_checkpoint:
        # policy_composers = importlib.import_module("tbsim.evaluation.policy_composers")
        # composer_class = getattr(policy_composers, 'Diffuser')
        # composer = composer_class(cfg, 'cuda')
        # model, exp_config = composer.get_policy(policy_wrapper=False)
        # model.train()
        policy_config = get_registered_experiment_config(cfg.policy_config_name)
        spatial_planner = load_goal_predictor(
            cfg=policy_config,
            checkpoint_path=cfg.policy_ckpt
        )
        model.set_policy(spatial_planner)

    # Checkpointing
    if cfg.train.validation.enabled and cfg.train.save.save_best_validation:
        assert (
            cfg.train.save.every_n_steps > cfg.train.validation.every_n_steps
        ), "checkpointing frequency ("+str(cfg.train.save.every_n_steps)+") needs to be greater than validation frequency ("+str(cfg.train.validation.every_n_steps)+")"
        for metric_name, metric_key in model.checkpoint_monitor_keys.items():
            print(
                "Monitoring metrics {} under alias {}".format(metric_key, metric_name)
            )
            ckpt_valid_callback = pl.callbacks.ModelCheckpoint(
                dirpath=ckpt_dir,
                filename="iter{step}_ep{epoch}_%s{%s:.2f}" % (metric_name, metric_key),
                # explicitly spell out metric names, otherwise PL parses '/' in metric names to directories
                auto_insert_metric_name=False,
                save_top_k=cfg.train.save.best_k,  # save the best k models
                monitor=metric_key,
                mode="min",
                every_n_train_steps=cfg.train.save.every_n_steps,
                verbose=True,
            )
            train_callbacks.append(ckpt_valid_callback)

    if cfg.train.rollout.enabled and cfg.train.save.save_best_rollout:
        assert (
            cfg.train.save.every_n_steps > cfg.train.rollout.every_n_steps
        ), "checkpointing frequency needs to be greater than rollout frequency"
        ckpt_rollout_callback = pl.callbacks.ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="iter{step}_ep{epoch}_simADE{rollout/metrics_ego_ADE:.2f}",
            # explicitly spell out metric names, otherwise PL parses '/' in metric names to directories
            auto_insert_metric_name=False,
            save_top_k=cfg.train.save.best_k,  # save the best k models
            monitor="rollout/metrics_ego_ADE",
            mode="min",
            every_n_train_steps=cfg.train.save.every_n_steps,
            verbose=True,
        )
        train_callbacks.append(ckpt_rollout_callback)

    # a ckpt monitor to save at fixed interval
    ckpt_fixed_callback = pl.callbacks.ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="iter{step}",
        auto_insert_metric_name=False,
        save_top_k=1,
        monitor=None,
        every_n_train_steps=5000,
        verbose=True,
    )
    train_callbacks.append(ckpt_fixed_callback)

    # # NOTE: this is annoying to be able to load in EMA weights at test time,
    # #           implemented in the lightning module class instead.
    # if 'use_ema' in cfg.algo.__dict__ and cfg.algo.use_ema:
    #     from tbsim.utils.ema import EMA
    #     ema_callback = EMA(decay=cfg.algo.ema_decay, every_n_train_steps=cfg.algo.ema_step)
    #     train_callbacks.append(ema_callback)

    # Logging
    assert not (cfg.train.logging.log_tb and cfg.train.logging.log_wandb)
    logger = None
    if debug:
        print("Debugging mode, suppress logging.")
    elif cfg.train.logging.log_tb:
        logger = TensorBoardLogger(
            save_dir=root_dir, version=version_key, name=None, sub_dir="logs/"
        )
        print("Tensorboard event will be saved at {}".format(logger.log_dir))
    elif cfg.train.logging.log_wandb:
        assert (
            "WANDB_APIKEY" in os.environ
        ), "Set api key by `export WANDB_APIKEY=<your-apikey>`"
        apikey = os.environ["WANDB_APIKEY"]
        wandb.login(key=apikey)
        logger = WandbLogger(
            name=cfg.name, project=cfg.train.logging.wandb_project_name
        )
        # record the entire config on wandb
        logger.experiment.config.update(cfg.to_dict())
        logger.watch(model=model)
    else:
        print("WARNING: not logging training stats")

    # adding learning rate decay callback
    lr_decay_callback = BatchLearningRateDecay(initial_lr=cfg.algo.optim_params.policy.learning_rate.initial, final_lr=5e-7, total_steps=100000)
    train_callbacks.append(lr_decay_callback)

    # Train
    cfg.train.validation.every_n_steps = 5000
    cfg.train.validation.batch_size = 16
    trainer = pl.Trainer(
        devices=cfg.devices.num_gpus,
        accelerator="gpu",
        default_root_dir=root_dir,
        # checkpointing
        enable_checkpointing=cfg.train.save.enabled,
        # logging
        logger=logger,
        # flush_logs_every_n_steps=cfg.train.logging.flush_every_n_steps,
        log_every_n_steps=cfg.train.logging.log_every_n_steps,
        # training
        max_steps=cfg.train.training.num_steps,
        # validation
        val_check_interval=cfg.train.validation.every_n_steps,
        limit_val_batches=cfg.train.validation.num_steps_per_epoch,
        # all callbacks
        callbacks=train_callbacks,
        # device & distributed training setup
        # gpus=cfg.devices.num_gpus,
        strategy=cfg.train.parallel_strategy,
        # setting for overfit debugging
        # limit_val_batches=0,
        # overfit_batches=2,
        num_sanity_val_steps=0,
    )

    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--policy_ckpt",
        type=str,
        default=None,
        help="Directory to look for saved checkpoints"
    )

    parser.add_argument(
        "--ckpt_key",
        type=str,
        default=None,
        help="Directory to look for saved checkpoints"
    )

    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="Directory to look for saved checkpoints"
    )


    parser.add_argument(
        "--policy_config_name",
        type=str,
        default=None,
        help="The Policy config name",
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
        "--wandb_project_name",
        type=str,
        default=None,
        help="(optional) if provided, override the wandb project name defined in the config",
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
        "--remove_exp_dir",
        action="store_true",
        help="Whether to automatically remove existing experiment directory of the same name (remember to set this to "
        "True to avoid unexpected stall when launching cloud experiments).",
    )

    parser.add_argument(
        "--on_ngc",
        action="store_true",
        help="whether running the script on ngc (this will change some behaviors like avoid writing into dataset)"
    )

    parser.add_argument(
        "--load_policy_checkpoint",
        action="store_true",
        help="Whether it loads policy checkpoint for goal sampling"
    )

    parser.add_argument(
        "--load_checkpoint",
        action="store_true",
        help="Whether it loads policy checkpoint for goal sampling"
    )

    parser.add_argument(
        "--debug", action="store_true", help="Debug mode, suppress wandb logging, etc."
    )

    parser.add_argument(
        "--mllm_checkpoint",
        type=str,
        default=None,
        help="(optional) if provided, override the experiment name defined in the config",
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
        default_config.root_dir = os.path.abspath(args.output_dir)

    if args.wandb_project_name is not None:
        default_config.train.logging.wandb_project_name = args.wandb_project_name

    if args.on_ngc:
        ngc_job_id = socket.gethostname()
        default_config.name = default_config.name + "_" + ngc_job_id
    
    if args.policy_config_name:
        default_config.policy_config_name = args.policy_config_name
        default_config.policy_ckpt = args.policy_ckpt
        default_config.num_goal_samples = 1

    # This is the for loding the Diffuser Model. Not the actual Spatial Planner Policy Model
    if args.ckpt_dir is not None:
        default_config.ckpt.policy.ckpt_key = args.ckpt_key
        default_config.ckpt.policy.ckpt_dir = args.ckpt_dir
        default_config.ckpt.policy.ngc_job_id = None

    default_config.train.on_ngc = args.on_ngc

    # Add MLLM checkpoint path to config if provided
    if args.mllm_checkpoint is not None:
        default_config.algo.mllm_checkpoint = args.mllm_checkpoint

    if args.debug:
        # Test policy rollout
        default_config.train.validation.every_n_steps = 5000
        default_config.train.save.every_n_steps = 14000
        default_config.train.rollout.every_n_steps = 14000
        default_config.train.rollout.num_episodes = 1

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
    main(default_config, auto_remove_exp_dir=args.remove_exp_dir, debug=args.debug, load_checkpoint=args.load_checkpoint, load_policy_checkpoint=args.load_policy_checkpoint)