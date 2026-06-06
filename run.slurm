#!/bin/bash
# Created by the University of Melbourne job script generator for SLURM
# Wed Jul 10 2024 23:21:58 GMT+1000 (Australian Eastern Standard Time)

# Partition for the job:
#SBATCH --partition=feit-gpu-a100

# Multithreaded (SMP) job: must run on one node 
#SBATCH --nodes=1

# The name of the job:
#SBATCH --job-name="ctg"

# The project ID which this job should run under:
#SBATCH --account="punim2030"

# Maximum number of tasks/CPU cores used by the job:
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12

# Number of GPUs requested per node:
#SBATCH --gres=gpu:4
# Slurm QoS:
#SBATCH --qos=feit

# The amount of memory in megabytes per node:
#SBATCH --mem=180000

# Use this email address:
#SBATCH --mail-user=dkariyawasan@student.unimelb.edu.au

# Send yourself an email when the job:
# aborts abnormally (fails)
#SBATCH --mail-type=FAIL
# begins
#SBATCH --mail-type=BEGIN
# ends successfully
#SBATCH --mail-type=END

# The maximum running time of the job in days-hours:mins:sec
#SBATCH --time=1-00:0:00

# check that the script is launched with sbatch
if [ "x$SLURM_JOB_ID" == "x" ]; then
   echo "You need to submit your job to the queuing system with sbatch"
   exit 1
fi

# Run the job from the directory where it was launched (default)

# The modules to load:
module load Anaconda3/2022.10

# The job command(s):
module load CUDA/11.8.0; eval "$(conda shell.bash hook)"; conda activate bg3.9
export PYTHONPATH="$PYTHONPATH:.";export CUDA_HOME="/apps/easybuild-2022/easybuild/software/Core/CUDA/11.8.0";export CUDA_VISIBLE_DEVICES="0,1,2,3";export MASTER_PORT="34229";export LAUNCHER="pytorch"
# python scripts/train.py --dataset_path /data/gpfs/projects/punim2030/nuscenes --config_name trajdata_nusc_diff --output_dir /data/gpfs/projects/punim2030/ctg --debug
# python scripts/train.py --dataset_path /data/gpfs/projects/punim2030/nuscenes --config_name trajdata_nusc_diff --policy_ckpt /data/gpfs/projects/punim2030/ctg/test/run51/checkpoints/iter34000.ckpt --policy_config_name trajdata_nusc_spatial_planner --debug --load_policy_checkpoint
# python scripts/scene_editor.py --results_root_dir nusc_results_dmmdiff_local_llm_no_guide_v3/ --num_scenes_per_batch 1 --dataset_path /data/gpfs/projects/punim2030/nuscenes --env trajdata --policy_ckpt_dir diffuser_trained_models/test/run48 --policy_ckpt_key iter300000.ckpt --eval_class Diffuser --editing_source config heuristic --registered_name 'trajdata_nusc_scene_diff' --render
# python scripts/scene_editor.py --results_root_dir nusc_results_4/ --num_scenes_per_batch 1 --dataset_path /data/gpfs/projects/punim2030/nuscenes --env trajdata --policy_ckpt_dir diffuser_trained_models/test/run35 --policy_ckpt_key iter350000.ckpt --eval_class Diffuser --editing_source config heuristic --registered_name 'trajdata_nusc_scene_diff' --render
# torchrun --nproc_per_node 4 scripts/precalculate_goals.py --dataset_path /data/gpfs/projects/punim2030/nuscenes --config_name trajdata_nusc_spatial_planner --output_dir /home/dkariyawasan/projects/digi-twin/CTG/goal_samples --goal_ckpt /data/gpfs/projects/punim2030/ctg/test/run51/checkpoints/iter34000.ckpt --num_goal_samples 10#
# python scripts/train.py --dataset_path /data/gpfs/projects/punim2030/nuscenes --config_name trajdata_nusc_spatial_planner --output_dir /data/gpfs/projects/punim2030/ctg --debug
# torchrun --nproc_per_node 4 scripts/train_lyft.py --dataset_path /data/gpfs/projects/punim2030/nuscenes --config_name trajdata_l5_spatial_planner --output_dir /data/gpfs/projects/punim2030/ctg/l5 --debug
# python scripts/train_lyft.py --config_name trajdata_l5_diff --policy_ckpt /data/gpfs/projects/punim2030/ctg/l5/test/run2/checkpoints/iter240000.ckpt --load_policy_checkpoint --policy_config_name trajdata_l5_spatial_planner --debug
# torchrun --nproc_per_node 4 scripts/precalculate_goals_fix.py --dataset_path /data/gpfs/projects/punim2030/nuscenes --config_name trajdata_nusc_spatial_planner --goal_ckpt /data/gpfs/projects/punim2030/ctg/test/run332/checkpoints/iter50000.ckpt --name mllm_goal_predictor --output_dir precalculated_goals/old
python scripts/train_mllm_diff.py --dataset_path /data/gpfs/projects/punim2030/nuscenes --config_name trajdata_nusc_diff --policy_ckpt /data/gpfs/projects/punim2030/ctg/test/run332/checkpoints/iter50000.ckpt --load_policy_checkpoint --policy_config_name trajdata_nusc_spatial_planner --debug

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s