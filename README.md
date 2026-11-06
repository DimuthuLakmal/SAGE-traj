# SAGE-Traj: Semantic-Aware Goal Estimator for Trajectory Generation

SAGE-Traj is a vehicle trajectory generation framework that first predicts a high-level semantic goal and then conditions a diffusion model on that goal to generate realistic trajectories. The repository extends [CTG](https://github.com/NVlabs/CTG), which is built on [traffic-behavior-simulation (tbsim)](https://github.com/NVlabs/traffic-behavior-simulation). Its diffusion implementation also builds on [Diffuser](https://github.com/jannerm/diffuser), with optional STL-based guidance from [STLCG](https://github.com/StanfordASL/stlcg).

## Architecture: goal inference service

The semantic goal predictor (a multimodal LLM with a goal head) is **not** part of this repository's model. It runs as a separate **goal inference service** that SAGE-Traj queries over gRPC. For every sample, SAGE-Traj renders the map frames and the coordinate table, sends them to the service, and conditions the diffusion model on the returned goal.

- **Goal predictor service code:** `(https://github.com/DimuthuLakmal/goal_predictor_vllm.git)`

The goal predictor is trained, and its checkpoint is served, from that repository. It has its own Python environment. This repository only contains the gRPC client (`tbsim/utils/goal_service/`).

> **The goal inference service must be running before you start diffusion training or closed-loop simulation.** Both stages query it for every batch and wait for it to come up, up to a timeout.

This README describes the end-to-end workflow:

1. Start the goal inference service.
2. Train the diffusion trajectory generator, conditioned on goals from the service, or download our pretrained checkpoint (Section 4.1).
3. Run closed-loop simulation with the trained diffusion model, which also uses the service.
4. Parse and evaluate the simulation results.

The commands below are terminal equivalents of the corresponding configurations in `.vscode/launch.json`.

---

## 1. Repository setup

### 1.1 Create the Python environment

Python 3.9 is required because the repository and its upstream dependencies were developed against that version.

```bash
conda create -n bg3.9 python=3.9
conda activate bg3.9
```

### 1.2 Install PyTorch

The tested environment uses PyTorch 2.2.2 built for CUDA 11.8:

```bash
pip install \
  torch==2.2.2+cu118 \
  torchvision==0.17.2+cu118 \
  torchaudio==2.2.2+cu118 \
  --extra-index-url https://download.pytorch.org/whl/cu118
```

If your machine needs a different CUDA build, install the matching PyTorch 2.2.x wheels instead.

### 1.3 Install the customized `trajdata` dependency

The project expects the customized `trajdata` implementation used by CTG.

```bash
cd ..
git clone https://github.com/AIasd/trajdata.git
cd trajdata
pip install -r trajdata_requirements.txt
pip install -e .
cd ../SAGE-Traj
```

Adjust the final `cd` command if your repository directory has a different name. The same applies to the steps below.

### 1.4 Install l5kit

`tbsim` imports `l5kit` utilities, including on the nuScenes simulation path. l5kit declares old pins (`torch<2`, `numpy~=1.19`) that conflict with this environment, so install it without dependencies. Its runtime dependencies are covered by `requirements.txt`.

```bash
cd ..
git clone https://github.com/woven-planet/l5kit.git
cd l5kit/l5kit
pip install --no-deps -e .
cd ../../SAGE-Traj
```

### 1.5 Install the spline planner

```bash
cd ..
git clone https://github.com/NVlabs/spline-planner.git Pplan
cd Pplan
pip install -e .
cd ../SAGE-Traj
```

### 1.6 Optional STLCG installation

Install this dependency when using STL/rule-based guidance.

```bash
cd ..
git clone https://github.com/StanfordASL/stlcg.git
cd stlcg
git checkout dev
pip install graphviz
pip install -e .
cd ../SAGE-Traj
```

### 1.7 Install the remaining dependencies and this repository

`requirements.txt` was exported from the tested `bg3.9` environment. It leaves out the packages installed in the steps above. Install it **last**, because it intentionally overrides a few older pins from `trajdata_requirements.txt` (`pandas`, `tqdm`, `dill`, `pyarrow`). The header of the file lists what was left out and why.

Run the following from the repository root:

```bash
pip install -r requirements.txt
pip install -e .
```

### 1.8 Optional: flash-attn

`flash-attn` is only used by the legacy in-repo InternVL modules (`tbsim/models/internvl`), and most of those imports are optional. It needs a CUDA toolchain to build, so it is not in `requirements.txt`. The tested environment uses:

```bash
pip install flash-attn==2.6.0.post1 --no-build-isolation
```

---

## 2. Prepare nuScenes

Download nuScenes, including the v1.3 map expansion, and arrange it as follows:

```text
nuscenes/
├── maps/
├── v1.0-mini/       # optional for development
└── v1.0-trainval/
```

All commands in this README use the following environment variable. Replace the path with the location of your dataset:

```bash
export NUSCENES_ROOT=/path/to/nuscenes
```

Run all subsequent commands from the SAGE-Traj repository root.

You may also define convenient output locations:

```bash
export DIFF_OUTPUT_DIR=$PWD/diffuser_trained_models
export RESULTS_ROOT=$PWD/sage_traj_results
```

The launch configurations currently contain machine-specific absolute paths under `/data/gpfs/...`. Replace those paths with paths valid on your machine when using VS Code.

---

## 3. Start the goal inference service

Set up, and if needed train, the goal predictor by following the instructions in the goal predictor service repository: `(https://github.com/DimuthuLakmal/goal_predictor_vllm.git)`.

### 3.1 Launch the service

Start the service in **its own environment** (not `bg3.9`), in a separate terminal, tmux session, or background job, and leave it running. With the reference implementation, run this from the service repository:

```bash
# In the goal predictor service repository and environment
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false \
  python -m internvl.inference.serve --config internvl/inference/configs/goal_predictor.yaml
```

See the service repository for the exact command, configuration file, and checkpoint path. By default the service listens on port `50051`.

When the service and SAGE-Traj share a machine, give them different GPUs. For example, put the service on one GPU and train on the others with `CUDA_VISIBLE_DEVICES`.

### 3.2 Point SAGE-Traj at the service

SAGE-Traj connects to `localhost:50051` by default (`goal_service_address` in `tbsim/configs/algo_config.py`). If the service runs on another host or port, set:

```bash
export GOAL_SERVICE_ADDR=<host>:<port>   # e.g. gpu-node-12:50051
```

`GOAL_SERVICE_ADDR` overrides the configured address. The per-request timeout is `goal_service_timeout` (default 120 s).

### 3.3 Check that the service is ready

From the `bg3.9` environment, in the SAGE-Traj repository root:

```bash
python -c "
from tbsim.utils.goal_service.client import get_client
print('goal service ready:', get_client(timeout_seconds=1800).model_version)
"
```

This blocks until the service answers its health check, up to the given timeout. Loading the model can take several minutes. Continue only once it prints the model version.

### 3.4 SLURM example

`scripts/train_goal_service.slurm` runs both processes in one job: it starts the service on one GPU, waits for the health check, and then trains on the remaining GPUs. The VS Code compound launch configuration `Goal Service: Server + Train Diffuser` does the same. Adjust the paths and environment names in both to match your machine.

---

## 4. Train the diffusion model

Make sure the goal inference service is running and reachable (Section 3). The diffusion model gets its goal condition from the service during training, so there is no goal-predictor checkpoint to pass here.

### Command

```bash
python scripts/train.py \
  --dataset_path "$NUSCENES_ROOT" \
  --config_name trajdata_nusc_diff \
  --output_dir "$DIFF_OUTPUT_DIR" \
  --debug
```

For multi-GPU training, set `CUDA_VISIBLE_DEVICES` and `NUM_GPUS`, and keep the service's GPU out of `CUDA_VISIBLE_DEVICES`.

To resume from a saved diffusion checkpoint, add:

```bash
--ckpt_dir "$DIFF_OUTPUT_DIR/test/run0" --ckpt_key iter60000.ckpt --load_checkpoint
```

Training creates versioned run directories, similar to:

```text
diffuser_trained_models/
└── test/
    └── run0/
        ├── checkpoints/
        │   └── iterXXXXX.ckpt
        ├── logs/
        └── videos/
```

Select the trained diffusion checkpoint after training:

```bash
export DIFF_RUN_DIR=$DIFF_OUTPUT_DIR/test/run0
export DIFF_CKPT_KEY=iter100000.ckpt
```

> **Checkpoint directory note:** `scene_editor.py` receives a checkpoint directory (`--policy_ckpt_dir`) and a checkpoint key (`--policy_ckpt_key`) separately. Pass the **run directory**. The checkpoint resolver searches it recursively for exactly one `.ckpt` file whose name contains the key, and for the run's `config.json`. If the key matches more than one file (e.g. `iter8000` also matches `iter80000...`), loading fails, so use the full filename.

### 4.1 Skip training: use the pretrained SAGE-Traj checkpoint

To run closed-loop simulation without training, download our trained diffusion checkpoint:

- **Pretrained SAGE-Traj checkpoint:** `https://drive.google.com/drive/folders/1YIUcZAsxfRCGJ0b7N7gRaffcRRLANzUb?usp=sharing`

The download contains the training `config.json` and the checkpoint file. Keep them in this layout:

```text
pretrained/sage_traj/
├── config.json
└── checkpoints/
    └── <CHECKPOINT_FILENAME>.ckpt
```

Then point the simulation at it instead of a run you trained:

```bash
export DIFF_RUN_DIR=$PWD/pretrained/sage_traj
export DIFF_CKPT_KEY=<CHECKPOINT_FILENAME>.ckpt
```

and continue with Section 5. You still need the goal inference service from Section 3: the pretrained diffusion model is conditioned on its goals and queries it during simulation.

---

## 5. Run closed-loop simulation

The goal inference service must also be running during closed-loop simulation (Section 3), because the diffusion policy queries it for goals at every replanning step.

### Command

```bash
python scripts/scene_editor.py \
  --results_root_dir "$RESULTS_ROOT" \
  --num_scenes_per_batch 1 \
  --dataset_path "$NUSCENES_ROOT" \
  --env trajdata \
  --policy_ckpt_dir "$DIFF_RUN_DIR" \
  --policy_ckpt_key "$DIFF_CKPT_KEY" \
  --eval_class Diffuser \
  --editing_source config heuristic \
  --registered_name trajdata_nusc_scene_diff \
  --render
```

The rollout configuration appends `scene_edit_eval` to the supplied result root. Therefore, the evaluation files will normally be written to:

```text
$RESULTS_ROOT/scene_edit_eval/
```

This directory may contain files such as the rollout configuration, statistics, rendered videos, and trajectory data.

### Guidance and rendering options

- `--editing_source config heuristic` enables both configured and heuristic edits/guidance.
- Use `--editing_source none` for an unguided/no-rule rollout, when supported by the selected configuration.
- Remove `--render` when videos are unnecessary; this reduces evaluation time and storage use.
- `--num_scenes_per_batch 1` is the safest setting. Increase it only after checking GPU memory consumption and environment compatibility.
- Use a distinct `--results_root_dir` for each experiment to avoid mixing results from different checkpoints or guidance settings.

Example unguided run:

```bash
python scripts/scene_editor.py \
  --results_root_dir "${RESULTS_ROOT}_no_rules" \
  --num_scenes_per_batch 1 \
  --dataset_path "$NUSCENES_ROOT" \
  --env trajdata \
  --policy_ckpt_dir "$DIFF_RUN_DIR" \
  --policy_ckpt_key "$DIFF_CKPT_KEY" \
  --eval_class Diffuser \
  --editing_source none \
  --registered_name trajdata_nusc_scene_diff
```

---

## 6. Parse and evaluate the closed-loop results

Set the rollout result directory and ground-truth histogram path:

```bash
export ROLLOUT_DIR=$RESULTS_ROOT/scene_edit_eval
export GT_HIST=$PWD/gt_files/trajdata_nusc_new/GroundTruth/hist_stats_new_scene_current_speed.json
```

### Command

```bash
python scripts/parse_scene_edit_results.py \
  --results_dir "$ROLLOUT_DIR" \
  --eval_out_dir "$ROLLOUT_DIR" \
  --gt_hist "$GT_HIST"
```

The parser estimates distributional distances by default in the current implementation. The ground-truth histogram is required for the corresponding EMD/Wasserstein-style comparisons.

To save parsed histogram data as part of the evaluation, add:

```bash
--save_hist_data
```

Complete example:

```bash
python scripts/parse_scene_edit_results.py \
  --results_dir "$ROLLOUT_DIR" \
  --eval_out_dir "$ROLLOUT_DIR" \
  --gt_hist "$GT_HIST" \
  --save_hist_data
```

Ensure that the selected ground-truth histogram uses the same dataset split, preprocessing, normalization, scene selection, and stationary-agent policy as the rollout being evaluated. The default parser settings are:

```text
normalization_by = scene
disable_control_on_stationary = current_speed
```

These can be changed explicitly when needed:

```bash
python scripts/parse_scene_edit_results.py \
  --results_dir "$ROLLOUT_DIR" \
  --eval_out_dir "$ROLLOUT_DIR" \
  --gt_hist "$GT_HIST" \
  --normalization_by scene \
  --disable_control_on_stationary current_speed
```

---

## 7. Complete end-to-end command sequence

After setting `NUSCENES_ROOT`, the complete workflow is:

```bash
# 0. In a separate terminal, using the goal predictor service repository and its environment,
#    start the goal inference service and leave it running (see Section 3):
#      (https://github.com/DimuthuLakmal/goal_predictor_vllm.git)
#    If it is not on localhost:50051:
# export GOAL_SERVICE_ADDR=<host>:<port>

conda activate bg3.9

# Output locations
export DIFF_OUTPUT_DIR=$PWD/diffuser_trained_models
export RESULTS_ROOT=$PWD/sage_traj_results

# 1. Wait until the goal inference service is ready
python -c "
from tbsim.utils.goal_service.client import get_client
print('goal service ready:', get_client(timeout_seconds=1800).model_version)
"

# 2. Train the diffusion model (queries the goal service)
python scripts/train.py \
  --dataset_path "$NUSCENES_ROOT" \
  --config_name trajdata_nusc_diff \
  --output_dir "$DIFF_OUTPUT_DIR"

# Update these values after inspecting the generated diffusion run.
export DIFF_RUN_DIR=$DIFF_OUTPUT_DIR/test/run0
export DIFF_CKPT_KEY=iter100000.ckpt
# Or skip step 2 and use the pretrained checkpoint (Section 4.1):
# export DIFF_RUN_DIR=$PWD/pretrained/sage_traj
# export DIFF_CKPT_KEY=<CHECKPOINT_FILENAME>.ckpt

# 3. Closed-loop simulation (the goal service must still be running)
python scripts/scene_editor.py \
  --results_root_dir "$RESULTS_ROOT" \
  --num_scenes_per_batch 1 \
  --dataset_path "$NUSCENES_ROOT" \
  --env trajdata \
  --policy_ckpt_dir "$DIFF_RUN_DIR" \
  --policy_ckpt_key "$DIFF_CKPT_KEY" \
  --eval_class Diffuser \
  --editing_source config heuristic \
  --registered_name trajdata_nusc_scene_diff \
  --render

# 4. Parse results
export ROLLOUT_DIR=$RESULTS_ROOT/scene_edit_eval
export GT_HIST=$PWD/gt_files/trajdata_nusc_new/GroundTruth/hist_stats_new_scene_current_speed.json

python scripts/parse_scene_edit_results.py \
  --results_dir "$ROLLOUT_DIR" \
  --eval_out_dir "$ROLLOUT_DIR" \
  --gt_hist "$GT_HIST"
```
