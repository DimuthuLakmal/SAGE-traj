# SAGE-Traj: Semantic-Aware Goal Estimator for Trajectory Generation

SAGE-Traj is a vehicle trajectory generation framework that first predicts a high-level semantic goal and then conditions a diffusion model on that goal to generate realistic trajectories. The repository extends [CTG](https://github.com/NVlabs/CTG), which is built on [traffic-behavior-simulation (tbsim)](https://github.com/NVlabs/traffic-behavior-simulation). Its diffusion implementation also builds on [Diffuser](https://github.com/jannerm/diffuser), with optional STL-based guidance from [STLCG](https://github.com/StanfordASL/stlcg).

This README describes the intended end-to-end workflow:

1. Train the semantic goal predictor.
2. Train the diffusion trajectory generator using the trained goal predictor.
3. Run closed-loop simulation with the trained diffusion model.
4. Parse and evaluate the simulation results.

The commands below are terminal equivalents of the corresponding configurations in `.vscode/launch.json`.

---

## 1. Repository setup

### 1.1 Create the Python environment

Python 3.9 is recommended because the repository and its upstream dependencies were developed against that version.

```bash
conda create -n bg3.9 python=3.9
conda activate bg3.9
```

### 1.2 Install this repository

Run the following from the repository root:

```bash
pip install -e .
```

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

Adjust the final `cd` command if your repository directory has a different name.

### 1.4 Install the spline planner

```bash
cd ..
git clone https://github.com/NVlabs/spline-planner.git Pplan
cd Pplan
pip install -e .
cd ../SAGE-Traj
```

### 1.5 Optional STLCG installation

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

### 1.6 Additional model dependencies

The semantic module uses Hugging Face multimodal/model utilities. Ensure that the environment contains the compatible versions required by the checked-out code, including packages such as `transformers`, `accelerate`, `torch`, `torchvision`, and the remaining CTG/tbsim dependencies.

The original project may require the following legacy PyTorch stack, depending on the CUDA and upstream dependency versions used on the target machine:

```bash
pip install \
  torch==1.11.0+cu113 \
  torchvision==0.12.0+cu113 \
  torchaudio==0.11.0 \
  torchmetrics==0.11.1 \
  torchtext \
  --extra-index-url https://download.pytorch.org/whl/cu113
```

Do not install this legacy stack blindly on a newer environment. Match PyTorch to the installed CUDA driver and verify compatibility with the repository's current `transformers` and Lightning code.

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
export GOAL_OUTPUT_DIR=$PWD/goal_predictor_trained_models
export DIFF_OUTPUT_DIR=$PWD/diffuser_trained_models
export RESULTS_ROOT=$PWD/sage_traj_results
export MLLM_CKPT=/path/to/multimodal_checkpoint.ckpt
```

The launch configurations currently contain machine-specific absolute paths under `/data/gpfs/...`. Replace those paths with paths valid on your machine when using VS Code.

---

## 3. Train the semantic goal predictor

Download the pretrained multimodal checkpoint and set `MLLM_CKPT` to its path. This checkpoint is used to initialize the semantic goal predictor's multimodal encoder.

### Command

```bash
python scripts/train.py \
  --dataset_path "$NUSCENES_ROOT" \
  --config_name trajdata_nusc_spatial_planner \
  --output_dir "$GOAL_OUTPUT_DIR" \
  --mllm_checkpoint "$MLLM_CKPT" \
  --debug
```

Training creates versioned run directories. With the output path above, checkpoints will generally be stored in a structure similar to:

```text
goal_predictor_trained_models/
└── test/
    └── run0/
        ├── checkpoints/
        │   └── iterXXXXX.ckpt
        ├── logs/
        └── videos/
```

The run number and checkpoint iteration depend on existing runs and training progress. Select the desired checkpoint before starting the next stage:

```bash
export GOAL_CKPT=$GOAL_OUTPUT_DIR/test/run0/checkpoints/iter50000.ckpt
```

Verify that it exists:

```bash
test -f "$GOAL_CKPT" && echo "Goal checkpoint found: $GOAL_CKPT"
```

---

## 4. Train the diffusion model with the semantic goal predictor

The selected goal-predictor checkpoint is loaded through `--policy_ckpt`, while `--policy_config_name` tells the training script which goal-predictor configuration to reconstruct.

### Command

```bash
python scripts/train_mllm_diff.py \
  --dataset_path "$NUSCENES_ROOT" \
  --config_name trajdata_nusc_diff \
  --policy_ckpt "$GOAL_CKPT" \
  --load_policy_checkpoint \
  --policy_config_name trajdata_nusc_spatial_planner \
  --output_dir "$DIFF_OUTPUT_DIR" \
  --debug
```

Select the trained diffusion checkpoint after training:

```bash
export DIFF_RUN_DIR=$DIFF_OUTPUT_DIR/test/run0
export DIFF_CKPT_KEY=iter100000.ckpt
```

> **Checkpoint directory note:** `scene_editor.py` receives a checkpoint directory and checkpoint filename separately. Depending on how the trained run is organized, `--policy_ckpt_dir` may need to be either the run directory or its `checkpoints` subdirectory. The checked-in launch configuration passes the run directory. Follow that convention first, and confirm that the repository's checkpoint resolver finds the requested key.

---

## 5. Run closed-loop simulation

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
# Output locations
export GOAL_OUTPUT_DIR=$PWD/goal_predictor_trained_models
export DIFF_OUTPUT_DIR=$PWD/diffuser_trained_models
export RESULTS_ROOT=$PWD/sage_traj_results

# 1. Train semantic goal predictor
python scripts/train.py \
  --dataset_path "$NUSCENES_ROOT" \
  --config_name trajdata_nusc_spatial_planner \
  --output_dir "$GOAL_OUTPUT_DIR"

# Update these values after inspecting the generated run/checkpoints directory.
export GOAL_CKPT=$GOAL_OUTPUT_DIR/test/run0/checkpoints/iter50000.ckpt

# 2. Train diffusion model conditioned on the goal predictor
python scripts/train_mllm_diff.py \
  --dataset_path "$NUSCENES_ROOT" \
  --config_name trajdata_nusc_diff \
  --policy_ckpt "$GOAL_CKPT" \
  --load_policy_checkpoint \
  --policy_config_name trajdata_nusc_spatial_planner \
  --output_dir "$DIFF_OUTPUT_DIR"

# Update these values after inspecting the generated diffusion run.
export DIFF_RUN_DIR=$DIFF_OUTPUT_DIR/test/run0
export DIFF_CKPT_KEY=iter100000.ckpt

# 3. Closed-loop simulation
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

---

## 8. VS Code launch configurations

The same workflow can be run from VS Code using these configurations in order:

1. `Python Debugger: Train Goal Predictor`
2. `Python Debugger: Train DMMDiff with MLLM Goal Predictor`
3. `Python Debugger: Scene Diffuser with MLLM Goal Predictor`
4. `Python Debugger: Results`

Before launching them, update all hard-coded values in `.vscode/launch.json`, particularly:

- the Python interpreter path;
- the nuScenes dataset path;
- the semantic goal-predictor checkpoint;
- the diffusion-model run directory and checkpoint key;
- the rollout result directory;
- the ground-truth histogram path.

The terminal commands in this README are preferable for reproducible experiments because paths and checkpoints are explicit and can be recorded in shell scripts or job-submission files.

---

## 9. Configuration locations

Important configuration files include:

- `tbsim/configs/algo_config.py`: algorithm-level configuration classes;
- `tbsim/configs/registry.py`: registered experiment names, including `trajdata_nusc_spatial_planner` and `trajdata_nusc_diff`;
- `tbsim/configs/trajdata_nusc_config.py`: nuScenes/trajdata dataset settings;
- `tbsim/configs/scene_edit_config.py`: closed-loop simulation and guidance settings;
- `.vscode/launch.json`: VS Code debug configurations and example paths.

Check these files before a full run to confirm batch sizes, worker counts, GPU count, distributed strategy, validation frequency, checkpoint interval, rollout scenes, and guidance rules.

---

## 10. Reproducibility checklist

Before reporting results, record the following for each experiment:

- repository commit;
- Python, CUDA, PyTorch, Lightning, `transformers`, and `trajdata` versions;
- nuScenes version and dataset split;
- semantic goal-predictor checkpoint;
- diffusion-model checkpoint;
- registered training and rollout configuration names;
- external configuration overrides, if any;
- random seed;
- number of rollout scenes and start frames;
- enabled guidance rules;
- parser normalization and stationary-agent settings;
- ground-truth histogram file used for evaluation.

Also preserve the generated `config.json`, logs, and result statistics together with each checkpoint/result directory.

---

## 11. Common issues

### A run starts with an unexpected configuration

Confirm that the correct registered name is passed:

```text
Goal predictor: trajdata_nusc_spatial_planner
Diffusion training: trajdata_nusc_diff
Closed-loop simulation: trajdata_nusc_scene_diff
```

### A checkpoint cannot be found

Check both the run directory and its `checkpoints` subdirectory. Confirm the exact filename, including the iteration number and `.ckpt` suffix.

### Existing output directories cause confusion

Training automatically creates incrementing `runN` directories. Do not assume that the newest run is `run0`; inspect the output directory after each run.

### GPU out-of-memory errors

Reduce training/validation batch sizes, the number of concurrently simulated scenes, or the number of GPUs/workers in the relevant configuration. Rendering and loading the semantic model can add substantial memory overhead.

### Results are not found by the parser

Pass the nested rollout directory, normally:

```text
<results_root_dir>/scene_edit_eval
```

rather than only `<results_root_dir>`.

### Distributional metrics are inconsistent

Use a ground-truth histogram generated with matching preprocessing and parser settings. A histogram from a different split, normalization method, or stationary-agent rule is not directly comparable.

---

## Pretrained models

Pretrained model links are not currently included in this repository. Update this section when semantic goal-predictor and diffusion-model checkpoints are released.
