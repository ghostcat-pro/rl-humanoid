# Walk to Destination Task Walkthrough

I have implemented the "Walk to Destination" task for the humanoid agent. This task requires the agent to walk to a specific target position on a flat surface.

## Changes Implemented

### 1. New Environment Asset
Created `envs/assets/humanoid_flat.xml`, which is a copy of the humanoid model but with the stairs removed and replaced by a large flat plane.

### 2. New Environment Class
Implemented `HumanoidDestinationEnv` in `envs/humanoid_destination.py`.

- **Goal:** Walk to a target position (default: x=5, y=0).
- **Rewards:**
  - Progress towards target.
  - Bonus for reaching within 0.5m.
  - Survival reward (staying upright).
  - Control cost penalty.
- **Observation:** Standard humanoid observations + relative vector to target.

### 3. Environment Registration
Registered `HumanoidDestination-v0` in `envs/__init__.py`.

### 4. Training Script
Created `train_destination.py` to train the agent using Stable-Baselines3 (PPO).

- Uses Hydra for configuration.
- Overrides the environment name to `HumanoidDestination-v0`.
- Saves outputs to `outputs_destination/` (or wherever Hydra is configured to output).

### 5. Bug Fix
Fixed a bug in `utils/make_env.py` that prevented `DummyVecEnv` (used when `n_envs=1`) from being initialized correctly.

## How to Run

To train the agent on the new task, run:

```bash
python train_destination.py
```

You can override hyperparameters using Hydra syntax:

```bash
python train_destination.py training.total_timesteps=1000000
```

To specify a custom output directory:

```bash
python train_destination.py hydra.run.dir=outputs_destination/my_run
```

## Visualization

I have generated a video of the trained agent:

**Humanoid Destination Demo**

To generate your own video from a trained model:

```bash
python evaluate_with_video.py --env_id HumanoidDestination-v0 --model_path <path_to_model.zip> --vecnorm_path <path_to_vecnorm.pkl> --video_dir videos/
```

### Example:

```bash
.venv/bin/python evaluate_with_video.py --env_id HumanoidDestination-v0 --model_path outputs/2025-11-21/01-50-52/final_model.zip --vecnorm_path outputs/2025-11-21/01-50-52/vecnormalize_final.pkl --video_dir outputs/2025-11-21/01-50-52/videos --episodes 3
```

## Verification Results

I verified the implementation by running a short training session:

```bash
python train_destination.py training.total_timesteps=2000 env.vec_env.n_envs=1 algo.hyperparams.batch_size=256 algo.hyperparams.n_steps=256 hydra.run.dir=outputs_destination/test_run_2
```

The training completed successfully, and the following files were generated in `outputs_destination/test_run_2`:

- `final_model.zip`
- `vecnormalize_final.pkl`
- `progress.csv`
- `train_destination.log`
- `eval/` directory
- `checkpoints/` directory