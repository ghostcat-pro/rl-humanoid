# HOW TO RUN (Windows / PowerShell)

This repo’s evaluation commands were written with Linux shell quoting in mind. On Windows PowerShell, JSON must be wrapped in single quotes, and any inner double quotes must be escaped. Here’s the exact command that works for the latest HumanoidCircuit flat run:

```powershell
python scripts/evaluate/evaluate_sb3.py --env_id HumanoidCircuit-v0 `
  --model_path outputs/2025-12-12/09-48-00/eval/best_model.zip `
  --vecnorm_path outputs/2025-12-12/09-48-00/vecnormalize_final.pkl `
  --deterministic --render --episodes 3 `
  --env_kwargs '{\"waypoints\": [[10.0, 0.0], [10.0, 10.0], [0.0, 10.0], [0.0, 0.0]], \"waypoint_reach_threshold\": 1.0, \"stairs\": [], \"terrain_width\": 30.0, \"progress_reward_weight\": 200.0, \"waypoint_bonus\": 100.0, \"height_reward_weight\": 0.0, \"forward_reward_weight\": 0.5, \"ctrl_cost_weight\": 0.1, \"contact_cost_weight\": 5e-7, \"healthy_reward\": 5.0, \"terminate_when_unhealthy\": true, \"healthy_z_range\": [0.8, 3.0]}'
```

Key PowerShell notes:
- Wrap the whole JSON in single quotes.
- Escape inner double quotes with backslashes (`\"`).
- Use backticks (`) for line breaks in PowerShell (as shown above) or keep it on one line.
- Rendering can be toggled with `--render`; use `--save_video --video_folder videos` to record instead of showing the viewer.
- Activate the venv first if needed: `.\.venv\Scripts\activate`.

If you need a different env/config, adjust `--env_id`, model/vecnorm paths, and the JSON inside `--env_kwargs` accordingly.
