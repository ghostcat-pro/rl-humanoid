# S2 Ablation After-Action Report

## Evaluation Protocol

```json
{
  "generated_utc": "2026-09-30T08:14:42.044595+00:00",
  "manifest": "experiments/robot2026_camera_ready/s2_ablation_manifest.csv",
  "episodes": 100,
  "eval_seed": 123,
  "deterministic": true,
  "completed_runs": 15,
  "missing_runs": 0
}
```

Success criterion: `max_step_reached >= num_steps` from each run config.

## Arm Summary

| arm | seeds_completed | seeds | mean_success_rate | sd_success_rate | mean_reward_across_seeds | sd_reward_across_seeds | mean_length_across_seeds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| s2_base | 5 | 42 43 44 45 46 | 28.80 | 40.47 | 14665.08 | 731.23 | 945.57 |
| s2_no_shaping | 5 | 42 43 44 45 46 | 81.20 | 8.35 | 14495.39 | 912.18 | 913.07 |
| s2_world_frame_termination | 5 | 42 43 44 45 46 | 35.00 | 32.27 | 12806.65 | 2198.60 | 821.97 |

## Seed Summary

| arm | seed | episodes | success_rate | mean_reward | std_reward | mean_length | mean_max_step_reached | mean_final_x |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| s2_base | 42 | 100 | 92.00 | 15338.36 | 3121.65 | 936.44 | 7.60 | 12.21 |
| s2_base | 43 | 100 | 47.00 | 14978.70 | 2386.48 | 957.48 | 7.07 | 7.21 |
| s2_base | 44 | 100 | 0.00 | 14758.87 | 1722.52 | 983.76 | 1.98 | 2.85 |
| s2_base | 45 | 100 | 0.00 | 13419.62 | 2794.62 | 892.60 | 1.57 | 2.97 |
| s2_base | 46 | 100 | 5.00 | 14829.85 | 2136.95 | 957.55 | 6.05 | 6.40 |
| s2_no_shaping | 42 | 100 | 83.00 | 13443.25 | 5362.35 | 849.31 | 6.71 | 8.06 |
| s2_no_shaping | 43 | 100 | 94.00 | 15436.61 | 3117.02 | 952.11 | 7.56 | 11.61 |
| s2_no_shaping | 44 | 100 | 76.00 | 13900.53 | 3319.70 | 885.03 | 7.17 | 8.02 |
| s2_no_shaping | 45 | 100 | 72.00 | 15453.44 | 876.87 | 990.45 | 7.66 | 7.83 |
| s2_no_shaping | 46 | 100 | 81.00 | 14243.10 | 3223.13 | 888.45 | 7.46 | 9.50 |
| s2_world_frame_termination | 42 | 100 | 13.00 | 13791.30 | 3525.51 | 889.06 | 5.99 | 6.27 |
| s2_world_frame_termination | 43 | 100 | 0.00 | 14941.90 | 27.52 | 1000.00 | 1.00 | 2.61 |
| s2_world_frame_termination | 44 | 100 | 25.00 | 9782.94 | 2464.63 | 610.60 | 5.84 | 6.34 |
| s2_world_frame_termination | 45 | 100 | 76.00 | 14284.83 | 1737.53 | 907.94 | 7.59 | 7.59 |
| s2_world_frame_termination | 46 | 100 | 61.00 | 11232.27 | 3425.86 | 702.23 | 6.75 | 7.27 |

## Missing Runs

_None._

## Conclusions

- All 15 pre-registered runs completed and were evaluated with 100 deterministic
  episodes on evaluation seed 123.
- The original S2 seed 42 result remains strong at 92% success, but the four
  additional `s2_base` seeds did not reproduce that level. Across five seeds,
  `s2_base` achieved 28.8% mean success with high seed-to-seed variability
  (s.d. 40.5 percentage points).
- World-frame termination did not categorically prevent S2 learning. The
  `s2_world_frame_termination` arm achieved 35.0% mean success, with two seeds
  above 60% and two seeds at or below 13%. This weakens any claim that fixed
  world-frame bounds always prevent stair learning in this setup. The more
  defensible claim is that world-frame termination can be harmful and unstable
  on elevated terrain, but the effect was not decisive across these five seeds.
- Removing the explicit stair-shaping terms did not reduce S2 completion in this
  campaign. The `s2_no_shaping` arm achieved the best and most stable result:
  81.2% mean success with 8.3 percentage-point s.d. This directly contradicts a
  strong claim that height progress and step milestone bonuses are necessary for
  S2 success.
- The combination of high rewards and low success in some `s2_base` seeds
  indicates that reward-based checkpoint selection can prefer long-surviving
  policies that do not complete the stairs. For example, `s2_base` seed 44
  averaged 983.8 steps and 14758.9 reward but 0% success and only 1.98 mean
  maximum steps reached.
- The camera-ready paper should therefore shift from causal reward-shaping
  claims to a more modest, transparent multi-seed finding: under the current
  reward and checkpoint-selection protocol, S2 learning is highly seed-sensitive,
  explicit stair shaping was not beneficial in the controlled ablation, and
  world-frame termination showed mixed but generally less stable behavior.

## Paper Changes Supported

- Replace single-seed S2 claims with seed-level and across-seed summaries.
- Clearly state that the checkpoint used for final evaluation was selected by
  mean reward over 5 deterministic evaluation episodes during training, not by
  stair-completion success.
- Remove or heavily soften claims that explicit height/step reward shaping
  enabled S2 success. The controlled ablation shows the opposite trend.
- Remove or heavily soften claims that world-frame termination prevents S2
  learning. The controlled ablation shows mixed outcomes, not categorical
  failure.
- Present the original S2 result as one seed within a five-seed distribution,
  not as representative on its own.
- Add reward misalignment/checkpoint-selection as a limitation: the healthy
  reward can make non-progressing, long-surviving policies look good under
  reward-based checkpoint selection.
- Keep terrain-relative termination as a practical design recommendation, but
  phrase it as reducing a known failure mode rather than as the single causal
  factor that enables stair learning.
