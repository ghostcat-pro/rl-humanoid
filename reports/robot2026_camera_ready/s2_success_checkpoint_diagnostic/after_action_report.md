# S2 Success-Based Checkpoint Diagnostic

## Protocol

```json
{
  "generated_utc": "2026-09-30T10:02:58.923293+00:00",
  "manifest": "experiments/robot2026_camera_ready/s2_ablation_manifest.csv",
  "eval_seed": 123,
  "coarse_episodes": 10,
  "refine_episodes": 20,
  "final_episodes": 100,
  "coarse_interval": 1000000,
  "refine_window": 1000000,
  "scan_checkpoints_evaluated": 561,
  "selected_runs": 15,
  "missing_runs": 0
}
```

This diagnostic does not replace the reward-selected S2 ablation report.
It asks whether stored periodic checkpoints contain better stair-completion policies than the reward-selected `eval/best_model.zip` checkpoints.

Selection criterion: maximize success rate, then mean maximum step reached, final x-position, mean reward, and later checkpoint step.

## Final Arm Summary

| arm | seeds_completed | seeds | mean_success_rate | sd_success_rate | mean_reward_across_seeds | sd_reward_across_seeds | mean_length_across_seeds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| s2_base | 5 | 42 43 44 45 46 | 42.80 | 46.22 | 14827.82 | 750.16 | 951.34 |
| s2_no_shaping | 5 | 42 43 44 45 46 | 86.60 | 7.64 | 14812.01 | 783.54 | 935.12 |
| s2_world_frame_termination | 5 | 42 43 44 45 46 | 51.20 | 36.16 | 12792.21 | 1786.16 | 817.19 |

## Selected Checkpoints

| arm | seed | checkpoint_step | selection_stage | selection_success_rate | final_success_rate | final_mean_reward | final_mean_length | final_mean_max_step_reached |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| s2_base | 42 | 28750000 | refine | 100.00 | 97.00 | 15693.08 | 957.64 | 7.86 |
| s2_base | 43 | 30000000 | refine | 85.00 | 86.00 | 15356.65 | 973.36 | 7.69 |
| s2_base | 44 | 30000000 | refine | 0.00 | 0.00 | 14516.84 | 967.56 | 1.96 |
| s2_base | 45 | 19500000 | refine | 5.00 | 1.00 | 13766.95 | 910.20 | 2.19 |
| s2_base | 46 | 29000000 | refine | 30.00 | 30.00 | 14805.58 | 947.93 | 6.76 |
| s2_world_frame_termination | 42 | 21750000 | refine | 80.00 | 57.00 | 12970.80 | 826.10 | 6.92 |
| s2_world_frame_termination | 43 | 7500000 | refine | 0.00 | 0.00 | 9721.16 | 642.04 | 2.01 |
| s2_world_frame_termination | 44 | 27000000 | refine | 100.00 | 99.00 | 13253.24 | 835.53 | 7.98 |
| s2_world_frame_termination | 45 | 23000000 | refine | 50.00 | 38.00 | 14262.61 | 911.15 | 6.92 |
| s2_world_frame_termination | 46 | 29250000 | refine | 70.00 | 62.00 | 13753.27 | 871.12 | 7.06 |
| s2_no_shaping | 42 | 27000000 | refine | 100.00 | 92.00 | 14853.09 | 945.42 | 7.54 |
| s2_no_shaping | 43 | 30000000 | refine | 100.00 | 96.00 | 15683.43 | 964.31 | 7.68 |
| s2_no_shaping | 44 | 29750000 | refine | 90.00 | 79.00 | 14333.90 | 913.16 | 7.41 |
| s2_no_shaping | 45 | 30000000 | refine | 90.00 | 87.00 | 15422.99 | 988.03 | 7.79 |
| s2_no_shaping | 46 | 29000000 | refine | 90.00 | 79.00 | 13766.63 | 864.67 | 7.21 |

## Missing Runs

_None._

## Conclusions

- Success-based checkpoint selection improved the results for all three arms
  relative to the reward-selected report:
  - `s2_base`: 28.8% -> 42.8% mean success.
  - `s2_world_frame_termination`: 35.0% -> 51.2% mean success.
  - `s2_no_shaping`: 81.2% -> 86.6% mean success.
- This confirms that reward-based checkpoint selection was a real confound.
  Some useful stair-completion policies existed in stored periodic checkpoints
  but were not selected by the original reward-based `EvalCallback` criterion.
- The confound does not fully explain the five-seed result. Even after
  success-based checkpoint selection, `s2_base` remains highly variable and much
  weaker than `s2_no_shaping`: 42.8% mean success with 46.2 percentage-point
  s.d. versus 86.6% with 7.6 percentage-point s.d.
- The `s2_no_shaping` arm remains the most stable and successful arm under both
  checkpoint-selection protocols. This reinforces the conclusion that the tested
  explicit height/step shaping terms did not improve S2 completion.
- World-frame termination also improves under success-based checkpoint
  selection, but remains variable. It cannot be described as categorically
  preventing learning in this S2 setup.
- This diagnostic used a staged checkpoint scan, not an exhaustive final
  evaluation of all 1,800 checkpoints: coarse scan every 1M steps with 10
  episodes, refinement within +/-1M of the coarse winner with 20 episodes, and
  final 100-episode evaluation of the selected checkpoint. The conclusion is
  strong enough to identify checkpoint selection as a confound, but exact
  success-selected optima may differ slightly under a denser scan.

## Paper Implications

- The paper can now explicitly report two protocol-dependent summaries:
  reward-selected checkpoints and success-selected diagnostic checkpoints.
- The strongest defensible statement is that reward-based checkpointing
  understated stair-completion performance for some seeds, but did not reverse
  the overall finding that `s2_no_shaping` was the most reliable S2 arm.
- Claims that reward shaping enabled S2 success should still be removed or
  heavily softened.
- Claims that world-frame termination prevents S2 learning should still be
  removed or heavily softened.
- Add checkpoint-selection mismatch as a major methodological lesson: in sparse
  task-completion settings, reward-selected checkpoints can prefer survival over
  actual task completion.
