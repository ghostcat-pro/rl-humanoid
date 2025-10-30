# Model Training Analysis - Key Findings

## Executive Summary

Evaluated 7 different Humanoid-v5 models trained with varying timesteps (1M to 100M). 

**Key Discovery**: More training is NOT always better. The 30M model significantly outperforms the 50M and 100M models.

## Performance Rankings

| Rank | Run ID    | Training Steps | Mean Reward | Success Rate | Status |
|------|-----------|----------------|-------------|--------------|--------|
| 🥇 1 | 13-02-09  | 30M            | 9870.22     | 100%         | ⭐ BEST |
| 🥈 2 | 16-30-12  | 50M            | 7435.45     | 75%          | Good |
| 🥈 2 | 17-47-25  | 50M            | 7435.45     | 75%          | Good |
| 4    | 12-25-32  | 10M            | 6920.77     | 100%         | Decent |
| 5    | 21-33-51  | 100M           | 4331.37     | 40%          | Poor |
| 6    | 11-33-29  | 1M             | 489.76      | 0%           | Failed |
| 7    | 12-17-01  | 1M             | 489.76      | 0%           | Failed |

## Critical Insight: The 30M Model is Optimal

### Best Model (13-02-09 - 30M steps)
- **Mean Reward**: 9870.22 ± 73.74
- **Success Rate**: 100% (all 20 episodes successful)
- **Consistency**: Excellent (very low variance)
- **Episode Length**: Full 1000 steps every time
- **Reward Range**: 9716 - 9974 (near perfect)

### Why 30M Beats 50M and 100M

The performance actually **decreases** with additional training beyond 30M:

```
30M steps  → 9870 reward (100% success) ✅ PEAK PERFORMANCE
50M steps  → 7435 reward (75% success)  ⬇️ Degradation
100M steps → 4331 reward (40% success) ⬇️⬇️ Significant degradation
```

This is a textbook case of **overtraining/overfitting**:
- The model learned optimal behavior at 30M steps
- Beyond 30M, it began overfitting to specific training scenarios
- Lost generalization ability, became less robust

## Configuration Comparison

### Best Model (30M) Configuration:
```yaml
Total Timesteps: 30,000,000
Num Parallel Envs: 16
Learning Rate: 0.00025
Batch Size: 8192
N Steps: 1024
N Epochs: 10
Entropy Coef: 0.005
```

### 100M Model (Poor Performance) Configuration:
```yaml
Total Timesteps: 100,000,000
Num Parallel Envs: 16
Learning Rate: 0.0003
Batch Size: 8192
N Steps: 2048
N Epochs: 10
Entropy Coef: 0.0
```

**Key Differences**:
1. **Training Duration**: 100M trained 3.3x longer (likely overtrained)
2. **Learning Rate**: 100M used slightly higher LR (0.0003 vs 0.00025)
3. **N Steps**: 100M used 2048 vs 1024 (double rollout length)
4. **Entropy Coefficient**: 100M used 0.0 vs 0.005 (no exploration bonus)

The lack of entropy bonus (0.0) in the 100M model likely caused it to stop exploring and overfit.

## Recommendations

### For Deployment:
✅ **Use model: `outputs/2025-10-28/13-02-09/`** (30M steps)
- Best performance
- Most consistent
- 100% success rate

### For Future Training:
1. **Optimal training duration**: ~30M steps for Humanoid-v5
2. **Use entropy bonus**: Keep ent_coef around 0.005 for exploration
3. **Monitor performance**: Stop training when validation performance plateaus
4. **Early stopping**: Implement checkpointing and keep best model, not final model

### Training Duration Guidelines:
- ❌ 1M steps: Too early (model hasn't learned)
- ✅ 10M steps: Basic walking (6920 reward)
- ⭐ 30M steps: Optimal performance (9870 reward) 
- ⚠️ 50M steps: Starting to degrade (7435 reward)
- ❌ 100M steps: Significant overtraining (4331 reward)

## Files Generated

- `results.txt` - Full detailed results with all configurations
- `evaluate_all_models.py` - Script to re-run evaluation
- `evaluate_stats.py` - Individual model evaluation script

## Next Steps

1. Use the 30M model (13-02-09) for your application
2. If training new models, stop around 30M steps
3. Always use validation/evaluation during training to detect overtraining
4. Consider implementing early stopping based on evaluation metrics
