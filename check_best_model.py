import numpy as np

data = np.load('outputs/2025-11-30/11-56-10/eval/evaluations.npz')
means = np.mean(data['results'], axis=1)
best_idx = np.argmax(means)

print(f'Best evaluation at timestep: {data["timesteps"][best_idx]}')
print(f'Mean reward: {means[best_idx]:.2f}')
print(f'Individual episode rewards: {data["results"][best_idx]}')
print(f'\nAll evaluation timesteps and mean rewards:')
for ts, mean_rew in zip(data['timesteps'], means):
    print(f'  {ts:,} steps: {mean_rew:.2f}')
