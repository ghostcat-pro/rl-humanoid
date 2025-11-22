#!/usr/bin/env python3
"""
Evaluate all trained models in the outputs directory and generate a comprehensive results report.
"""

import os
import sys
import subprocess
from pathlib import Path
import json
import re
import yaml

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

def find_all_models():
    """Find all final_model.zip files in outputs directory."""
    outputs_dir = Path("outputs")
    models = []
    
    for model_path in outputs_dir.rglob("final_model.zip"):
        run_dir = model_path.parent
        vecnorm_path = run_dir / "vecnormalize_final.pkl"
        config_path = run_dir / ".hydra" / "config.yaml"
        
        # Check if vecnorm exists
        if vecnorm_path.exists():
            # Try to load config
            config = None
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        config = yaml.safe_load(f)
                except Exception as e:
                    print(f"Warning: Could not load config from {config_path}: {e}")
            
            models.append({
                'run_dir': str(run_dir),
                'model_path': str(model_path),
                'vecnorm_path': str(vecnorm_path),
                'timestamp': run_dir.name,
                'config': config
            })
    
    return sorted(models, key=lambda x: x['timestamp'])


def evaluate_model(model_info, env_id=None, episodes=20):
    """Evaluate a single model and return results."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_info['run_dir']}")
    print(f"{'='*60}")

    # Auto-detect environment from config if not specified
    if env_id is None and model_info.get('config'):
        env_id = model_info['config'].get('env', {}).get('name', 'Humanoid-v5')
    elif env_id is None:
        env_id = 'Humanoid-v5'

    print(f"Environment: {env_id}")

    cmd = [
        ".venv/bin/python", "evaluate_stats.py",
        "--env_id", env_id,
        "--model_path", model_info['model_path'],
        "--vecnorm_path", model_info['vecnorm_path'],
        "--deterministic",
        "--episodes", str(episodes)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        output = result.stdout
        
        # Parse results from output
        results = {
            'run_dir': model_info['run_dir'],
            'timestamp': model_info['timestamp'],
            'episodes': episodes,
            'success': result.returncode == 0,
            'output': output,
            'config': model_info.get('config')
        }
        
        # Extract statistics using regex
        if "Mean reward:" in output:
            mean_match = re.search(r'Mean reward:\s+([\d.]+)\s+±\s+([\d.]+)', output)
            if mean_match:
                results['mean_reward'] = float(mean_match.group(1))
                results['std_reward'] = float(mean_match.group(2))
            
            min_match = re.search(r'Min reward:\s+([\d.]+)', output)
            if min_match:
                results['min_reward'] = float(min_match.group(1))
            
            max_match = re.search(r'Max reward:\s+([\d.]+)', output)
            if max_match:
                results['max_reward'] = float(max_match.group(1))
            
            median_match = re.search(r'Median reward:\s+([\d.]+)', output)
            if median_match:
                results['median_reward'] = float(median_match.group(1))
            
            length_match = re.search(r'Mean length:\s+([\d.]+)\s+steps', output)
            if length_match:
                results['mean_length'] = float(length_match.group(1))
            
            success_match = re.search(r'Success rate.*:\s+([\d.]+)%', output)
            if success_match:
                results['success_rate'] = float(success_match.group(1))
        
        return results
        
    except subprocess.TimeoutExpired:
        print(f"ERROR: Evaluation timed out for {model_info['run_dir']}")
        return {
            'run_dir': model_info['run_dir'],
            'timestamp': model_info['timestamp'],
            'success': False,
            'error': 'Timeout'
        }
    except Exception as e:
        print(f"ERROR: {e}")
        return {
            'run_dir': model_info['run_dir'],
            'timestamp': model_info['timestamp'],
            'success': False,
            'error': str(e)
        }


def generate_report(all_results, output_file="results.txt"):
    """Generate a comprehensive text report."""

    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("RL HUMANOID MODEL EVALUATION RESULTS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total Models Evaluated: {len(all_results)}\n")
        f.write(f"Episodes per Model: {all_results[0]['episodes'] if all_results else 'N/A'}\n")
        f.write(f"Evaluation Mode: Deterministic\n")
        f.write("=" * 80 + "\n\n")
        
        # Summary table
        f.write("SUMMARY TABLE\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Run':<15} {'Total Steps':<15} {'Mean Reward':<15} {'Std Dev':<12} {'Success':<10}\n")
        f.write("-" * 80 + "\n")
        
        successful_results = [r for r in all_results if r['success'] and 'mean_reward' in r]
        
        for result in successful_results:
            run_name = result['timestamp']
            config = result.get('config', {})
            total_steps = config.get('training', {}).get('total_timesteps', 'N/A') if config else 'N/A'
            if total_steps != 'N/A':
                total_steps = f"{total_steps/1e6:.1f}M"
            mean_reward = f"{result.get('mean_reward', 0):.2f}"
            std_reward = f"±{result.get('std_reward', 0):.2f}"
            success_rate = f"{result.get('success_rate', 0):.0f}%"
            f.write(f"{run_name:<15} {total_steps:<15} {mean_reward:<15} {std_reward:<12} {success_rate:<10}\n")
        
        f.write("-" * 80 + "\n\n")
        
        # Ranking
        if successful_results:
            f.write("RANKING (by Mean Reward)\n")
            f.write("-" * 80 + "\n")
            ranked = sorted(successful_results, key=lambda x: x.get('mean_reward', 0), reverse=True)
            
            for i, result in enumerate(ranked, 1):
                run_name = result['timestamp']
                mean_reward = result.get('mean_reward', 0)
                config = result.get('config', {})
                total_steps = config.get('training', {}).get('total_timesteps', 'N/A') if config else 'N/A'
                if total_steps != 'N/A':
                    total_steps = f" ({total_steps/1e6:.0f}M steps)"
                else:
                    total_steps = ""
                f.write(f"{i}. {run_name}{total_steps}: {mean_reward:.2f}\n")
            
            f.write("\n")
            
            # Best model
            best = ranked[0]
            f.write("BEST MODEL\n")
            f.write("-" * 80 + "\n")
            f.write(f"Run Directory: {best['run_dir']}\n")
            f.write(f"Timestamp: {best['timestamp']}\n")
            f.write(f"Mean Reward: {best.get('mean_reward', 0):.2f} ± {best.get('std_reward', 0):.2f}\n")
            f.write(f"Min/Max Reward: {best.get('min_reward', 0):.2f} / {best.get('max_reward', 0):.2f}\n")
            f.write(f"Median Reward: {best.get('median_reward', 0):.2f}\n")
            f.write(f"Mean Episode Length: {best.get('mean_length', 0):.2f} steps\n")
            f.write(f"Success Rate: {best.get('success_rate', 0):.1f}%\n")
            
            # Add config info for best model
            best_config = best.get('config')
            if best_config:
                f.write("\nTraining Configuration:\n")
                training = best_config.get('training', {})
                algo = best_config.get('algo', {})
                hyperparams = algo.get('hyperparams', {})
                env = best_config.get('env', {})
                
                f.write(f"  Total Timesteps: {training.get('total_timesteps', 'N/A'):,}\n")
                f.write(f"  Environment: {env.get('name', 'N/A')}\n")
                f.write(f"  Num Envs: {env.get('vec_env', {}).get('n_envs', 'N/A')}\n")
                f.write(f"  Learning Rate: {hyperparams.get('learning_rate', 'N/A')}\n")
                f.write(f"  Batch Size: {hyperparams.get('batch_size', 'N/A')}\n")
                f.write(f"  N Steps: {hyperparams.get('n_steps', 'N/A')}\n")
                f.write(f"  N Epochs: {hyperparams.get('n_epochs', 'N/A')}\n")
                f.write(f"  Entropy Coefficient: {hyperparams.get('ent_coef', 'N/A')}\n")
            f.write("\n")
        
        # Detailed results for each model
        f.write("=" * 80 + "\n")
        f.write("DETAILED RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        for result in all_results:
            f.write(f"Run Directory: {result['run_dir']}\n")
            f.write(f"Timestamp: {result['timestamp']}\n")
            f.write("-" * 80 + "\n")
            
            # Add config info
            config = result.get('config')
            if config:
                f.write("\nTraining Configuration:\n")
                training = config.get('training', {})
                algo = config.get('algo', {})
                hyperparams = algo.get('hyperparams', {})
                env = config.get('env', {})
                vecnorm = config.get('vecnorm', {})
                
                f.write(f"  Total Timesteps: {training.get('total_timesteps', 'N/A'):,}\n")
                f.write(f"  Environment: {env.get('name', 'N/A')}\n")
                f.write(f"  Num Parallel Envs: {env.get('vec_env', {}).get('n_envs', 'N/A')}\n")
                f.write(f"  Policy: {algo.get('policy', 'N/A')}\n")
                f.write(f"  Device: {algo.get('device', 'N/A')}\n")
                f.write(f"  Learning Rate: {hyperparams.get('learning_rate', 'N/A')}\n")
                f.write(f"  Batch Size: {hyperparams.get('batch_size', 'N/A')}\n")
                f.write(f"  N Steps: {hyperparams.get('n_steps', 'N/A')}\n")
                f.write(f"  N Epochs: {hyperparams.get('n_epochs', 'N/A')}\n")
                f.write(f"  Gamma: {hyperparams.get('gamma', 'N/A')}\n")
                f.write(f"  GAE Lambda: {hyperparams.get('gae_lambda', 'N/A')}\n")
                f.write(f"  Clip Range: {hyperparams.get('clip_range', 'N/A')}\n")
                f.write(f"  Entropy Coef: {hyperparams.get('ent_coef', 'N/A')}\n")
                f.write(f"  Value Function Coef: {hyperparams.get('vf_coef', 'N/A')}\n")
                f.write(f"  Max Grad Norm: {hyperparams.get('max_grad_norm', 'N/A')}\n")
                f.write(f"  VecNormalize Enabled: {vecnorm.get('enabled', 'N/A')}\n")
                f.write(f"  Checkpoint Every: {training.get('checkpoint_every_steps', 'N/A'):,} steps\n")
                f.write("\n")
            
            f.write("Evaluation Results:\n")
            if result['success'] and 'mean_reward' in result:
                f.write(f"  Episodes: {result['episodes']}\n")
                f.write(f"  Mean Reward: {result.get('mean_reward', 0):.2f} ± {result.get('std_reward', 0):.2f}\n")
                f.write(f"  Min Reward: {result.get('min_reward', 0):.2f}\n")
                f.write(f"  Max Reward: {result.get('max_reward', 0):.2f}\n")
                f.write(f"  Median Reward: {result.get('median_reward', 0):.2f}\n")
                f.write(f"  Mean Episode Length: {result.get('mean_length', 0):.2f} steps\n")
                f.write(f"  Success Rate (>5000): {result.get('success_rate', 0):.1f}%\n")
            else:
                f.write(f"  Status: FAILED\n")
                if 'error' in result:
                    f.write(f"  Error: {result['error']}\n")
            
            f.write("\n" + "=" * 80 + "\n\n")
    
    print(f"\n✓ Results saved to: {output_file}")


def main():
    print("Finding all models...")
    models = find_all_models()
    
    if not models:
        print("ERROR: No models found in outputs directory")
        sys.exit(1)
    
    print(f"Found {len(models)} models to evaluate")
    
    episodes = 20
    all_results = []
    
    for i, model_info in enumerate(models, 1):
        print(f"\n[{i}/{len(models)}] Evaluating {model_info['run_dir']}...")
        results = evaluate_model(model_info, episodes=episodes)
        all_results.append(results)
    
    # Generate report
    generate_report(all_results)
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
