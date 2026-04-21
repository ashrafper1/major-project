# C:\...\insulin-rl-7-9\evaluate_model.py
# This is the final, definitive, fully vetted evaluation script with plotting.

import gymnasium as gym
import simglucose
import os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import time
import numpy as np
import random
from typing import Callable
from simglucose.simulation.scenario import CustomScenario
import matplotlib.pyplot as plt

# --- 1. WRAPPERS ---
class RandomizedScenarioWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def _create_random_scenario(self):
        # Scenario with a meal (80g carbs) at 60 minutes
        return CustomScenario(start_time=0, scenario=[(60, 80)])

    def reset(self, **kwargs):
        new_scenario = self._create_random_scenario()
        self.env.unwrapped.env.scenario = new_scenario
        return self.env.reset(**kwargs)

class SimglucoseAdvantageWrapper(gym.Wrapper):
    def __init__(self, env, action_penalty_weight=0.01):
        super().__init__(env)
        self.action_penalty_weight = action_penalty_weight
        self.last_action = 0.0
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, 0.0, 0.0]),
            high=np.array([1000.0, 100.0, 300.0]),
            dtype=np.float32
        )

    def _augment_observation(self, obs, info):
        raw_info = info.get('raw_info', {})
        iob = raw_info.get('IOB', 0.0)
        cob = raw_info.get('CHO', 0.0)
        return np.array([obs[0], iob, cob], dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_action = 0.0
        raw_info_reset = {'IOB': 0.0, 'CHO': 0.0}
        return self._augment_observation(obs, raw_info_reset), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info['raw_info'] = info.copy()
        augmented_obs = self._augment_observation(obs, info)
        self.last_action = action[0]
        return augmented_obs, reward, terminated, truncated, info

# --- 2. REWARD FUNCTION ---
def gaussian_reward_shaping(bg_history, **kwargs):
    bg = bg_history[-1]
    mu, sigma, amplitude = 115, 20, 0.05
    reward = amplitude * np.exp(-0.5 * ((bg - mu) / sigma) ** 2)
    scaling_factor = 0.0001
    
    if bg < 70:
        penalty = 5.0 * ((bg - 70) ** 2)
        return -scaling_factor * penalty
    elif bg > 180:
        hyper_penalty_multiplier = 2.0
        penalty = hyper_penalty_multiplier * ((bg - 180) ** 2)
        return -scaling_factor * penalty
    else:
        return reward

# --- 3. ENVIRONMENT CREATOR ---
def make_env(patient_name_list):
    def _init():
        patient_name = random.choice(patient_name_list)
        
        if 'simglucose-v0' not in gym.envs.registry:
            gym.register(id='simglucose-v0', entry_point='simglucose.envs.simglucose_gym_env:T1DSimGymnaisumEnv')

        env = gym.make('simglucose-v0', patient_name=patient_name, reward_fun=gaussian_reward_shaping, max_episode_steps=1152) # 4 days
        
        env = SimglucoseAdvantageWrapper(env)
        env = Monitor(env, info_keywords=('raw_info',))
        env = RandomizedScenarioWrapper(env)
        return env
    return _init

# --- Main Execution Block ---
if __name__ == "__main__":
    # --- 4. Configuration ---
    MODEL_TIMESTAMP = "final_stable_v2_20250911-211359"
    EVALUATION_PATIENT = ['adult#001']

    model_dir = os.path.join("models", MODEL_TIMESTAMP)
    model_path = os.path.join(model_dir, "best_model.zip")
    stats_path = os.path.join(model_dir, "vec_normalize.pkl")

    # Check if model and stats files exist before running
    if not os.path.exists(model_path) or not os.path.exists(stats_path):
        print(f"Error: Model or normalization file not found.")
        print(f"Looked for: {model_path}")
        print(f"And for: {stats_path}")
        exit()

    print(f"--- Loading model from: {model_path} ---")

    # --- 5. Create the Evaluation Environment ---
    eval_env = DummyVecEnv([make_env(EVALUATION_PATIENT)])
    
    print(f"--- Loading normalization stats from: {stats_path} ---")
    eval_env = VecNormalize.load(stats_path, eval_env)
    
    eval_env.training = False
    eval_env.norm_reward = False

    model = PPO.load(model_path, env=eval_env)

    # --- 6. Run the Evaluation Loop & Collect Data ---
    print("\n--- Starting Model Evaluation ---")
    
    all_bg_values = []
    all_actions = []

    obs = eval_env.reset()
    for i in range(1152): # 4 full days
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)
        
        original_obs = eval_env.get_original_obs()
        bg_value = original_obs[0][0]
        all_bg_values.append(bg_value)
        
        insulin_rate = action[0][0]
        all_actions.append(insulin_rate)
        
        if i % 48 == 0: # Print every 4 hours
            print(f"Time: {i*5//60}h | Step {i+1}: BG = {bg_value:.2f} mg/dL | Insulin Rate = {insulin_rate:.4f} U/hr")

    eval_env.close()
    print("\n--- Evaluation Run Complete ---")

    # --- 7. Generate the Quantitative Evaluation Report ---
    print("\n--- Quantitative Clinical Evaluation Report ---")
    
    bg_values = np.array(all_bg_values)
    actions = np.array(all_actions)
    
    time_in_range = np.mean((bg_values >= 70) & (bg_values <= 180)) * 100
    time_below_range = np.mean(bg_values < 70) * 100
    time_above_range = np.mean(bg_values > 180) * 100
    mean_bg = np.mean(bg_values)
    std_bg = np.std(bg_values)
    total_insulin = np.sum(actions) * (5/60) 
    
    print(f"\nClinical Outcomes:")
    print(f"  - Time in Range (70-180 mg/dL): {time_in_range:.2f} %")
    print(f"  - Time Below Range (<70 mg/dL):  {time_below_range:.2f} %")
    print(f"  - Time Above Range (>180 mg/dL): {time_above_range:.2f} %")
    print(f"  - Mean Glucose:                 {mean_bg:.2f} mg/dL")
    print(f"  - Glucose Variability (Std Dev):{std_bg:.2f} mg/dL")

    print(f"\nAgent Behavior:")
    print(f"  - Total Insulin Administered:     {total_insulin:.2f} U")

    # --- 8. Generate the Visualization Plot ---
    print("\n--- Generating Evaluation Plot ---")

    time_in_hours = np.arange(len(all_bg_values)) * 5 / 60

    fig, ax1 = plt.subplots(figsize=(15, 7))

    color = 'tab:blue'
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('Blood Glucose (mg/dL)', color=color)
    ax1.plot(time_in_hours, all_bg_values, color=color, label='Blood Glucose')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    ax1.axhline(y=180, color='r', linestyle='--', label='Hyperglycemia Threshold (180 mg/dL)')
    ax1.axhline(y=70, color='g', linestyle='--', label='Hypoglycemia Threshold (70 mg/dL)')
    
    ax2 = ax1.twinx()  
    color = 'tab:orange'
    ax2.set_ylabel('Insulin Rate (U/hr)', color=color)
    ax2.plot(time_in_hours, all_actions, color=color, alpha=0.7, label='Insulin Rate')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(f'PPO Model Performance for Patient: {EVALUATION_PATIENT[0]}')
    fig.tight_layout()
    
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')

    plt.show()