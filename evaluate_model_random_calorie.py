import gymnasium as gym
import simglucose
import os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import time
import numpy as np
import random
from simglucose.simulation.scenario import CustomScenario
import matplotlib.pyplot as plt

# --- 1. RANDOMIZED SCENARIO WRAPPER ---
# This class generates random meals every time the environment resets
class RandomizedScenarioWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def _create_random_scenario(self):
        scenario = []
        # We are simulating 4 days. Let's create random meals for each day.
        for day in range(4):
            day_offset = day * 1440  # 1440 minutes in one day
            
            # Random Breakfast (Between 7:00 AM and 9:00 AM)
            # Time is in minutes. 7*60 = 420, 9*60 = 540
            b_time = day_offset + random.randint(420, 540)
            b_carbs = random.randint(30, 60) # Random carbs between 30g and 60g
            scenario.append((b_time, b_carbs))
            
            # Random Lunch (Between 12:00 PM and 2:00 PM)
            l_time = day_offset + random.randint(720, 840)
            l_carbs = random.randint(50, 90) # Lunch is usually larger
            scenario.append((l_time, l_carbs))

            # Random Dinner (Between 7:00 PM and 9:00 PM)
            d_time = day_offset + random.randint(1140, 1260)
            d_carbs = random.randint(40, 80)
            scenario.append((d_time, d_carbs))

        # Sort the meals by time (required by SimGlucose)
        scenario.sort(key=lambda x: x[0])
        
        # Print the scenario to the console so you can see what the agent is facing
        print("\n--- NEW RANDOM SCENARIO GENERATED ---")
        for t, c in scenario:
            print(f"Time: {t} min | Carbs: {c}g")
        print("-------------------------------------")
        
        return CustomScenario(start_time=0, scenario=scenario)

    def reset(self, **kwargs):
        new_scenario = self._create_random_scenario()
        # Inject the new random scenario into the simulator
        self.env.unwrapped.env.scenario = new_scenario
        return self.env.reset(**kwargs)

# --- 2. FEATURE ENGINEERING WRAPPER (Must match training) ---
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

# --- 3. REWARD FUNCTION (Required to load env, even if not training) ---
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

# --- 4. ENVIRONMENT CREATOR ---
def make_env(patient_name_list):
    def _init():
        patient_name = random.choice(patient_name_list)
        
        if 'simglucose-v0' not in gym.envs.registry:
            gym.register(id='simglucose-v0', entry_point='simglucose.envs.simglucose_gym_env:T1DSimGymnaisumEnv')

        # 4 days = 1152 steps
        env = gym.make('simglucose-v0', patient_name=patient_name, reward_fun=gaussian_reward_shaping, max_episode_steps=1152) 
        
        env = SimglucoseAdvantageWrapper(env)
        env = Monitor(env, info_keywords=('raw_info',))
        # APPLY THE RANDOM WRAPPER HERE
        env = RandomizedScenarioWrapper(env)
        return env
    return _init

# --- Main Execution Block ---
if __name__ == "__main__":
    # --- Configuration ---
    # ENSURE THIS MATCHES YOUR FOLDER NAME EXACTLY
    MODEL_TIMESTAMP = "final_stable_v2_20250911-211359"
    EVALUATION_PATIENT = ['adult#001']

    model_dir = os.path.join("models", MODEL_TIMESTAMP)
    model_path = os.path.join(model_dir, "best_model.zip")
    stats_path = os.path.join(model_dir, "vec_normalize.pkl")

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        exit()

    print(f"--- Loading model from: {model_path} ---")

    # --- Setup Eval Environment ---
    eval_env = DummyVecEnv([make_env(EVALUATION_PATIENT)])
    eval_env = VecNormalize.load(stats_path, eval_env)
    eval_env.training = False
    eval_env.norm_reward = False

    model = PPO.load(model_path, env=eval_env)

    # --- Run Simulation ---
    print("\n--- Starting Randomized Meal Evaluation (4 Days) ---")
    
    all_bg_values = []
    all_actions = []

    obs = eval_env.reset()
    for i in range(1152): 
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)
        
        original_obs = eval_env.get_original_obs()
        bg_value = original_obs[0][0]
        all_bg_values.append(bg_value)
        insulin_rate = action[0][0]
        all_actions.append(insulin_rate)

        if i % 48 == 0:
             print(f"Step {i}: BG={bg_value:.1f}, Insulin={insulin_rate:.4f}")

    eval_env.close()

    # --- Calculate Metrics ---
    bg_values = np.array(all_bg_values)
    time_in_range = np.mean((bg_values >= 70) & (bg_values <= 180)) * 100
    time_below_range = np.mean(bg_values < 70) * 100
    time_above_range = np.mean(bg_values > 180) * 100
    mean_bg = np.mean(bg_values)
    std_bg = np.std(bg_values)
    total_insulin = np.sum(all_actions) * (5/60)
    
    print("\n--- Randomized Test Results ---")
    print(f"Time in Range (70-180): {time_in_range:.2f} %")
    print(f"Time Below Range (<70):  {time_below_range:.2f} %")
    print(f"Time Above Range (>180): {time_above_range:.2f} %")
    print(f"Mean BG: {mean_bg:.2f} mg/dL")
    print(f"Variability (SD): {std_bg:.2f} mg/dL")
    print(f"Total Insulin: {total_insulin:.2f} U")

    # --- Plotting ---
    time_in_hours = np.arange(len(all_bg_values)) * 5 / 60
    fig, ax1 = plt.subplots(figsize=(15, 7))

    color = 'tab:blue'
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('Blood Glucose (mg/dL)', color=color)
    ax1.plot(time_in_hours, all_bg_values, color=color, label='Blood Glucose')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Add Safe Range Lines
    ax1.axhline(y=180, color='r', linestyle='--', label='Hyperglycemia Limit')
    ax1.axhline(y=70, color='g', linestyle='--', label='Hypoglycemia Limit')
    
    ax2 = ax1.twinx()  
    color = 'tab:orange'
    ax2.set_ylabel('Insulin Rate (U/hr)', color=color)
    ax2.plot(time_in_hours, all_actions, color=color, alpha=0.5, label='Insulin Rate')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(f'PPO Performance: Randomized Meals (Adult#001)')
    fig.tight_layout()
    plt.show()+9
    