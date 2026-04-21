import gymnasium as gym
import simglucose
import os
import time
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from simglucose.simulation.scenario import CustomScenario

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
MODEL_TIMESTAMP = "final_stable_v2_20250911-211359" 
PATIENT_ID = 'adult#001'

MODEL_DIR = os.path.join("models", MODEL_TIMESTAMP)
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.zip")
STATS_PATH = os.path.join(MODEL_DIR, "vec_normalize.pkl")

# ==============================================================================
# 2. SCENARIO WRAPPER (Hybrid)
# ==============================================================================
class HybridScenarioWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.scenario_data = [] 
    
    def _create_hybrid_scenario(self):
        scenario_list = []
        self.scenario_data = [] 
        
        for day in range(4):
            day_offset = day * 1440 
            
            # Fixed Breakfast
            t_b, c_b = day_offset + 480, 45
            scenario_list.append((t_b, c_b))
            self.scenario_data.append((t_b, c_b, "Fixed Breakfast"))
            
            # Fixed Lunch
            t_l, c_l = day_offset + 780, 70
            scenario_list.append((t_l, c_l))
            self.scenario_data.append((t_l, c_l, "Fixed Lunch"))
            
            # Fixed Dinner
            t_d, c_d = day_offset + 1200, 60
            scenario_list.append((t_d, c_d))
            self.scenario_data.append((t_d, c_d, "Fixed Dinner"))

            # Random Snack
            possible_times = list(range(day_offset + 150, day_offset + 400)) + \
                             list(range(day_offset + 900, day_offset + 1100))
            t_snack = random.choice(possible_times)
            c_snack = random.randint(20, 35) 
            scenario_list.append((t_snack, c_snack))
            self.scenario_data.append((t_snack, c_snack, "Random Snack"))

        scenario_list.sort(key=lambda x: x[0])
        self.scenario_data.sort(key=lambda x: x[0])
        return CustomScenario(start_time=0, scenario=scenario_list)

    def reset(self, **kwargs):
        new_scenario = self._create_hybrid_scenario()
        self.env.unwrapped.env.scenario = new_scenario
        return self.env.reset(**kwargs)

# ==============================================================================
# 3. STANDARD WRAPPERS
# ==============================================================================
class SimglucoseAdvantageWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
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
        raw_info_reset = {'IOB': 0.0, 'CHO': 0.0}
        return self._augment_observation(obs, raw_info_reset), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info['raw_info'] = info.copy()
        augmented_obs = self._augment_observation(obs, info)
        return augmented_obs, reward, terminated, truncated, info

def gaussian_reward_shaping(bg_history, **kwargs):
    return 0.0

def make_env(patient_name_list):
    def _init():
        patient_name = random.choice(patient_name_list)
        if 'simglucose-v0' not in gym.envs.registry:
            gym.register(id='simglucose-v0', entry_point='simglucose.envs.simglucose_gym_env:T1DSimGymnaisumEnv')
        env = gym.make('simglucose-v0', patient_name=patient_name, reward_fun=gaussian_reward_shaping, max_episode_steps=1152) 
        env = SimglucoseAdvantageWrapper(env)
        env = Monitor(env, info_keywords=('raw_info',))
        env = HybridScenarioWrapper(env)
        return env
    return _init

def analyze_period(bg_list):
    arr = np.array(bg_list)
    tir = np.mean((arr >= 70) & (arr <= 180)) * 100
    avg = np.mean(arr)
    return tir, avg

def get_time_str(total_mins):
    day = int(total_mins // 1440) + 1
    mins = total_mins % 1440
    hr = int(mins // 60)
    mn = int(mins % 60)
    return f"Day {day} {hr:02d}:{mn:02d}"

# ==============================================================================
# 4. MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        exit()

    print(f"--- STARTING HYBRID EVALUATION (FIXED MEALS + RANDOM SNACKS) ---")
    
    eval_env = DummyVecEnv([make_env([PATIENT_ID])])
    eval_env = VecNormalize.load(STATS_PATH, eval_env)
    eval_env.training = False
    eval_env.norm_reward = False
    model = PPO.load(MODEL_PATH, env=eval_env)

    obs = eval_env.reset()
    
    scenario_data = eval_env.envs[0].scenario_data
    meal_dict = {item[0]: (item[1], item[2]) for item in scenario_data}
    
    all_bg = []
    all_insulin = []
    daily_bg = []

    steps_per_day = 288
    
    # --- SIMULATION LOOP ---
    for day in range(1, 5):
        print(f"\n[STARTING DAY {day}]")
        daily_bg = []
        
        for step in range(steps_per_day):
            global_step = (day - 1) * steps_per_day + step
            current_time = global_step * 5
            
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            
            bg = eval_env.get_original_obs()[0][0]
            ins = action[0][0]
            
            all_bg.append(bg)
            all_insulin.append(ins)
            daily_bg.append(bg)
            
            if current_time in meal_dict:
                carbs, label = meal_dict[current_time]
                t_str = get_time_str(current_time)
                print(f"[{t_str}] Event: {label} ({carbs}g CHO) - BG: {bg:.1f}")

        tir, avg = analyze_period(daily_bg)
        print(f"--- DAY {day} SUMMARY ---")
        print(f"Avg Glucose: {avg:.1f} mg/dL")
        print(f"Time in Range: {tir:.2f}%")

    eval_env.close()

    # ==============================================================================
    # 5. FINAL AGGREGATE SUMMARY (NEW ADDITION)
    # ==============================================================================
    print("\n" + "="*40)
    print("FINAL AGGREGATE SUMMARY (ALL 4 DAYS)")
    print("="*40)
    
    bg_total = np.array(all_bg)
    ins_total = np.array(all_insulin)
    
    tir_total = np.mean((bg_total >= 70) & (bg_total <= 180)) * 100
    tbr_total = np.mean(bg_total < 70) * 100
    tar_total = np.mean(bg_total > 180) * 100
    mean_bg_total = np.mean(bg_total)
    std_bg_total = np.std(bg_total)
    total_ins_delivered = np.sum(ins_total) * (5/60) # Convert rate to units
    
    print(f"Total Time in Range (70-180): {tir_total:.2f}%")
    print(f"Total Time Below Range (<70): {tbr_total:.2f}%")
    print(f"Total Time Above Range (>180): {tar_total:.2f}%")
    print(f"Overall Mean Glucose:         {mean_bg_total:.2f} mg/dL")
    print(f"Glucose Variability (SD):     {std_bg_total:.2f} mg/dL")
    print(f"Total Insulin Delivered:      {total_ins_delivered:.2f} U")
    print("="*40 + "\n")

    # ==============================================================================
    # 6. GRAPH PLOTTING
    # ==============================================================================
    print("Generating Graph...")
    
    time_hours = np.arange(len(all_bg)) * 5 / 60
    
    fig, ax1 = plt.subplots(figsize=(15, 7))

    # Blood Glucose
    color = 'tab:blue'
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('Blood Glucose (mg/dL)', color=color)
    ax1.plot(time_hours, all_bg, color=color, label='Blood Glucose')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Thresholds
    ax1.axhline(y=180, color='r', linestyle='--', label='Hyperglycemia Threshold (180 mg/dL)')
    ax1.axhline(y=70, color='g', linestyle='--', label='Hypoglycemia Threshold (70 mg/dL)')
    
    # Insulin
    ax2 = ax1.twinx()  
    color = 'tab:orange'
    ax2.set_ylabel('Insulin Rate (U/hr)', color=color)
    ax2.plot(time_hours, all_insulin, color=color, alpha=0.7, label='Insulin Rate')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(f'PPO Performance: Hybrid Scenario (Fixed + Random) - {PATIENT_ID}')
    fig.tight_layout()
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.show()