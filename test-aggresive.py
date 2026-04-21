import gymnasium as gym
import simglucose
import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from simglucose.simulation.scenario import CustomScenario

# ==========================================
# 1. WRAPPERS
# ==========================================
class SimglucoseAdvantageWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.last_bg = 140.0 
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, 0.0, 0.0, -50.0]),
            high=np.array([1000.0, 100.0, 300.0, 50.0]),
            dtype=np.float32
        )

    def _augment_observation(self, obs, info):
        current_bg = obs[0]
        iob = info.get('IOB', 0.0)
        cob = info.get('CHO', 0.0)
        roc = current_bg - self.last_bg
        return np.array([current_bg, iob, cob, roc], dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_bg = obs[0]
        return self._augment_observation(obs, info), info

    def step(self, action):
        if self.last_bg < 110:
            action = np.array([0.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = self.env.step(action)
        augmented_obs = self._augment_observation(obs, info)
        self.last_bg = obs[0]
        return augmented_obs, reward, terminated, truncated, info

class FixedScenarioWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
    def reset(self, **kwargs):
        new_scenario = CustomScenario(start_time=0, scenario=[(60, 80)])
        self.env.unwrapped.scenario = new_scenario
        return self.env.reset(**kwargs)

# ==========================================
# 2. RUN TEST
# ==========================================
def run_test():
    # --- REGISTRATION ---
    if 'simglucose-v0' not in gym.envs.registry:
        gym.register(id='simglucose-v0', entry_point='simglucose.envs.simglucose_gym_env:T1DSimGymnaisumEnv')

    # --- AUTO-DETECT MODEL ---
    models_root = "models"
    try:
        candidates = [os.path.join(models_root, d) for d in os.listdir(models_root) if d.startswith("aggressive_v1")]
        latest_folder = max(candidates, key=os.path.getmtime)
        print(f"--- Testing Model: {latest_folder} ---")
        model_path = os.path.join(latest_folder, "final_model")
        stats_path = os.path.join(latest_folder, "vec_normalize.pkl")
    except Exception as e:
        print(f"Error finding model: {e}")
        return

    # --- MAKE ENV ---
    def make_env():
        # Note: We must re-register inside the lambda scope sometimes if using multiprocessing, 
        # but for DummyVecEnv this scope is fine.
        env = gym.make('simglucose-v0', patient_name="adult#004", reward_fun=lambda x, **k: 0)
        env = FixedScenarioWrapper(env)
        env = SimglucoseAdvantageWrapper(env)
        return env

    env = DummyVecEnv([make_env])
    
    # LOAD STATS
    try:
        env = VecNormalize.load(stats_path, env)
        env.training = False
        env.norm_reward = False
    except Exception as e:
        print(f"Error loading stats: {e}")
        return

    # LOAD MODEL
    model = PPO.load(model_path)

    # RUN
    obs = env.reset()
    all_bg = []
    all_insulin = []
    
    for i in range(500):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)
        
        real_bg = env.envs[0].last_bg
        all_bg.append(real_bg)
        all_insulin.append(action[0][0])
        
        if done[0]:
            break
            
    # VISUALIZE
    bg_np = np.array(all_bg)
    tir = np.sum((bg_np >= 70) & (bg_np <= 180)) / len(bg_np) * 100
    mean_bg = np.mean(bg_np)
    print(f"Time in Range: {tir:.2f}%")
    print(f"Mean BG: {mean_bg:.1f}")

    plt.figure(figsize=(10, 8))
    
    plt.subplot(2,1,1)
    plt.plot(all_bg, label="Glucose", linewidth=2)
    plt.axhline(70, c='red', ls='--', label="Hypo")
    plt.axhline(180, c='orange', ls='--', label="Hyper")
    plt.fill_between(range(len(all_bg)), 70, 180, color='green', alpha=0.1)
    plt.axvline(x=20, c='brown', label="Meal (80g)")
    plt.title("Aggressive Model Performance")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2,1,2)
    plt.bar(range(len(all_insulin)), all_insulin, color='purple', width=1)
    plt.title("Insulin Actions")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_test()