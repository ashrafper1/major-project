import gymnasium as gym
import simglucose
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import time
import numpy as np
import random
from typing import Callable
from simglucose.simulation.scenario import CustomScenario

# ==========================================
# 1. SCENARIO RANDOMIZER
# ==========================================
class RandomizedScenarioWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def _create_random_scenario(self):
        meal_time = random.randint(30, 120)
        meal_amount = random.randint(30, 100)
        return CustomScenario(start_time=0, scenario=[(meal_time, meal_amount)])

    def reset(self, **kwargs):
        new_scenario = self._create_random_scenario()
        self.env.unwrapped.scenario = new_scenario
        return self.env.reset(**kwargs)

# ==========================================
# 2. INTELLIGENT WRAPPER ("LIVING HELL" LOGIC)
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
        # 1. SAFETY SHIELD (Low side only)
        # Prevents immediate death so training continues
        if self.last_bg < 110:
            action = np.array([0.0], dtype=np.float32)

        obs, reward, terminated, truncated, info = self.env.step(action)
        current_bg = obs[0]
        
        # 2. LIVING HELL LOGIC (High side)
        # We REMOVED 'terminated = True'. 
        # Instead, we apply 'punishment' via the reward function below.
        # This allows the agent to live through the high spike and learn to fix it.
        
        augmented_obs = self._augment_observation(obs, info)
        self.last_bg = current_bg
        
        return augmented_obs, reward, terminated, truncated, info

# ==========================================
# 3. SYMMETRIC PUNISHMENT REWARD
# ==========================================
def symmetric_punishment_reward(bg_history, **kwargs):
    bg = bg_history[-1]
    
    # 1. THE SAFE ZONE
    if 110 <= bg <= 150:
        return 1.0
    
    # 2. THE PAIN ZONES
    target = 130.0
    dist = abs(bg - target)
    
    # Quadratic Penalty
    # If BG > 300, dist is >170.
    # Penalty = (170^2) / 100 = 289. 
    # This is a HUGE penalty per step. The agent will be screaming.
    penalty = (dist ** 2) / 100.0
    
    return -penalty

# ==========================================
# 4. SETUP
# ==========================================
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

def make_env(patient_name_list):
    def _init():
        patient_name = random.choice(patient_name_list)
        if 'simglucose-v0' not in gym.envs.registry:
            gym.register(id='simglucose-v0', entry_point='simglucose.envs.simglucose_gym_env:T1DSimGymnaisumEnv')

        # Reward function handles the pain. No early termination logic here.
        env = gym.make('simglucose-v0', patient_name=patient_name, reward_fun=symmetric_punishment_reward, max_episode_steps=1000)
        env = RandomizedScenarioWrapper(env)
        env = SimglucoseAdvantageWrapper(env)
        env = Monitor(env)
        return env
    return _init

# ==========================================
# 5. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    PATIENT_COHORT = [f'adult#{i:03}' for i in range(1, 11)]
    NUM_ENV = 4
    TIMESTEPS = 150000 
    
    experiment_timestamp = time.strftime("living_hell_%Y%m%d-%H%M%S")
    log_dir = os.path.join("logs", experiment_timestamp)
    model_dir = os.path.join("models", experiment_timestamp)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"--- Initializing 'Living Hell' Protocol: {experiment_timestamp} ---")
    print("Logic: Symmetric Pain | Survive > 300 (but suffer) | No Action Penalty")

    # Env
    env = DummyVecEnv([make_env(PATIENT_COHORT) for i in range(NUM_ENV)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=1000., gamma=0.995)

    # Eval
    eval_env = DummyVecEnv([make_env(['adult#001'])])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=1000., gamma=0.995)
    
    eval_callback = EvalCallback(eval_env, best_model_save_path=model_dir,
                                 log_path=log_dir, eval_freq=5000,
                                 deterministic=True, render=False)

    # Model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=linear_schedule(0.0003), 
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.995,
        ent_coef=0.01,
        max_grad_norm=0.5
    )

    print(f"\n--- Starting Training ---")
    model.learn(total_timesteps=TIMESTEPS, callback=eval_callback)
    print("--- Training Finished ---")

    model.save(os.path.join(model_dir, "final_model"))
    env.save(os.path.join(model_dir, "vec_normalize.pkl"))
    print(f"Model saved to {model_dir}")

    env.close()
    eval_env.close()