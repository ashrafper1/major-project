# test_erppo.py
import gymnasium as gym
import simglucose
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import random
import matplotlib.pyplot as plt
from simglucose.simulation.scenario import CustomScenario


class MealWrapper(gym.Wrapper):
    def reset(self, **kwargs):
        self.env.unwrapped.scenario = CustomScenario(
            start_time=0,
            scenario=[(random.randint(30, 120), random.randint(30, 80))]
        )
        return self.env.reset(**kwargs)


class FeatureWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.last_insulin = 0.05
        self.prev_bg      = 120.0

        self.action_space = gym.spaces.Box(
            low=np.array([0.0]),
            high=np.array([0.2]),
            dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=np.array([  0.0, -50.0, 0.0,   0.0]),
            high=np.array([600.0,  50.0, 10.0, 100.0]),
            dtype=np.float32
        )

    def _augment(self, obs, info):
        bg       = float(np.clip(obs[0], 0, 600))
        delta_bg = float(np.clip(bg - self.prev_bg, -50, 50))
        iob      = float(np.clip(info.get('IOB', 0.0), 0, 10))
        cob      = float(np.clip(info.get('CHO', 0.0), 0, 100))
        return np.array([bg, delta_bg, iob, cob], dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_insulin = 0.05
        self.prev_bg      = float(obs[0])
        return self._augment(obs, info), info

    def step(self, action):
        insulin = float(np.clip(action[0], 0.0, 0.2))
        obs, reward, terminated, truncated, info = self.env.step(
            np.array([insulin], dtype=np.float32)
        )
        self.prev_bg      = float(obs[0])
        self.last_insulin = insulin
        info['bg']        = float(obs[0])
        return self._augment(obs, info), reward, terminated, truncated, info


def make_env():
    def _init():
        if 'simglucose-v0' not in gym.envs.registry:
            gym.register(
                id='simglucose-v0',
                entry_point='simglucose.envs.simglucose_gym_env:T1DSimGymnaisumEnv'
            )
        env = gym.make('simglucose-v0',
                       patient_name='adult#001',
                       max_episode_steps=1000)
        env = MealWrapper(env)
        env = FeatureWrapper(env)
        return env
    return _init


# ── UPDATE THIS PATH ──────────────────────────
model_path = r"models\ERPPO_20260327_183257\final_model"
norm_path  = r"models\ERPPO_20260327_183257\vec_normalize.pkl"

env = DummyVecEnv([make_env()])
env = VecNormalize.load(norm_path, env)
env.training   = False
env.norm_reward = False

model = PPO.load(model_path)

obs = env.reset()
glucose, insulin = [], []

for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, _, dones, _ = env.step(action)
    real = env.get_original_obs()
    glucose.append(real[0][0])
    insulin.append(action[0][0])

    if dones[0]:
        obs = env.reset()

glucose = np.array(glucose)
insulin = np.array(insulin)

tir   = np.mean((glucose >= 70) & (glucose <= 180)) * 100
hypo  = np.mean(glucose < 70)  * 100
hyper = np.mean(glucose > 180) * 100

print("\n📊 --- ER-PPO RESULTS ---")
print(f"TIR:           {tir:.2f}%")
print(f"Hypo:          {hypo:.2f}%")
print(f"Hyper:         {hyper:.2f}%")
print(f"Mean Glucose:  {np.mean(glucose):.2f}")
print(f"Total Insulin: {np.sum(insulin):.4f}")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

ax1.plot(glucose, color='steelblue', linewidth=1.5)
ax1.axhline(70,  linestyle='--', color='orange', label='Hypo (70)')
ax1.axhline(180, linestyle='--', color='red',    label='Hyper (180)')
ax1.fill_between(range(len(glucose)), 70, 180,
                 alpha=0.08, color='green', label='TIR zone')
ax1.set_ylabel("Blood Glucose (mg/dL)")
ax1.set_title(f"ER-PPO  |  TIR: {tir:.1f}%  Hypo: {hypo:.1f}%  Hyper: {hyper:.1f}%")
ax1.legend(fontsize=8)

ax2.plot(insulin, color='coral', linewidth=1.5)
ax2.set_ylabel("Insulin (U/min)")
ax2.set_xlabel("Time Step")

plt.tight_layout()
plt.show()