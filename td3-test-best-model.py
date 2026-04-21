# test_td3_best.py
import gymnasium as gym
import simglucose
import numpy as np
import random
import matplotlib.pyplot as plt
from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from simglucose.simulation.scenario import CustomScenario


class MealWrapper(gym.Wrapper):
    def reset(self, **kwargs):
        self.env.unwrapped.scenario = CustomScenario(
            start_time=0,
            scenario=[(random.randint(30, 120), random.randint(30, 80))]
        )
        return self.env.reset(**kwargs)


class FeatureWrapper1D(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Box(
            low=np.array([0.0]), high=np.array([1.6]), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0]), high=np.array([600.0]), dtype=np.float32
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return np.array([float(obs[0])], dtype=np.float32), info

    def step(self, action):
        insulin = float(np.clip(action[0], 0.0, 1.6))
        obs, reward, terminated, truncated, info = self.env.step(
            np.array([insulin], dtype=np.float32)
        )
        info['bg'] = float(obs[0])
        return np.array([float(obs[0])], dtype=np.float32), reward, terminated, truncated, info


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
        env = FeatureWrapper1D(env)
        return env
    return _init


# ── PATH ─────────────────────────────────────
model_path = r"models\TD3_FINAL_STABLE_20260327_103858\final_model"
norm_path  = r"models\TD3_FINAL_STABLE_20260327_103858\vec_normalize.pkl"

env = DummyVecEnv([make_env()])
env = VecNormalize.load(norm_path, env)
env.training    = False
env.norm_reward = False

model = TD3.load(model_path)

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

glucose    = np.array(glucose)
insulin    = np.array(insulin)
time_hours = np.arange(len(glucose)) * 3 / 60

tir   = np.mean((glucose >= 70) & (glucose <= 180)) * 100
hypo  = np.mean(glucose < 70)  * 100
hyper = np.mean(glucose > 180) * 100

print("\n📊 --- TD3 RESULTS ---")
print(f"Model:         TD3_FINAL_STABLE_20260327_103858")
print(f"TIR:           {tir:.2f}%")
print(f"Hypo:          {hypo:.2f}%")
print(f"Hyper:         {hyper:.2f}%")
print(f"Mean Glucose:  {np.mean(glucose):.2f}")
print(f"Min Glucose:   {np.min(glucose):.2f}")
print(f"Max Glucose:   {np.max(glucose):.2f}")
print(f"Total Insulin: {np.sum(insulin):.4f}")

fig, ax1 = plt.subplots(figsize=(14, 6))

ax1.plot(time_hours, glucose, color='steelblue',
         linewidth=1.5, label='Blood Glucose')
ax1.axhline(180, linestyle='--', color='red',
            linewidth=1.2, label='Hyperglycemia Threshold (180 mg/dL)')
ax1.axhline(70,  linestyle='--', color='green',
            linewidth=1.2, label='Hypoglycemia Threshold (70 mg/dL)')
ax1.fill_between(time_hours, 70, 180, alpha=0.07, color='green')
ax1.set_ylabel("Blood Glucose (mg/dL)", color='steelblue', fontsize=11)
ax1.tick_params(axis='y', labelcolor='steelblue')
ax1.set_ylim(30, 350)
ax1.set_xlabel("Time (hours)", fontsize=11)
ax1.set_title(
    f"TD3 Model Performance for Patient: adult#001\n"
    f"TIR: {tir:.1f}%  |  Hypo: {hypo:.1f}%  |  Hyper: {hyper:.1f}%",
    fontsize=12
)

ax2 = ax1.twinx()
ax2.plot(time_hours, insulin, color='orange',
         linewidth=1.2, alpha=0.85, label='Insulin Rate')
ax2.set_ylabel("Insulin Rate (U/min)", color='orange', fontsize=11)
ax2.tick_params(axis='y', labelcolor='orange')
ax2.set_ylim(-0.01, max(insulin) * 1.4 if max(insulin) > 0 else 1)

l1, lb1 = ax1.get_legend_handles_labels()
l2, lb2 = ax2.get_legend_handles_labels()
ax1.legend(l1 + l2, lb1 + lb2, loc='upper right', fontsize=9)

plt.tight_layout()
plt.savefig("td3_final_stable_result.png", dpi=150)
plt.show()
print("📈 Saved: td3_final_stable_result.png")