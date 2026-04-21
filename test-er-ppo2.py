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


# ── UPDATE PATH ───────────────────────────────
model_path = r"models\ERPPO_20260327_183257\final_model"
norm_path  = r"models\ERPPO_20260327_183257\vec_normalize.pkl"

env = DummyVecEnv([make_env()])
env = VecNormalize.load(norm_path, env)
env.training    = False
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

# convert steps to hours (simglucose = 3 min per step)
time_hours = np.arange(len(glucose)) * 3 / 60

tir   = np.mean((glucose >= 70) & (glucose <= 180)) * 100
hypo  = np.mean(glucose < 70)  * 100
hyper = np.mean(glucose > 180) * 100

print("\n📊 --- ER-PPO RESULTS ---")
print(f"TIR:           {tir:.2f}%")
print(f"Hypo:          {hypo:.2f}%")
print(f"Hyper:         {hyper:.2f}%")
print(f"Mean Glucose:  {np.mean(glucose):.2f}")
print(f"Total Insulin: {np.sum(insulin):.4f}")

# ── DUAL AXIS PLOT (same style as PPO graph) ──
fig, ax1 = plt.subplots(figsize=(14, 6))

# glucose on left axis
ax1.plot(time_hours, glucose, color='steelblue',
         linewidth=1.5, label='Blood Glucose')
ax1.axhline(180, linestyle='--', color='red',
            linewidth=1.2, label='Hyperglycemia Threshold (180 mg/dL)')
ax1.axhline(70,  linestyle='--', color='green',
            linewidth=1.2, label='Hypoglycemia Threshold (70 mg/dL)')
ax1.fill_between(time_hours, 70, 180,
                 alpha=0.07, color='green')
ax1.set_ylabel("Blood Glucose (mg/dL)", color='steelblue', fontsize=11)
ax1.tick_params(axis='y', labelcolor='steelblue')
ax1.set_ylim(30, 280)
ax1.set_xlabel("Time (hours)", fontsize=11)
ax1.set_title(
    f"ER-PPO Model Performance for Patient: adult#001\n"
    f"TIR: {tir:.1f}%  |  Hypo: {hypo:.1f}%  |  Hyper: {hyper:.1f}%",
    fontsize=12
)

# insulin on right axis
ax2 = ax1.twinx()
ax2.plot(time_hours, insulin, color='orange',
         linewidth=1.2, alpha=0.85, label='Insulin Rate')
ax2.set_ylabel("Insulin Rate (U/min)", color='orange', fontsize=11)
ax2.tick_params(axis='y', labelcolor='orange')
ax2.set_ylim(-0.01, max(insulin) * 1.3)

# combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2,
           loc='upper right', fontsize=9)

plt.tight_layout()
plt.savefig("erppo_result.png", dpi=150)
plt.show()
print("📈 Saved to erppo_result.png")