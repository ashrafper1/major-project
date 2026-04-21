import gymnasium as gym
import simglucose
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from collections import deque
import random
import matplotlib.pyplot as plt
from simglucose.simulation.scenario import CustomScenario


# ---------------- RANDOM MEAL ----------------
class RandomScenario(gym.Wrapper):
    def reset(self, **kwargs):
        meal_time = random.randint(30, 120)
        meal_amount = random.randint(30, 80)
        self.env.unwrapped.scenario = CustomScenario(
            start_time=0,
            scenario=[(meal_time, meal_amount)]
        )
        return self.env.reset(**kwargs)


# ---------------- SAME WRAPPER ----------------
class EvalWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.last_bg = 140.0
        self.last_action = 0.0
        self.insulin_history = deque(maxlen=6)

        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, -100.0, 0.0]),
            high=np.array([1000.0, 100.0, 10.0]),
            dtype=np.float32
        )

    def _get_obs(self, bg):
        roc = bg - self.last_bg
        iob = sum(self.insulin_history)
        return np.array([bg, roc, iob], dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_bg = obs[0]
        self.last_action = 0.0
        self.insulin_history.clear()
        return self._get_obs(obs[0]), info

    def step(self, action):
        action = np.clip(action, 0, 1.3)

        if self.last_bg < 80:
            action = np.array([0.0])

        smoothed = 0.6 * self.last_action + 0.4 * action[0]
        action = np.array([smoothed])

        obs, reward, terminated, truncated, info = self.env.step(action)
        bg = obs[0]

        self.insulin_history.append(smoothed)

        self.last_bg = bg
        self.last_action = smoothed

        return self._get_obs(bg), reward, terminated, truncated, info


# ---------------- ENV ----------------
def make_env():
    def _init():
        if 'simglucose-v0' not in gym.envs.registry:
            gym.register(
                id='simglucose-v0',
                entry_point='simglucose.envs.simglucose_gym_env:T1DSimGymnaisumEnv'
            )

        env = gym.make(
            'simglucose-v0',
            patient_name='adult#001',
            max_episode_steps=300
        )

        env = RandomScenario(env)
        env = EvalWrapper(env)

        return env
    return _init


# ---------------- PATHS ----------------
model_path = r"models\TD3_FINAL_FIXED_20260327_144334\final_model"
norm_path = r"models\TD3_FINAL_FIXED_20260327_144334\vec_normalize.pkl"


# ---------------- LOAD ENV ----------------
env = DummyVecEnv([make_env()])

# 🔥 CRITICAL FIX
env = VecNormalize.load(norm_path, env)
env.training = False
env.norm_reward = False

model = TD3.load(model_path)


# ---------------- RUN ----------------
obs = env.reset()

glucose_values = []
insulin_values = []

for _ in range(300):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)

    # 🔥 UNNORMALIZE OBS (VERY IMPORTANT)
    real_obs = env.get_original_obs()

    bg = real_obs[0][0]

    glucose_values.append(bg)
    insulin_values.append(action[0][0])


glucose_values = np.array(glucose_values)
insulin_values = np.array(insulin_values)


# ---------------- METRICS ----------------
tir = np.mean((glucose_values >= 70) & (glucose_values <= 180)) * 100
hypo = np.mean(glucose_values < 70) * 100
hyper = np.mean(glucose_values > 180) * 100

print("\n📊 --- TD3 FINAL RESULTS ---")
print(f"TIR: {tir:.2f}%")
print(f"Hypo: {hypo:.2f}%")
print(f"Hyper: {hyper:.2f}%")
print(f"Mean Glucose: {np.mean(glucose_values):.2f}")
print(f"Total Insulin: {np.sum(insulin_values):.2f}")


# ---------------- GRAPH ----------------
plt.figure()
plt.plot(glucose_values)
plt.axhline(70, linestyle='--')
plt.axhline(180, linestyle='--')
plt.title("TD3 Glucose Control")
plt.xlabel("Time Step")
plt.ylabel("Blood Glucose")
plt.show()