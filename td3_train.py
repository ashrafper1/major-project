import gymnasium as gym
import simglucose
import os
from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np
import time
import random
from collections import deque
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


# ---------------- FINAL TD3 WRAPPER ----------------
class FinalTD3Wrapper(gym.Wrapper):
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
        # ✅ safe insulin
        action = np.clip(action, 0, 1.3)

        bg_before = self.last_bg

        # ✅ hypo safety shield
        if bg_before < 80:
            action = np.array([0.0])

        # ✅ smoother insulin
        smoothed = 0.6 * self.last_action + 0.4 * action[0]
        smoothed_action = np.array([smoothed])

        obs, _, terminated, truncated, info = self.env.step(smoothed_action)
        bg = obs[0]

        self.insulin_history.append(smoothed)

        # ---------------- FINAL CORRECTED REWARD ----------------
        # Positive reward structure (CRITICAL FIX)

        if 90 <= bg <= 150:
            reward = 2.0
        elif 70 <= bg < 90 or 150 < bg <= 180:
            reward = 1.0
        else:
            reward = -1.0

        # strong penalties (only extreme cases)
        if bg < 50:
            reward -= 3.0
        if bg > 250:
            reward -= 2.0

        # ROC penalty (discourage rising glucose)
        delta = bg - bg_before
        reward -= 0.01 * max(delta, 0)

        # smooth insulin penalty
        reward -= 0.005 * ((smoothed - self.last_action) ** 2)

        # clip for stability
        reward = np.clip(reward, -5, 2)

        # ❗ DO NOT TERMINATE
        terminated = False

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
        env = FinalTD3Wrapper(env)
        env = Monitor(env)

        return env

    return _init


# ---------------- MAIN ----------------
if __name__ == "__main__":

    TIMESTEPS = 15000

    run_name = time.strftime("TD3_FINAL_FIXED_%Y%m%d_%H%M%S")
    model_dir = os.path.join("models", run_name)
    os.makedirs(model_dir, exist_ok=True)

    print(f"🚀 FINAL TD3 RUN (FIXED REWARD): {run_name}")
    print(f"   Timesteps: {TIMESTEPS}")

    env = DummyVecEnv([make_env()])
    env = VecNormalize(env, norm_obs=True, norm_reward=False)

    n_actions = env.action_space.shape[-1]

    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.05 * np.ones(n_actions)
    )

    model = TD3(
        "MlpPolicy",
        env,
        learning_rate=5e-5,
        batch_size=256,
        buffer_size=100000,
        learning_starts=1000,
        gamma=0.99,
        tau=0.005,
        action_noise=action_noise,
        policy_delay=2,
        train_freq=(1, "step"),
        gradient_steps=1,
        verbose=1
    )

    model.learn(total_timesteps=TIMESTEPS)

    model.save(os.path.join(model_dir, "final_model"))
    model.get_env().save(os.path.join(model_dir, "vec_normalize.pkl"))

    print(f"\n✅ DONE: {model_dir}")