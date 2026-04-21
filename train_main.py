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

# ---------------- FEATURE WRAPPER ----------------
class FeatureWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.last_action = 0.0

        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, 0.0, 0.0]),
            high=np.array([1000.0, 100.0, 300.0]),
            dtype=np.float32
        )

    def _augment(self, obs, info):
        iob = info.get('IOB', 0.0)
        cob = info.get('CHO', 0.0)
        return np.array([obs[0], iob, cob], dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_action = 0.0
        return self._augment(obs, info), info

    def step(self, action):
        # ✅ safer insulin
        action = np.clip(action, 0, 1.6)

        obs, reward, terminated, truncated, info = self.env.step(action)

        bg = obs[0]

        # ✅ PPO-style Gaussian reward
        mu, sigma = 115, 20
        reward = 0.05 * np.exp(-0.5 * ((bg - mu) / sigma) ** 2)

        if bg < 70:
            reward -= 0.0001 * 5.0 * ((bg - 70) ** 2)
        elif bg > 180:
            reward -= 0.0001 * 2.0 * ((bg - 180) ** 2)

        # ✅ smooth insulin
        reward -= 0.01 * ((action[0] - self.last_action) ** 2)
        self.last_action = action[0]

        return self._augment(obs, info), reward, terminated, truncated, info

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
            max_episode_steps=1000   # ✅ long horizon (important)
        )

        env = RandomScenario(env)
        env = FeatureWrapper(env)
        env = Monitor(env)
        return env
    return _init

# ---------------- MAIN ----------------
if __name__ == "__main__":

    TIMESTEPS = 25000

    run_name = time.strftime("TD3_FINAL_PROPER_%Y%m%d_%H%M%S")
    model_dir = os.path.join("models", run_name)
    os.makedirs(model_dir, exist_ok=True)

    print("🚀 Training FINAL TD3 (IOB + COB)...")

    env = DummyVecEnv([make_env()])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.02 * np.ones(n_actions)
    )

    model = TD3(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        batch_size=256,
        buffer_size=100000,
        learning_starts=2000,
        gamma=0.99,
        tau=0.005,
        action_noise=action_noise,
        policy_delay=2,
        verbose=1
    )

    model.learn(total_timesteps=TIMESTEPS)

    model.save(os.path.join(model_dir, "final_model"))
    model.get_env().save(os.path.join(model_dir, "vec_normalize.pkl"))

    print(f"\n✅ DONE: {model_dir}")