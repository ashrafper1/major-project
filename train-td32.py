# train_td3.py
import gymnasium as gym
import simglucose
import os, time, random
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from simglucose.simulation.scenario import CustomScenario


# ─────────────────────────────────────────────
# TIR LOGGER
# ─────────────────────────────────────────────
class TIRLogCallback(BaseCallback):
    def __init__(self, freq=512):
        super().__init__()
        self.freq = freq
        self.timesteps, self.tir_vals, self.rew_vals = [], [], []
        self._bgs, self._rews = [], []

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "bg"      in info: self._bgs.append(info["bg"])
            if "episode" in info: self._rews.append(info["episode"]["r"])

        if self.num_timesteps % self.freq == 0 and self._bgs:
            arr = np.array(self._bgs)
            self.timesteps.append(self.num_timesteps)
            self.tir_vals.append(np.mean((arr >= 70) & (arr <= 180)) * 100)
            if self._rews:
                self.rew_vals.append(np.mean(self._rews[-50:]))
            self._bgs = []
        return True


# ─────────────────────────────────────────────
# MEAL WRAPPER
# ─────────────────────────────────────────────
class MealWrapper(gym.Wrapper):
    def reset(self, **kwargs):
        self.env.unwrapped.scenario = CustomScenario(
            start_time=0,
            scenario=[(random.randint(30, 120), random.randint(30, 80))]
        )
        return self.env.reset(**kwargs)


# ─────────────────────────────────────────────
# FEATURE WRAPPER
# exact same as ER-PPO so results are comparable
# ─────────────────────────────────────────────
class FeatureWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.last_insulin = 0.05
        self.prev_bg      = 120.0

        # same safe range as ER-PPO
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
        obs, _, terminated, truncated, info = self.env.step(
            np.array([insulin], dtype=np.float32)
        )
        bg = float(obs[0])

        # same clean bounded reward as ER-PPO
        if 70 <= bg <= 180:
            reward = 1.0
        elif 50 <= bg < 70 or 180 < bg <= 250:
            reward = -0.5
        else:
            reward = -2.0

        reward -= 0.05 * (insulin - self.last_insulin) ** 2
        reward  = float(np.clip(reward, -2.0, 1.0))

        self.prev_bg      = bg
        self.last_insulin = insulin
        info['bg']        = bg

        return self._augment(obs, info), reward, terminated, truncated, info


# ─────────────────────────────────────────────
# ENV FACTORY
# ─────────────────────────────────────────────
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
        env = Monitor(env)
        return env
    return _init


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":

    TIMESTEPS = 50_000

    run_name  = time.strftime("TD3_%Y%m%d_%H%M%S")
    model_dir = os.path.join("models", run_name)
    os.makedirs(model_dir, exist_ok=True)

    print("🚀 TD3 Training...")

    env = DummyVecEnv([make_env()])
    env = VecNormalize(env, norm_obs=True, norm_reward=False)

    n_actions    = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.01 * np.ones(n_actions)   # small noise — safe range
    )

    model = TD3(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        batch_size=256,
        buffer_size=100000,
        learning_starts=2000,  # collect 2k steps before learning
        gamma=0.99,
        tau=0.005,
        action_noise=action_noise,
        policy_delay=2,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1
    )

    tir_cb = TIRLogCallback(freq=512)

    model.learn(total_timesteps=TIMESTEPS, callback=tir_cb)

    model.save(os.path.join(model_dir, "final_model"))
    env.save(os.path.join(model_dir, "vec_normalize.pkl"))
    print(f"\n✅ DONE: {model_dir}")

    # ── TRAINING FIGURE ──────────────────────────────────
    if tir_cb.timesteps:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

        ax1.plot(tir_cb.timesteps, tir_cb.tir_vals,
                 color='steelblue', linewidth=2.5)
        ax1.axhline(75, linestyle='--', color='green',
                    linewidth=1.5, label='Target 75%')
        ax1.fill_between(
            tir_cb.timesteps, 75, tir_cb.tir_vals,
            where=[v >= 75 for v in tir_cb.tir_vals],
            alpha=0.2, color='green', label='Above target'
        )
        ax1.set_ylabel("Time-in-Range (%)", fontsize=11)
        ax1.set_ylim(0, 100)
        ax1.set_title("TD3 — Training Progress", fontsize=13)
        ax1.legend(fontsize=9)

        if tir_cb.rew_vals:
            n = min(len(tir_cb.timesteps), len(tir_cb.rew_vals))
            ax2.plot(tir_cb.timesteps[:n], tir_cb.rew_vals[:n],
                     color='coral', linewidth=2.5)
            ax2.set_ylabel("Mean Episode Reward", fontsize=11)
            ax2.axhline(0, linestyle='--', color='gray', alpha=0.5)

        ax2.set_xlabel("Timesteps", fontsize=11)
        plt.tight_layout()
        fig.savefig(os.path.join(model_dir, "training_curve.png"), dpi=150)
        plt.show()
        print("📈 Saved.")