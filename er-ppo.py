# er_ppo.py
import gymnasium as gym
import simglucose
import os, time, random
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from simglucose.simulation.scenario import CustomScenario


class AdaptiveEntropyCallback(BaseCallback):
    def __init__(self, ent_start=0.05, ent_end=0.001, total=50000):
        super().__init__()
        self.ent_start = ent_start
        self.ent_end   = ent_end
        self.total     = total
        self.rbuf      = []
        self.prev      = None

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.rbuf.append(info["episode"]["r"])

        p   = self.num_timesteps / self.total
        ent = self.ent_start * ((self.ent_end / self.ent_start) ** p)

        if len(self.rbuf) >= 20:
            curr = np.mean(self.rbuf[-20:])
            if self.prev is not None and curr < self.prev * 0.88:
                ent = min(ent * 2.5, self.ent_start)
            self.prev = curr

        self.model.ent_coef = float(ent)
        return True


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

        # ✅ action max 0.2 — verified safe range from verify.py
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


if __name__ == "__main__":

    TIMESTEPS = 50_000
    N_ENVS    = 4

    run_name  = time.strftime("ERPPO_%Y%m%d_%H%M%S")
    model_dir = os.path.join("models", run_name)
    os.makedirs(model_dir, exist_ok=True)

    print("🚀 ER-PPO (PPO + Adaptive Entropy)")

    env = DummyVecEnv([make_env() for _ in range(N_ENVS)])
    env = VecNormalize(env, norm_obs=True, norm_reward=False)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,
        vf_coef=0.25,
        max_grad_norm=0.5,
        use_sde=True,
        sde_sample_freq=4,
        # ✅ small init std — prevents overdose at start
        policy_kwargs=dict(
            net_arch=[256, 256],
            log_std_init=-2.0
        ),
        verbose=1
    )

    ent_cb = AdaptiveEntropyCallback(
        ent_start=0.05,
        ent_end=0.001,
        total=TIMESTEPS
    )
    tir_cb = TIRLogCallback(freq=512)

    model.learn(
        total_timesteps=TIMESTEPS,
        callback=[ent_cb, tir_cb],
        reset_num_timesteps=True
    )

    model.save(os.path.join(model_dir, "final_model"))
    env.save(os.path.join(model_dir, "vec_normalize.pkl"))
    print(f"\n✅ DONE: {model_dir}")

    if tir_cb.timesteps:
        fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

        axes[0].plot(tir_cb.timesteps, tir_cb.tir_vals,
                     color='steelblue', linewidth=2.5)
        axes[0].axhline(75, linestyle='--', color='green',
                        linewidth=1.5, label='Target 75%')
        axes[0].fill_between(
            tir_cb.timesteps, 75, tir_cb.tir_vals,
            where=[v >= 75 for v in tir_cb.tir_vals],
            alpha=0.2, color='green', label='Above target'
        )
        axes[0].set_ylabel("Time-in-Range (%)", fontsize=11)
        axes[0].set_ylim(0, 100)
        axes[0].set_title("ER-PPO — Adaptive Entropy Training", fontsize=13)
        axes[0].legend(fontsize=9)

        if tir_cb.rew_vals:
            n = min(len(tir_cb.timesteps), len(tir_cb.rew_vals))
            axes[1].plot(tir_cb.timesteps[:n], tir_cb.rew_vals[:n],
                         color='coral', linewidth=2.5)
            axes[1].set_ylabel("Mean Episode Reward", fontsize=11)
            axes[1].axhline(0, linestyle='--', color='gray', alpha=0.5)

        axes[2].plot(
            tir_cb.timesteps,
            [0.05 * (0.02) ** (t / TIMESTEPS) for t in tir_cb.timesteps],
            color='purple', linewidth=2
        )
        axes[2].set_ylabel("Entropy Coef", fontsize=11)
        axes[2].set_xlabel("Timesteps", fontsize=11)

        plt.tight_layout()
        fig.savefig(os.path.join(model_dir, "training_curve.png"), dpi=150)
        plt.show()
        print("📈 Saved.")