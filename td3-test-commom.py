# test_all_td3.py
import gymnasium as gym
import simglucose
import numpy as np
import os
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


# ── 1D wrapper ────────────────────────────────
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


# ── 3D wrapper ────────────────────────────────
class FeatureWrapper3D(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Box(
            low=np.array([0.0]), high=np.array([1.6]), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, 0.0, 0.0]),
            high=np.array([600.0, 100.0, 300.0]),
            dtype=np.float32
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        iob = float(info.get('IOB', 0.0))
        cob = float(info.get('CHO', 0.0))
        return np.array([float(obs[0]), iob, cob], dtype=np.float32), info

    def step(self, action):
        insulin = float(np.clip(action[0], 0.0, 1.6))
        obs, reward, terminated, truncated, info = self.env.step(
            np.array([insulin], dtype=np.float32)
        )
        iob = float(info.get('IOB', 0.0))
        cob = float(info.get('CHO', 0.0))
        info['bg'] = float(obs[0])
        return np.array([float(obs[0]), iob, cob], dtype=np.float32), reward, terminated, truncated, info


# ── 4D wrapper ────────────────────────────────
class FeatureWrapper4D(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev_bg      = 120.0
        self.last_insulin = 0.0
        self.action_space = gym.spaces.Box(
            low=np.array([0.0]), high=np.array([1.6]), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=np.array([  0.0, -50.0, 0.0,   0.0]),
            high=np.array([600.0,  50.0, 10.0, 100.0]),
            dtype=np.float32
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_bg = float(obs[0])
        bg       = float(np.clip(obs[0], 0, 600))
        delta_bg = 0.0
        iob      = float(np.clip(info.get('IOB', 0.0), 0, 10))
        cob      = float(np.clip(info.get('CHO', 0.0), 0, 100))
        return np.array([bg, delta_bg, iob, cob], dtype=np.float32), info

    def step(self, action):
        insulin = float(np.clip(action[0], 0.0, 1.6))
        obs, reward, terminated, truncated, info = self.env.step(
            np.array([insulin], dtype=np.float32)
        )
        bg       = float(np.clip(obs[0], 0, 600))
        delta_bg = float(np.clip(bg - self.prev_bg, -50, 50))
        iob      = float(np.clip(info.get('IOB', 0.0), 0, 10))
        cob      = float(np.clip(info.get('CHO', 0.0), 0, 100))
        self.prev_bg      = bg
        self.last_insulin = insulin
        info['bg']        = bg
        return np.array([bg, delta_bg, iob, cob], dtype=np.float32), reward, terminated, truncated, info


def make_env(wrapper_cls):
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
        env = wrapper_cls(env)
        return env
    return _init


def test_model(model_dir):
    model_path = os.path.join(model_dir, "final_model")
    norm_path  = os.path.join(model_dir, "vec_normalize.pkl")

    if not os.path.exists(norm_path):
        print(f"  ⚠️  Skipping — vec_normalize.pkl not found")
        return None

    # try all 3 wrappers until one works
    for wrapper_cls in [FeatureWrapper4D, FeatureWrapper3D, FeatureWrapper1D]:
        try:
            env = DummyVecEnv([make_env(wrapper_cls)])
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

            glucose = np.array(glucose)
            insulin = np.array(insulin)

            tir   = np.mean((glucose >= 70) & (glucose <= 180)) * 100
            hypo  = np.mean(glucose < 70)  * 100
            hyper = np.mean(glucose > 180) * 100

            obs_dim = wrapper_cls.__name__
            return {
                "name":     os.path.basename(model_dir),
                "obs":      obs_dim,
                "tir":      tir,
                "hypo":     hypo,
                "hyper":    hyper,
                "mean_bg":  np.mean(glucose),
                "glucose":  glucose,
                "insulin":  insulin,
            }

        except Exception:
            continue

    print(f"  ❌ All wrappers failed for {os.path.basename(model_dir)}")
    return None


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
models_root = "models"
td3_dirs = sorted([
    os.path.join(models_root, d)
    for d in os.listdir(models_root)
    if d.startswith("TD3") and os.path.isdir(os.path.join(models_root, d))
])

print(f"Found {len(td3_dirs)} TD3 models\n")

results = []
for d in td3_dirs:
    print(f"Testing: {os.path.basename(d)}")
    r = test_model(d)
    if r:
        results.append(r)
        print(f"  ✅ [{r['obs']}] TIR: {r['tir']:.1f}%  Hypo: {r['hypo']:.1f}%  Hyper: {r['hyper']:.1f}%")

# ── SUMMARY TABLE ─────────────────────────────
print("\n" + "="*70)
print(f"{'Model':<38} {'Obs':>15} {'TIR':>6} {'Hypo':>6} {'Hyper':>7}")
print("="*70)
for r in sorted(results, key=lambda x: x['tir'], reverse=True):
    print(f"{r['name']:<38} {r['obs']:>15} {r['tir']:>5.1f}% {r['hypo']:>5.1f}% {r['hyper']:>6.1f}%")
print("="*70)

# ── BEST MODEL PLOT ───────────────────────────
if results:
    best = max(results, key=lambda x: x['tir'])
    print(f"\n🏆 Best: {best['name']}  TIR={best['tir']:.1f}%")

    glucose    = best['glucose']
    insulin    = best['insulin']
    time_hours = np.arange(len(glucose)) * 3 / 60

    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax1.plot(time_hours, glucose, color='steelblue', linewidth=1.5, label='Blood Glucose')
    ax1.axhline(180, linestyle='--', color='red',   linewidth=1.2, label='Hyperglycemia (180 mg/dL)')
    ax1.axhline(70,  linestyle='--', color='green', linewidth=1.2, label='Hypoglycemia (70 mg/dL)')
    ax1.fill_between(time_hours, 70, 180, alpha=0.07, color='green')
    ax1.set_ylabel("Blood Glucose (mg/dL)", color='steelblue', fontsize=11)
    ax1.tick_params(axis='y', labelcolor='steelblue')
    ax1.set_ylim(30, 350)
    ax1.set_xlabel("Time (hours)", fontsize=11)
    ax1.set_title(
        f"Best TD3: {best['name']}\n"
        f"TIR: {best['tir']:.1f}%  Hypo: {best['hypo']:.1f}%  Hyper: {best['hyper']:.1f}%",
        fontsize=12
    )
    ax2 = ax1.twinx()
    ax2.plot(time_hours, insulin, color='orange', linewidth=1.2, alpha=0.85, label='Insulin Rate')
    ax2.set_ylabel("Insulin Rate (U/min)", color='orange', fontsize=11)
    ax2.tick_params(axis='y', labelcolor='orange')
    ax2.set_ylim(-0.01, max(insulin) * 1.4 if max(insulin) > 0 else 1)

    l1, lb1 = ax1.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lb1 + lb2, loc='upper right', fontsize=9)

    plt.tight_layout()
    plt.savefig("best_td3_result.png", dpi=150)
    plt.show()

    # ── COMPARISON BAR CHART ──────────────────────
    fig2, ax = plt.subplots(figsize=(12, 5))
    names  = [r['name'].split('_20')[0] for r in results]
    tirs   = [r['tir'] for r in results]
    colors = ['green' if t >= 75 else 'steelblue' for t in tirs]
    bars   = ax.bar(names, tirs, color=colors, edgecolor='white')
    ax.axhline(75, linestyle='--', color='red', linewidth=1.5, label='Target 75%')
    ax.set_ylabel("Time-in-Range (%)", fontsize=11)
    ax.set_title("TD3 All Models — TIR Comparison", fontsize=13)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=9)
    plt.xticks(rotation=30, ha='right', fontsize=8)
    for bar, tir in zip(bars, tirs):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 1,
                f'{tir:.1f}%', ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig("td3_comparison.png", dpi=150)
    plt.show()
    print("📈 Saved: best_td3_result.png + td3_comparison.png")