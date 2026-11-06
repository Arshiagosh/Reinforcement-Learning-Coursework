"""Tabular Q-learning on MountainCar-v0 (HW5 Q7).

The continuous state (position, velocity) is binned on a 20 x 20 grid.
The submitted version mapped bins with `pos * 20 + vel` although
np.digitize returns 21 possible indices per axis, so neighbouring cells
shared Q-values; here each axis has exactly N_BINS cells.
"""
import gymnasium as gym
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np

N_BINS, EPISODES, ALPHA, GAMMA = 20, 10_000, 0.1, 0.99
rng = np.random.default_rng(0)
env = gym.make("MountainCar-v0")
low, high = env.observation_space.low, env.observation_space.high


def cell(obs):
    idx = ((obs - low) / (high - low) * N_BINS).astype(int)
    idx = np.clip(idx, 0, N_BINS - 1)
    return idx[0] * N_BINS + idx[1]


Q = np.zeros((N_BINS * N_BINS, env.action_space.n))
returns, success = [], []
for k in range(EPISODES):
    eps = max(0.01, 0.5 * (1 - 2 * k / EPISODES))       # decays to 0.01 at half-way
    obs, _ = env.reset(seed=int(rng.integers(1 << 30)))
    s, total, done = cell(obs), 0.0, False
    while not done:
        a = int(rng.integers(3)) if rng.random() < eps else int(Q[s].argmax())
        obs, r, term, trunc, _ = env.step(a)
        s2 = cell(obs)
        Q[s, a] += ALPHA * (r + GAMMA * Q[s2].max() * (not term) - Q[s, a])
        s, total, done = s2, total + r, term or trunc
    returns.append(total); success.append(term)
    if (k + 1) % 1000 == 0:
        print(f"episode {k + 1}: mean return (last 100) {np.mean(returns[-100:]):7.1f}, "
              f"reached the flag in {np.mean(success[-100:]):.0%}")

# greedy evaluation from 100 random starts
test = gym.make("MountainCar-v0")
steps = []
for i in range(100):
    obs, _ = test.reset(seed=10_000 + i)
    for t in range(200):
        obs, _, term, trunc, _ = test.step(int(Q[cell(obs)].argmax()))
        if term or trunc:
            break
    steps.append(t + 1 if term else np.nan)
print(f"greedy policy: reaches the flag in {np.mean(~np.isnan(steps)):.0%} of 100 starts, "
      f"median {np.nanmedian(steps):.0f} steps")

w = 100
avg = np.convolve(returns, np.ones(w) / w, mode="valid")
plt.figure(figsize=(7, 3.5))
plt.plot(returns, alpha=0.25, label="episode return")
plt.plot(np.arange(w - 1, EPISODES), avg, label=f"{w}-episode mean")
plt.xlabel("Episode"); plt.ylabel("Return"); plt.title("MountainCar-v0, tabular Q-learning")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig("figures/mountain_car_learning_curve.png", dpi=150)

# greedy value function -max_a Q (cost-to-go) over the binned state space
plt.figure(figsize=(4.5, 3.8))
plt.imshow(-Q.max(1).reshape(N_BINS, N_BINS).T, origin="lower", aspect="auto",
           extent=[low[0], high[0], low[1], high[1]], cmap="viridis")
plt.colorbar(label="steps to go (approx.)"); plt.xlabel("position"); plt.ylabel("velocity")
plt.title("Learned cost-to-go"); plt.tight_layout(); plt.savefig("figures/mountain_car_value.png", dpi=150)

render = gym.make("MountainCar-v0", render_mode="rgb_array")
obs, _ = render.reset(seed=1)
frames = []
for t in range(200):
    if t % 2 == 0:
        frames.append(render.render()[::2, ::2])
    obs, _, term, trunc, _ = render.step(int(Q[cell(obs)].argmax()))
    if term or trunc:
        break
imageio.mimsave("figures/mountain_car.gif", frames, fps=25, loop=0)
