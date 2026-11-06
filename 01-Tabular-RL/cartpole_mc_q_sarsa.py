"""CartPole-v1 with three tabular methods: every-visit Monte Carlo,
Q-learning and SARSA (comprehensive exercise, Q1).

The submitted notebook used a 60 x 60 x 120 x 120 grid, a Q-table of
about 10^8 entries that most states never visit. Cart position and
velocity matter little for keeping the pole up, so here they get 3 bins
each and the pole angle and angular velocity get 12: 1,296 states.
All three methods use the same grid, discount, exploration schedule and
number of episodes, so their learning curves can be compared directly.
"""
import gymnasium as gym
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np

BINS = np.array([3, 3, 12, 12])
LOW = np.array([-2.4, -3.0, -0.21, -3.5]); HIGH = -LOW
EPISODES, GAMMA, ALPHA = 6_000, 0.99, 0.1


def cell(obs):
    idx = ((obs - LOW) / (HIGH - LOW) * BINS).astype(int)
    return tuple(np.clip(idx, 0, BINS - 1))


def eps_at(k):
    return max(0.01, 1.0 * (1 - k / (0.6 * EPISODES)))


def train(method, seed):
    rng = np.random.default_rng(seed)
    env = gym.make("CartPole-v1")
    Q = np.zeros(tuple(BINS) + (2,)); N = np.zeros_like(Q)
    lengths = []
    act = lambda s, eps: int(rng.integers(2)) if rng.random() < eps else int(Q[s].argmax())
    for k in range(EPISODES):
        eps = eps_at(k)
        obs, _ = env.reset(seed=int(rng.integers(1 << 30)))
        s = cell(obs); a = act(s, eps); traj = []; done = False; steps = 0
        while not done:
            obs, r, term, trunc, _ = env.step(a)
            s2 = cell(obs); a2 = act(s2, eps); done = term or trunc; steps += 1
            if method == "Q-learning":
                Q[s + (a,)] += ALPHA * (r + GAMMA * Q[s2].max() * (not term) - Q[s + (a,)])
            elif method == "SARSA":
                Q[s + (a,)] += ALPHA * (r + GAMMA * Q[s2 + (a2,)] * (not term) - Q[s + (a,)])
            else:
                traj.append((s, a, r))
            s, a = s2, a2
        if method == "Monte Carlo":                     # every-visit, sample averages
            G = 0.0
            for s_, a_, r_ in reversed(traj):
                G = GAMMA * G + r_
                N[s_ + (a_,)] += 1
                Q[s_ + (a_,)] += (G - Q[s_ + (a_,)]) / N[s_ + (a_,)]
        lengths.append(steps)
    return Q, np.array(lengths)


def evaluate(Q, n=50):
    env, out = gym.make("CartPole-v1"), []
    for i in range(n):
        obs, _ = env.reset(seed=50_000 + i); t = 0; done = False
        while not done:
            obs, _, term, trunc, _ = env.step(int(Q[cell(obs)].argmax())); t += 1; done = term or trunc
        out.append(t)
    return np.array(out)


if __name__ == "__main__":
    methods, seeds = ["Monte Carlo", "Q-learning", "SARSA"], [0, 1, 2]
    curves, best = {}, {}
    for m in methods:
        runs = []
        for sd in seeds:
            Q, L = train(m, sd); runs.append(L)
            ev = evaluate(Q)
            print(f"{m:12s} seed {sd}: greedy policy balances {ev.mean():6.1f} steps on average "
                  f"(500 = episode limit), {np.mean(ev == 500):.0%} of runs reach 500")
            if m not in best or ev.mean() > best[m][1]:
                best[m] = (Q, ev.mean())
        curves[m] = np.array(runs)

    w = 100
    plt.figure(figsize=(7.5, 4))
    for m in methods:
        sm = np.array([np.convolve(r, np.ones(w) / w, mode="valid") for r in curves[m]])
        x = np.arange(w - 1, EPISODES)
        plt.plot(x, sm.mean(0), label=m)
        plt.fill_between(x, sm.min(0), sm.max(0), alpha=0.2)
    plt.xlabel("Episode"); plt.ylabel(f"Episode length ({w}-episode mean)")
    plt.title("CartPole-v1: tabular MC vs. Q-learning vs. SARSA (3 seeds, band = min/max)")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig("figures/cartpole_learning_curves.png", dpi=150)

    env = gym.make("CartPole-v1", render_mode="rgb_array")
    for m in methods:
        Q = best[m][0]; obs, _ = env.reset(seed=7); frames = []
        for t in range(500):
            if t % 3 == 0:
                frames.append(env.render()[::2, ::2])
            obs, _, term, trunc, _ = env.step(int(Q[cell(obs)].argmax()))
            if term or trunc:
                break
        imageio.mimsave(f"figures/cartpole_{m.lower().replace(' ', '_').replace('-', '')}.gif", frames, fps=20, loop=0)
