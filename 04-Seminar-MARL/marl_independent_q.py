"""Independent Q-learning for three agents on a 10 x 10 grid (MARL seminar demo).

Each agent has its own Q-table over its own cell and learns to reach its
own goal while avoiding obstacles; another agent in the way blocks the
move. This is the simplest MARL baseline: every agent treats the others
as part of the environment, which makes that environment non-stationary.

Changes from the seminar notebook: an agent that reaches its goal stops
there (its goal reward is terminal instead of being collected at every
step), the run is reproducible, and the learned policies are tested from
200 random start configurations instead of a single one.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

G, N_AGENTS, N_OBST = 10, 3, 8
GOALS = [(9, 9), (0, 9), (9, 0)]
ACTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0), (0, 0)]       # left, right, up, down, stay
ALPHA, GAMMA, EPISODES, MAX_STEPS = 0.2, 0.95, 6000, 60
R_GOAL, R_STEP, R_BLOCKED = 100.0, -1.0, -5.0
rng = np.random.default_rng(42)

obstacles = set()
while len(obstacles) < N_OBST:
    c = tuple(int(v) for v in rng.integers(0, G, 2))
    if c not in GOALS:
        obstacles.add(c)
free = [(i, j) for i in range(G) for j in range(G) if (i, j) not in obstacles and (i, j) not in GOALS]


def random_starts():
    idx = rng.choice(len(free), N_AGENTS, replace=False)
    return [free[i] for i in idx]


def step(pos, a, others):
    new = (min(G - 1, max(0, pos[0] + ACTIONS[a][0])), min(G - 1, max(0, pos[1] + ACTIONS[a][1])))
    if new in obstacles or new in others:
        return pos, True
    return new, False


def run_episode(Q, starts, eps, learn):
    pos, done, total = list(starts), [False] * N_AGENTS, 0.0
    path = [list(pos)]
    for t in range(MAX_STEPS):
        for i in range(N_AGENTS):
            if done[i]:
                continue
            s = pos[i]
            a = int(rng.integers(5)) if rng.random() < eps else int(Q[i][s].argmax())
            new, blocked = step(s, a, [p for k, p in enumerate(pos) if k != i])
            done[i] = new == GOALS[i]
            r = R_GOAL if done[i] else (R_BLOCKED if blocked else R_STEP)
            if learn:
                target = r if done[i] else r + GAMMA * Q[i][new].max()
                Q[i][s + (a,)] += ALPHA * (target - Q[i][s + (a,)])
            pos[i] = new; total += r
        path.append(list(pos))
        if all(done):
            break
    return total, all(done), t + 1, path


Q = [np.zeros((G, G, len(ACTIONS))) for _ in range(N_AGENTS)]
returns = []
for k in range(EPISODES):
    eps = max(0.02, 0.5 * (1 - k / (0.7 * EPISODES)))
    returns.append(run_episode(Q, random_starts(), eps, learn=True)[0])

tests = [run_episode(Q, random_starts(), 0.0, learn=False) for _ in range(200)]
ok = np.array([t[1] for t in tests]); steps = np.array([t[2] for t in tests])
print(f"greedy policies: all three agents reach their goals in {ok.mean():.0%} of 200 random starts, "
      f"median {np.median(steps[ok]):.0f} steps when they do")

w = 100
plt.figure(figsize=(7, 3.5))
plt.plot(np.convolve(returns, np.ones(w) / w, mode="valid"))
plt.xlabel("Episode"); plt.ylabel(f"Team return ({w}-episode mean)")
plt.title("Independent Q-learning, 3 agents"); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig("figures/marl_learning_curve.png", dpi=150)

path = run_episode(Q, [(0, 0), (9, 4), (4, 0)], 0.0, learn=False)[3]
colors = ["tab:blue", "tab:red", "tab:green"]
fig, ax = plt.subplots(figsize=(5, 5))


def draw(f):
    ax.clear(); ax.set_xlim(-0.5, G - 0.5); ax.set_ylim(G - 0.5, -0.5)
    ax.set_xticks(range(G)); ax.set_yticks(range(G)); ax.grid(True, alpha=0.3)
    for o in obstacles:
        ax.add_patch(plt.Rectangle((o[1] - 0.5, o[0] - 0.5), 1, 1, color="0.2"))
    for i in range(N_AGENTS):
        ax.plot(GOALS[i][1], GOALS[i][0], "*", ms=18, color=colors[i])
        tr = np.array([p[i] for p in path[: f + 1]])
        ax.plot(tr[:, 1], tr[:, 0], "--", color=colors[i])
        ax.plot(tr[-1, 1], tr[-1, 0], "o", ms=13, color=colors[i])
    ax.set_title(f"step {f}")


FuncAnimation(fig, draw, frames=len(path), interval=400).save("figures/marl_agents.gif", writer=PillowWriter(fps=3))
