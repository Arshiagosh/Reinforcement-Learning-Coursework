"""Six ways to solve deterministic FrozenLake-v1 (4x4).

HW4 Q6 and HW5 Q6. Model-based: policy iteration, value iteration.
Model-free: first-visit on-policy Monte Carlo control (epsilon-soft),
off-policy Monte Carlo control with weighted importance sampling,
SARSA and Q-learning. Every learned greedy policy is checked against the
optimal one from value iteration and rolled out in the environment.
"""
from collections import defaultdict

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

GAMMA = 0.95
ARROWS = np.array(["<", "v", ">", "^"])
rng = np.random.default_rng(0)


def make_env():
    return gym.make("FrozenLake-v1", is_slippery=False)


ENV = make_env().unwrapped
nS, nA = ENV.observation_space.n, ENV.action_space.n
P = ENV.P   # P[s][a] = [(prob, s', r, terminal)]


def q_from_v(V):
    Q = np.zeros((nS, nA))
    for s in range(nS):
        for a in range(nA):
            for p, s2, r, term in P[s][a]:
                Q[s, a] += p * (r + GAMMA * V[s2] * (not term))
    return Q


def policy_iteration():
    pi = np.zeros(nS, dtype=int)
    for it in range(1, 100):
        # exact policy evaluation: solve (I - gamma P_pi) V = r_pi
        Ppi, rpi = np.zeros((nS, nS)), np.zeros(nS)
        for s in range(nS):
            for p, s2, r, term in P[s][pi[s]]:
                rpi[s] += p * r
                if not term:
                    Ppi[s, s2] += p
        V = np.linalg.solve(np.eye(nS) - GAMMA * Ppi, rpi)
        new = q_from_v(V).argmax(1)
        if np.array_equal(new, pi):
            return pi, V, it
        pi = new


def value_iteration(tol=1e-10):
    V = np.zeros(nS)
    for it in range(1, 10_000):
        Vn = q_from_v(V).max(1)
        if np.max(np.abs(Vn - V)) < tol:
            return q_from_v(Vn).argmax(1), Vn, it
        V = Vn


def eps_greedy(Q, s, eps):
    if rng.random() < eps:
        return int(rng.integers(nA))
    best = np.flatnonzero(Q[s] == Q[s].max())
    return int(rng.choice(best))   # random tie-breaking: all-zero Q must not freeze on "left"


def episode(env, policy_fn, max_steps=100):
    s, _ = env.reset(seed=int(rng.integers(1 << 30)))
    traj = []
    for _ in range(max_steps):
        a = policy_fn(s)
        s2, r, term, trunc, _ = env.step(a)
        traj.append((s, a, r))
        s = s2
        if term or trunc:
            break
    return traj


def mc_on_policy(n_episodes=20_000, eps0=0.5):
    env, Q = make_env(), np.zeros((nS, nA))
    N = np.zeros((nS, nA))
    for k in range(n_episodes):
        eps = max(0.05, eps0 * (1 - k / n_episodes))
        traj = episode(env, lambda s: eps_greedy(Q, s, eps))
        G, first = 0.0, {}
        for t, (s, a, _) in enumerate(traj):
            first.setdefault((s, a), t)
        for t in reversed(range(len(traj))):
            s, a, r = traj[t]
            G = GAMMA * G + r
            if first[(s, a)] == t:                 # first visit only
                N[s, a] += 1
                Q[s, a] += (G - Q[s, a]) / N[s, a]
    return Q.argmax(1), Q


def mc_off_policy(n_episodes=20_000, eps=0.5):
    """Weighted importance sampling, greedy target, eps-greedy behaviour
    (Sutton and Barto, Sec. 5.7)."""
    env, Q, C = make_env(), np.zeros((nS, nA)), np.zeros((nS, nA))
    for _ in range(n_episodes):
        traj = episode(env, lambda s: eps_greedy(Q, s, eps))
        G, W = 0.0, 1.0
        for s, a, r in reversed(traj):
            G = GAMMA * G + r
            C[s, a] += W
            Q[s, a] += W / C[s, a] * (G - Q[s, a])
            if a != Q[s].argmax():
                break                               # target policy would not take a
            W /= (1 - eps) + eps / nA                # pi(a|s) = 1, b(a|s) for the greedy a
    return Q.argmax(1), Q


def td_control(kind, n_episodes=10_000, alpha=0.5, eps0=0.3):
    """SARSA or Q-learning with epsilon decaying linearly to 0.01 (GLIE)."""
    env, Q = make_env(), np.zeros((nS, nA))
    for k in range(n_episodes):
        eps = max(0.01, eps0 * (1 - k / n_episodes))
        s, _ = env.reset()
        a = eps_greedy(Q, s, eps)
        for _ in range(100):
            s2, r, term, trunc, _ = env.step(a)
            a2 = eps_greedy(Q, s2, eps)
            nxt = 0.0 if term else (Q[s2, a2] if kind == "sarsa" else Q[s2].max())
            Q[s, a] += alpha * (r + GAMMA * nxt - Q[s, a])
            s, a = s2, a2
            if term or trunc:
                break
    return Q.argmax(1), Q


def success_rate(pi, n=200):
    env, wins = make_env(), 0
    for _ in range(n):
        wins += sum(r for _, _, r in episode(env, lambda s: int(pi[s]))) > 0
    return wins / n


def optimal_actions(V):
    Q = q_from_v(V)
    return [set(np.flatnonzero(np.isclose(Q[s], Q[s].max()))) for s in range(nS)]


if __name__ == "__main__":
    holes = {s for s in range(nS) if ENV.desc.flatten()[s] in b"HG"}
    pi_pi, V_pi, it_pi = policy_iteration()
    pi_vi, V_vi, it_vi = value_iteration()
    opt = optimal_actions(V_vi)
    results = {
        f"Policy iteration ({it_pi} it.)": pi_pi,
        f"Value iteration ({it_vi} it.)": pi_vi,
        "MC on-policy": mc_on_policy()[0],
        "MC off-policy (weighted IS)": mc_off_policy()[0],
        "SARSA": td_control("sarsa")[0],
        "Q-learning": td_control("q")[0],
    }
    print("Map:\n" + "\n".join(row.decode() for row in ENV.desc.astype("S1").view("S4").flatten()))
    for name, pi in results.items():
        live = [s for s in range(nS) if s not in holes]
        ok = sum(pi[s] in opt[s] for s in live)
        grid = np.where([s in holes for s in range(nS)], "·", ARROWS[pi]).reshape(4, 4)
        print(f"\n{name}: reaches the goal in {success_rate(pi):.0%} of rollouts, optimal action in {ok}/{len(live)} states")
        print("\n".join(" ".join(row) for row in grid))

    # value function and optimal policy figure
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(V_vi.reshape(4, 4), cmap="Blues")
    for s in range(nS):
        i, j = divmod(s, 4)
        c = ENV.desc[i, j].decode()
        label = c if c in "HG" else ARROWS[pi_vi[s]]
        ax.text(j, i, f"{label}\n{V_vi[s]:.2f}", ha="center", va="center",
                color="white" if V_vi[s] > 0.6 else "black")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"FrozenLake: $V^*$ and optimal policy ($\\gamma$={GAMMA})")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig("figures/frozenlake_value.png", dpi=150)
