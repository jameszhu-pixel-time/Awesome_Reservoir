import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
import math
import random
from typing import List, Tuple
from dataclasses import dataclass
from Actors.Network import ConActor
from Critics.Network import QValueNetContinuous
from tqdm import tqdm
import matplotlib.pyplot as plt

class Goal1DEnv:
    """
    State: [x]
    Action: a in [low, high]
    Reward: - (x - x_goal)^2 - action_cost * a^2
    Done: |x - x_goal| < tol 或 时间到
    """
    def __init__(self,
                 x_goal=3.0,
                 action_low=-1.0,
                 action_high=1.0,
                 step_scale=0.2,
                 action_cost=0.001,
                 max_steps=200,
                 tol=0.05,
                 init_range=2.0,
                 seed=0):
        self.x_goal = x_goal
        self.low = action_low
        self.high = action_high
        self.step_scale = step_scale
        self.action_cost = action_cost
        self.max_steps = max_steps
        self.tol = tol
        self.init_range = init_range
        self.rng = np.random.RandomState(seed)
        self.reset()

    @property
    def state_dim(self): return 1

    @property
    def action_dim(self): return 1

    @property
    def action_bds(self): return [(self.low, self.high)]

    def reset(self):
        self.t = 0
        self.x = float(self.rng.uniform(-self.init_range, self.init_range))
        return np.array([self.x], dtype=np.float32)

    def step(self, a: float):
        a = float(np.clip(a, self.low, self.high))
        self.x = float(self.x + self.step_scale * a)
        self.t += 1
        dist2 = (self.x - self.x_goal) ** 2
        reward = - dist2 - self.action_cost * (a ** 2)
        done = (abs(self.x - self.x_goal) < self.tol) or (self.t >= self.max_steps)
        return np.array([self.x], dtype=np.float32), float(reward), bool(done), {}


# =========================
# 4) Replay Buffer
# =========================
class ReplayBuffer:
    def __init__(self, state_dim, action_dim, capacity=int(1e6)):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def push(self, s, a, r, s2, d):
        if np.any(np.isnan(s)) or np.any(np.isnan(a)) or np.isnan(r) or np.any(np.isnan(s2)):
            print("NaN detected in transition!", s, a, r, s2)
            raise "Nan"
        self.states[self.ptr] = s
        self.actions[self.ptr] = a
        self.rewards[self.ptr] = r
        self.next_states[self.ptr] = s2
        self.dones[self.ptr] = d
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.tensor(self.states[idxs]),
            torch.tensor(self.actions[idxs]),
            torch.tensor(self.rewards[idxs]),
            torch.tensor(self.next_states[idxs]),
            torch.tensor(self.dones[idxs]),
        )

class EVHERBuffer(ReplayBuffer):
    def __init__(self,state_dim, action_dim, capacity=int(1e6)):
        super.__init__(self,state_dim, action_dim, capacity=int(1e6))
    
    
    
# =========================
# 5) SAC Agent
# =========================
@dataclass
class SACConfig:
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 128
    lr_actor: float = 3e-4
    lr_q: float = 3e-4
    lr_alpha: float = 3e-4
    hidden_dim_q: int = 256
    target_entropy_coef: float = -1.0     # target entropy = coef * action_dim
    start_steps: int = 1000               # 随机探索步数
    updates_per_step: int = 1

class SACAgent:
    def __init__(self, state_dim, action_bds, cfg=SACConfig(), device=None):
        self.device = device if device is not None else torch.device("cpu")
        self.cfg = cfg
        action_dim = len(action_bds)

        # Actor
        self.actor = ConActor(state_dim, action_bds).to(self.device)

        # Twin Q + target
        self.q1 = QValueNetContinuous(state_dim, cfg.hidden_dim_q, action_dim).to(self.device)
        self.q2 = QValueNetContinuous(state_dim, cfg.hidden_dim_q, action_dim).to(self.device)
        self.q1_target = QValueNetContinuous(state_dim, cfg.hidden_dim_q, action_dim).to(self.device)
        self.q2_target = QValueNetContinuous(state_dim, cfg.hidden_dim_q, action_dim).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # Optims
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr_actor)
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=cfg.lr_q)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=cfg.lr_q)

        # 温度 α（自动调参）
        self.log_alpha = torch.tensor(0.0, requires_grad=True, device=self.device)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=cfg.lr_alpha)

        # 目标熵
        self.target_entropy = cfg.target_entropy_coef * action_dim

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, state: np.ndarray, deterministic=False):
        self.actor.eval()
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            raw_o = self.actor.layers(s)
            mu, log_sigma = torch.chunk(raw_o, 2, dim=-1)
            log_sigma = torch.clamp(log_sigma, -20.0, 2.0)
            sigma = torch.exp(log_sigma)
            dist = Normal(mu, sigma)
            u = mu if deterministic else dist.sample()
            a_tanh = torch.tanh(u)
            scaled = self.actor.low + (a_tanh + 1.0) * 0.5 * (self.actor.high - self.actor.low)
        self.actor.train()
        return scaled.cpu().numpy().flatten()

    def _soft_update(self, online: nn.Module, target: nn.Module, tau: float):
        with torch.no_grad():
            for p, p_targ in zip(online.parameters(), target.parameters()):
                p_targ.data.mul_(1.0 - tau).add_(tau * p.data)

    def update(self, replay: ReplayBuffer):
        cfg = self.cfg
        if replay.size < cfg.batch_size:
            return {}

        s, a, r, s2, d = replay.sample(cfg.batch_size)
        s = s.to(self.device)
        a = a.to(self.device)
        r = r.to(self.device)
        s2 = s2.to(self.device)
        d = d.to(self.device)

        # -------- Q targets --------
        with torch.no_grad():
            a2, logp_a2, _ = self.actor(s2)
            q1_t = self.q1_target(s2, a2)
            q2_t = self.q2_target(s2, a2)
            q_min_t = torch.min(q1_t, q2_t)
            target = r + cfg.gamma * (1.0 - d) * (q_min_t - self.alpha.detach() * logp_a2)

        # Q1
        q1_pred = self.q1(s, a)
        q1_loss = F.mse_loss(q1_pred, target)
        self.q1_opt.zero_grad(set_to_none=True)
        q1_loss.backward()
        self.q1_opt.step()

        # Q2
        q2_pred = self.q2(s, a)
        q2_loss = F.mse_loss(q2_pred, target)
        self.q2_opt.zero_grad(set_to_none=True)
        q2_loss.backward()
        self.q2_opt.step()

        # -------- Policy (maximize Q - alpha*logπ) --------
        a_new, logp_new, _ = self.actor(s)
        q1_pi = self.q1(s, a_new)
        q2_pi = self.q2(s, a_new)
        q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha.detach() * logp_new - q_pi).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        # -------- Temperature α (auto-tune) --------
        alpha_loss = -(self.log_alpha * (logp_new.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()

        # -------- Soft target update --------
        self._soft_update(self.q1, self.q1_target, cfg.tau)
        self._soft_update(self.q2, self.q2_target, cfg.tau)

        return {
            "q1_loss": float(q1_loss.item()),
            "q2_loss": float(q2_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha": float(self.alpha.item()),
        }


# =========================
# 6) 训练脚本（最小可跑）
# =========================
import matplotlib.pyplot as plt

def train_sac(episodes=80, steps_per_episode=200, seed=1):
    np.random.seed(seed); random.seed(seed); torch.manual_seed(seed)
    env = Goal1DEnv(x_goal=3.0, action_low=-1.0, action_high=1.0,
                    step_scale=0.2, action_cost=0.001, max_steps=steps_per_episode, seed=seed)
    cfg = SACConfig(
        gamma=0.99, tau=0.01, batch_size=128, lr_actor=3e-4, lr_q=3e-4, lr_alpha=3e-4,
        hidden_dim_q=256, target_entropy_coef=-1.0, start_steps=1000, updates_per_step=1
    )
    agent = SACAgent(env.state_dim, env.action_bds, cfg=cfg, device=torch.device("cpu"))
    replay = ReplayBuffer(env.state_dim, env.action_dim, capacity=100000)

    total_steps = 0
    ep_returns = []

    # ===== 打开交互模式 =====
    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 4))

    for ep in range(episodes):
        s = env.reset()
        ep_ret = 0.0
        for t in range(steps_per_episode):
            if total_steps < cfg.start_steps:
                a = np.array([np.random.uniform(env.low, env.high)], dtype=np.float32)
            else:
                a = agent.select_action(s, deterministic=False).astype(np.float32)

            s2, r, done, _ = env.step(a[0])
            replay.push(s, a, r, s2, float(done))
            s = s2
            ep_ret += r
            total_steps += 1

            for _ in range(cfg.updates_per_step):
                agent.update(replay)

            if done:
                break

        ep_returns.append(ep_ret)

        # ===== 实时更新图像 =====
        ax.clear()
        ax.plot(ep_returns, label="Return")
        ax.set_title("SAC Training Returns")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Return")
        ax.legend()
        plt.pause(0.01)  # 暂停一会儿刷新

        if (ep + 1) % 10 == 0:
            print(f"[Ep {ep+1:03d}] Return={ep_ret:.2f}, Buffer={replay.size}, Alpha={agent.alpha.item():.3f}")

    plt.ioff()
    plt.show()

    return agent, ep_returns


if __name__ == "__main__":
    # 训练 50~200 个 episode 即可看到学习曲线在提升（根据算力适当增减）
    agent, returns = train_sac(episodes=80, steps_per_episode=100, seed=1)
    # 如需评估确定性策略（μ）：
    # state = np.array([0.0], dtype=np.float32)
    # print("Det action from state 0:", agent.select_action(state, deterministic=True))