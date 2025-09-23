import torch
import torch.nn as nn
import torch.distributions as D


class Actor(nn.Module):
    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_feats, 16), nn.ReLU(),
            nn.Linear(16, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 16), nn.ReLU(),
            nn.Linear(16, out_feats * 2)  # 每个维度输出 μ 和 logσ
        )

    def forward(self, x):
        return self.layers(x)


class ConActor(Actor):
    def __init__(self, in_feats, action_bds):
        """
        action_bds: list of (low, high) tuples, 每个动作的边界
        """
        super().__init__(in_feats, len(action_bds))
        self.action_bds = action_bds
        self.register_buffer("low",  torch.tensor([lo for lo, _ in action_bds], dtype=torch.float32))
        self.register_buffer("high", torch.tensor([hi for _, hi in action_bds], dtype=torch.float32))
        scale = (self.high - self.low) / 2.0
        self.register_buffer("log_scale_sum", torch.log(scale).sum())

    def forward(self, x):
        raw_o = super().forward(x)   # [batch, out_feats*2]
        mu, log_sigma = torch.chunk(raw_o, 2, dim=-1)
        log_sigma = torch.clamp(log_sigma, -20.0, 2.0)### scale explode
        sigma = torch.exp(log_sigma)

        # 构建 Normal 分布
        dist = D.Normal(mu, sigma)

        # 采样动作（reparameterization trick）
        action_raw = dist.rsample()

        # log_prob 和 entropy (逐维求和更符合 policy gradient 习惯)
        log_prob = dist.log_prob(action_raw).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)

        # 映射到 action_bds
        scaled_actions = []
        for i, (low, high) in enumerate(self.action_bds):
            a = torch.tanh(action_raw[:, i])  # 限制到 [-1,1]
            a = low + (a + 1.0) * 0.5 * (high - low)  # 映射到 [low, high]
            scaled_actions.append(a.unsqueeze(-1))
        scaled_actions = torch.cat(scaled_actions, dim=-1)

        return scaled_actions, log_prob, entropy