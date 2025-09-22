import torch
from Actors.Network import ConActor
from torch import optim
import time
torch.manual_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# 参数
batch_size = 2048
in_feats = 64
action_bds = [(-1, 1), (0, 5), (-2, 2)]  # 三个连续动作

# 初始化模型
model = ConActor(in_feats=in_feats, action_bds=action_bds).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 随机输入
x = torch.randn(batch_size, in_feats, device=device)

# Forward 测试
torch.cuda.synchronize() if device == "cuda" else None
t0 = time.time()
actions, log_prob, entropy = model(x)
torch.cuda.synchronize() if device == "cuda" else None
t1 = time.time()
print("Forward time: {:.6f}s".format(t1 - t0))
print("Actions shape:", actions.shape)
print("Log_prob shape:", log_prob.shape)
print("Entropy shape:", entropy.shape)

# Backward 测试
loss = -log_prob.mean() - 0.01 * entropy.mean()  # 假设 PPO 损失的一部分
torch.cuda.synchronize() if device == "cuda" else None
t2 = time.time()
loss.backward()
optimizer.step()
torch.cuda.synchronize() if device == "cuda" else None
t3 = time.time()
print("Backward time: {:.6f}s".format(t3 - t2))
optimizer.zero_grad()
import ipdb
ipdb.set_trace()
actions, log_prob, entropy = model(x)
print(f"results after one step actions (first in batch):{actions[0]}")