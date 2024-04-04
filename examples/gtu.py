import torch

from xmixers.modules import Gtu

device = "cuda:0"
b, n, m, d = 2, 12, 23, 128

x = torch.randn(b, n, m, d).to(device)

gtu_causal = Gtu(
    embed_dim=d,
    causal=True,
    dims=[1, 2],
    in_dim=16,
).to(device)

gtu_none_causal = Gtu(
    embed_dim=d,
    causal=False,
    dims=[1, 2],
    in_dim=16,
).to(device)

print(gtu_causal)
print(gtu_none_causal)

y1 = gtu_causal(x)
y2 = gtu_none_causal(x)

print(x.shape)
print(y1.shape)
print(y2.shape)
