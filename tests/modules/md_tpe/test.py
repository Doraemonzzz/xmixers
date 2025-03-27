import torch

from xmixers.modules.pes.md import MdTpe


def test_md_tpe(embed_dim, num_heads, dims, shape, device):
    md_tpe = MdTpe(embed_dim=embed_dim, num_heads=num_heads, dims=dims).to(device)
    x = torch.randn(shape, device=device).requires_grad_()
    print("input shape: ", x.shape)
    o = md_tpe(x)
    print("output shape: ", o.shape)
    o.sum().backward()
    print("input grad: ", x.grad.shape)


if __name__ == "__main__":
    embed_dim = 128
    num_heads = 16
    dim = [-2, -3, -4]
    shape = (2, 16, 16, 16, embed_dim)
    device = "cuda"
    test_md_tpe(
        embed_dim=embed_dim, num_heads=num_heads, dims=dim, shape=shape, device=device
    )
