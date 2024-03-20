import torch


def long_conv_1d_op_naive(x, w, dim):
    """
    x:   b, n1, ... nk, d
    w:   ni, d
         w0, w1, ... , w(n-1) for causal
         w0, w1, ... , w(n-1), w0, w(-(n-1)), ... , w(-1) for non causal
    dim: i
    """
    # other dtype have numeric error
    assert w.dtype == torch.float32
    n = x.shape[dim]
    if w.shape[0] == n:  # causal situation
        w = torch.cat([w, torch.zeros_like(w).to(w)], dim=0)

    zero = w[:1]
    pos = w[1:n]
    neg = w[n + 1 :]

    c = torch.cat([zero, pos], dim=0)
    r = torch.cat([zero, neg.flip(0)], dim=0)
    vals = torch.cat([r, c[1:].flip(0)], dim=0)
    i, j = torch.ones(n, n).nonzero().T
    T = vals[j - i].reshape(n, n, -1)

    x = x.transpose(0, dim)

    y = (
        torch.einsum("n m d, m ... d -> n ... d", T.float(), x.float())
        .transpose(0, dim)
        .to(x.dtype)
    )

    return y


def long_conv_1d_op(x, w, dim):
    """
    x:   b, n1, ... nk, d
    w:   ni, d
         w0, w1, ... , w(n-1) for causal
         w0, w1, ... , w(n-1), w0, w(-(n-1)), ... , w(-1) for non causal
    dim: i
    """
    # other dtype have numeric error
    assert w.dtype == torch.float32
    m = len(x.shape)
    if dim < 0:
        dim += m
    n = x.shape[dim]
    x_fft = torch.fft.rfft(x.float(), 2 * n, dim=dim)
    # convert w to a shape that can be broadcast to x
    for _ in range(dim):
        w = w.unsqueeze(0)
    for _ in range(m - 2 - dim):
        w = w.unsqueeze(-2)

    w_fft = torch.fft.rfft(w.float(), 2 * n, dim=dim)
    y_fft = x_fft * w_fft

    index = [slice(None)] * m
    index[dim] = slice(0, n)
    y = torch.fft.irfft(y_fft, 2 * n, dim=dim)[tuple(index)].to(x.dtype)

    return y
