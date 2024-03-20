import pytest
import torch
from torch.testing import assert_close

from xmixers.ops import long_conv_1d_op, long_conv_1d_op_naive


def get_params():
    array = []
    for b in [1, 2]:
        for n in [66, 128, 256]:
            for m in [66, 128, 256]:
                for d in [63, 128]:
                    array.append((b, n, m, d))

    return array


@pytest.mark.parametrize("b, n, m, d", get_params())
@pytest.mark.parametrize("dim", [1, 2])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [True, False])
def test_srmsnorm(b, n, m, d, dim, dtype, causal):
    torch.manual_seed(2024)
    atol = 5e-2
    rtol = 1e-2
    device = torch.device("cuda")
    x = torch.randn((b, n, m, d), dtype=dtype, device=device).requires_grad_()
    dy = torch.randn((b, n, m, d), dtype=dtype, device=device)
    if dim == 1:
        l = n
    else:
        l = m

    ##### !!! w should be fp32, or there will have numeric error
    # w0
    zero = torch.randn((1, d), dtype=dtype, device=device)
    # w1, ... , w(n-1)
    pos = torch.randn((l - 1, d), dtype=dtype, device=device)
    if causal:
        w = torch.cat([zero, pos], dim=0).to(torch.float32).requires_grad_()
    else:
        # w(-(n-1)), ... , w(-1)
        neg = torch.randn((l - 1, d), dtype=dtype, device=device)
        w = torch.cat([zero, pos, zero, neg], dim=0).to(torch.float32).requires_grad_()

    # forward
    y_ref = long_conv_1d_op_naive(x, w, dim)
    y = long_conv_1d_op(x, w, dim)

    # backward
    y_ref.backward(dy, retain_graph=True)
    dx_ref, x.grad = x.grad.clone(), None
    dw_ref, w.grad = w.grad.clone(), None

    y.backward(dy, retain_graph=True)
    dx, x.grad = x.grad.clone(), None
    dw, w.grad = w.grad.clone(), None

    # test
    assert_close(y, y_ref, atol=atol, rtol=rtol)
    assert_close(dx, dx_ref, atol=atol, rtol=rtol)
    assert_close(dw, dw_ref, atol=atol, rtol=rtol)
