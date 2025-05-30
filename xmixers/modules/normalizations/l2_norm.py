from xopes.ops.normalize import normalize_fn


def l2_norm(x, eps=1e-5):
    return normalize_fn(
        x=x,
        weight=None,
        bias=None,
        residual=None,
        c=1,
        eps=eps,
        use_mean=False,
        num_groups=1,
    )


def rms_norm_fn(x, eps=1e-5):
    return normalize_fn(
        x=x,
        weight=None,
        bias=None,
        residual=None,
        c=x.shape[-1] ** 0.5,
        eps=eps,
        use_mean=False,
        num_groups=1,
    )
