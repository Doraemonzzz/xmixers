import torch.nn as nn

from xmixers.utils import XMIXERS_DEBUG, print_params


class BaseModule(nn.Module):
    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()

        if XMIXERS_DEBUG:
            # get local varables
            params = locals()
            # print params
            print_params(**params)
