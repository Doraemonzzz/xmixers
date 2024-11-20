import math

import numpy as np


# forgetting transformer: https://openreview.net/pdf?id=q2Lnyegkr8
def get_log_slopes_general(d, n_min, n_max):
    log_n_min = math.log(n_min)
    log_n_max = math.log(n_max)
    n_list = [
        math.exp((log_n_min + (log_n_max - log_n_min) * i / (d - 1))) for i in range(d)
    ]
    # exp(log_slope * n) = 1 / e => log_slope * n = -1 => log_slope = -1 / n
    log_slope_list = [-1 / n for n in n_list]
    return np.array(log_slope_list)


# alibi: https://arxiv.org/abs/2108.12409
def get_log_slopes_power_of_2(d):
    start = 2 ** (-(2 ** -(math.log2(d) - 3)))
    ratio = start
    return [start * ratio**i for i in range(d)]


# alibi: https://arxiv.org/abs/2108.12409
def get_log_slopes(d):
    if math.log2(d).is_integer():
        return -np.array(
            get_log_slopes_power_of_2(d)
        )  # In the paper, we only train models that have 2^a heads for some a. This function has
    else:  # some good properties that only occur when the input is a power of 2. To maintain that even
        closest_power_of_2 = 2 ** math.floor(
            math.log2(n)
        )  # when the number of heads is not a power of 2, we use this workaround.
        return -np.array(
            (
                get_slopes_power_of_2(closest_power_of_2)
                + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
            )
        )


if __name__ == "__main__":
    d = 8
    log_slope_alibi = get_log_slopes(d)
    log_slope_general = get_log_slopes_general(d, 2, 256)

    print(log_slope_alibi)
    print(log_slope_general)

    print(np.linalg.norm(log_slope_alibi - log_slope_general))
