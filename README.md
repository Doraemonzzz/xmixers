# Xmixers: A collection of SOTA efficient token/channel mixers

<p align="center">
💬 <a href="https://discord.gg/ZpqcpSDE8g" target="_blank">Discord</a> •
</p>

## Introduction
This repository aims to implement SOTA efficient token/channel mixers. Any technologies related to non-Vanilla Transformer are welcome. If you are interested in this repository, please join our [Discord](https://discord.gg/ZpqcpSDE8g).

## Install

Install Torch(>=2.6.0), [fla](https://github.com/fla-org/flash-linear-attention), [xopes](https://github.com/Doraemonzzz/xopes.git) first, then install xmixers:

```
git clone https://github.com/Doraemonzzz/xmixers.git
cd xmixers
pip install -e .
```

## Models

| Paper                                                     | Code                                                         | Config                                                       |
| --------------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Elucidating the Design Space of Decay in Linear Attention | [Link](https://github.com/Doraemonzzz/xmixers/blob/main/xmixers/modules/token_mixers/linear_attention/decay_linear_attention.py), core code: line 247 to 262. | [Core method link](https://github.com/Doraemonzzz/flame/tree/main/configs/xmixers/decay_linear_transformer/decay_parameterize/hgrn3), [Ablation link](https://github.com/Doraemonzzz/flame/tree/main/configs/xmixers/decay_linear_transformer/decay_parameterize) |
|                                                           |                                                              |                                                              |
|                                                           |                                                              |                                                              |



## Training

To reproduce the results of the paper, we conducted training using the [flame](https://github.com/Doraemonzzz/flame) framework. First, we configured the environment in accordance with flame's requirements, then used flame's training script, and only needed to replace "config" with the corresponding name.
