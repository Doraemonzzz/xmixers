# Xmixers: A collection of SOTA efficient token/channel mixers

<p align="center">
💬 <a href="https://discord.gg/ZpqcpSDE8g" target="_blank">Discord</a> •
</p>

# Introduction
This repository aims to implement SOTA efficient token/channel mixers. Any technologies related to non-Vanilla Transformer are welcome. If you are interested in this repository, please join our [Discord](https://discord.gg/ZpqcpSDE8g).

# Roadmap
- Token Mixers
    - Linear Attention
    - Linear RNN
    - Long Convolution
- Channel Mixers

# Finished Model
- [x] Mpa
- [x] T6
- [x] Mla
- [x] Hgrn2
  - left varlen
- [x] Hgrn2-scalar-decay
  - left varlen
- [x] Linear Transformer
- [x] Llama
- [x] Tnl
- [x] Deltanet
- [x] Vector Decay Deltanet
- [x] Scalar Decay Deltanet
- [ ] TTT
- [ ] GSA
- [ ] Titan
- [ ] NSA
- [ ] Alibi
- [ ] GPT
- [ ] Stick-breaking
- [ ] Forgetting Transformer

# Pretrained weights

- GPT
  - Doreamonzzz/xmixers_gpt_120m_50b
- LLaMA
  - Doreamonzzz/xmixers_llama_120m_50b
-

# ToDo
- [ ] Add init state for linear attention.
- [ ] Update attn mask treat for attention, mpa, tpa, mla;
- [x] Update init weights for every model.
- [ ] Rm bias for layernorm since it will raise nan error.
  - [ ] linear attention
    - [x] hgru2
    - [x] hgru3
  - [ ] vanilla attention
    - [x] attention
- [ ] Add causal.
- [ ] Update _initialize_weights, _init_weights.
  - [x] linear attention
  - [x] vanilla attention
  - [ ] long conv
- [ ] Update cache for token mixer.
  - linear attention
    - [x] hgru3
    - [ ] linear attention
    - [ ] tnl attention
  - long conv
    - [ ] gtu
  - vanilla attention
    - [x] attention
    - [ ] flex attention
    - [x] mpa
    - [ ] n_attention
- [ ] Add special init.
- [ ] Add causal.
- [x] Clear next_decoder_cache.
- [ ] Add varlen for softmax attn.

## Model
- [x] LLaMA.
- [x] GPT.

## Basic
- [ ] Add data type for class and function.

## Ops
- [x] long_conv_1d_op.

## Token Mixers
- [x] Gtu.

# Note
```
[Feature Add]
[Bug Fix]
[Benchmark Add]
[Document Add]
[README Add]
```
