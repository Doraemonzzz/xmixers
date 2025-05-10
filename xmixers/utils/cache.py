# adapted from https://github.com/sustcsonglin/flash-linear-attention/blob/main/fla/models/utils.py

from typing import Any, Dict, List, Optional, Tuple

import torch
import transformers


class XmixersCache(transformers.cache_utils.Cache):
    def __init__(self, seen_tokens: int = 0):
        self.states: List[Dict[str, Any]] = []
        self._seen_tokens = []

    def __getitem__(self, layer_idx: int) -> Dict[str, Any]:
        if layer_idx < len(self):
            return self.states[layer_idx]
        else:
            raise KeyError(
                f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}"
            )

    def __iter__(self):
        for state in self.states:
            yield state

    def __len__(self):
        return len(self.states)

    def update(
        self,
        recurrent_state: List[torch.Tensor] = None,
        attn_state: Tuple[torch.Tensor, torch.Tensor] = None,
        conv_state: Tuple[torch.Tensor] = None,
        ffn_state: torch.Tensor = None,
        mpa_state: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] = None,
        layer_idx: int = 0,
        offset: Optional[int] = 1,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Updates the cache with the new `recurrent_state`/`attn_state`/`conv_state` for the layer `layer_idx`.

        Args:
            recurrent_state (`torch.Tensor`, `optional`):
                The new recurrent state to cache.
            attn_state (`Tuple[torch.Tensor, torch.Tensor]`, `optional`):
                The new attention key/value states to cache.
            conv_state (`Tuple[torch.Tensor]`, `optional`):
                The new convolution state to cache.
            layer_idx (`int`, defaults to 0):
                The index of the layer to cache the states for.
            offset (`int`, `optional`, defaults to 1):
                The number of new tokens being processed.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass.

        Return:
            Dictionary of the updated state.
        """

        if attn_state is not None:
            input_size = attn_state[0].shape[-3]
            window_size = (
                cache_kwargs.get("window_size", None)
                if cache_kwargs is not None
                else None
            )
            if not isinstance(attn_state, Tuple) or len(attn_state) < 2:
                raise ValueError(
                    "`attn_state` must be a tuple of at least two tensors for key/value states"
                )
        if mpa_state is not None:
            input_size = mpa_state[0].shape[-3]
            window_size = (
                cache_kwargs.get("window_size", None)
                if cache_kwargs is not None
                else None
            )
            if not isinstance(mpa_state, Tuple) or len(mpa_state) != 4:
                raise ValueError(
                    "`mpa_state` must be a tuple of four tensors for k, v, k_head, v_head states"
                )
        # first time
        if len(self.states) <= layer_idx:
            if attn_state is not None:
                if window_size is not None and input_size > window_size:
                    attn_state = (
                        x[..., -window_size:, :].contiguous() for x in attn_state
                    )
            if mpa_state is not None:
                if window_size is not None and input_size > window_size:
                    mpa_state = (
                        x[..., -window_size:, :].contiguous() for x in mpa_state
                    )

            state = dict(
                recurrent_state=recurrent_state,
                attn_state=attn_state,
                conv_state=conv_state,
                ffn_state=ffn_state,
                mpa_state=mpa_state,
            )
            self.states.append(state)
            self._seen_tokens.append(offset)
        else:
            state = self.states[layer_idx]
            self._seen_tokens[layer_idx] += offset
            if recurrent_state is not None:
                state["recurrent_state"] = recurrent_state
            if attn_state is not None:
                k_state, v_state = state["attn_state"][:2]
                if window_size is not None and k_state.shape[-3] == window_size:
                    # DO NOT allocate new memory if the cache is full
                    # roll the key/value states to the left by `input_size`
                    new_attn_state = []
                    for i, state_ in enumerate(state["attn_state"]):
                        state_ = state_.roll(-input_size, 1)
                        state_[..., -input_size:, :] = attn_state[i]
                        new_attn_state.append(state)
                    attn_state = tuple(new_attn_state)
                else:
                    new_attn_state = []
                    for i, state_ in enumerate(state["attn_state"]):
                        new_state = torch.cat([state_, attn_state[i]], dim=1)
                        new_attn_state.append(new_state)
                    attn_state = tuple(new_attn_state)
                state["attn_state"] = attn_state
            if conv_state is not None:
                state["conv_state"] = conv_state
            if ffn_state is not None:
                state["ffn_state"] = ffn_state
            if mpa_state is not None:
                if window_size is not None and k_state.shape[-3] == window_size:
                    new_mpa_state = []
                    for i, x in enumerate(state["mpa_state"]):
                        # DO NOT allocate new memory if the cache is full
                        # roll the key/value states to the left by `input_size`
                        x_new = x.roll(-input_size, -3)
                        # replace the last `input_size` tokens with the new key/value states
                        x_new[..., -input_size:, :] = mpa_state[i]
                        new_mpa_state.append(x_new)
                    mpa_state = tuple(new_mpa_state)
                else:
                    k_state, v_state, k_head_state, v_head_state = state["mpa_state"]
                    if len(k_head_state.shape) == 1:
                        mpa_state = (
                            torch.cat([k_state, mpa_state[0]], 1),
                            torch.cat([v_state, mpa_state[1]], 1),
                            k_head_state,
                            v_head_state,
                        )
                    else:
                        mpa_state = (
                            torch.cat([k_state, mpa_state[0]], 1),
                            torch.cat([v_state, mpa_state[1]], 1),
                            torch.cat([k_head_state, mpa_state[2]], 1),
                            torch.cat([v_head_state, mpa_state[3]], 1),
                        )
                state["mpa_state"] = mpa_state

        return state

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        if len(self.states) <= layer_idx:
            return 0
        else:
            return self._seen_tokens[layer_idx]

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states. Cache does not have a maximum length."""
        return None

    def to_legacy_cache(self) -> Tuple:
        return tuple(self.states)

    @classmethod
    def from_legacy_cache(
        cls, past_key_values: Optional[Tuple] = None, seen_tokens: int = 0
    ):
        """Converts a cache in the legacy cache format into an equivalent `Cache`."""

        cache = cls(seen_tokens)
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                cache.states.append(past_key_values[layer_idx])
        return cache
