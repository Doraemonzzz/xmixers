# adapted from https://github.com/sustcsonglin/flash-linear-attention/blob/main/fla/models/utils.py

from typing import Any, Dict, List, Optional, Tuple

import torch
import transformers


class XmixersCache(transformers.cache_utils.Cache):
    def __init__(self, seen_tokens: int = 0):

        self.states: List[Dict[str, Any]] = []

        self._seen_tokens = seen_tokens  # Used in `generate` to keep tally of how many tokens the cache has seen

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
        recurrent_state: torch.Tensor = None,
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

        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += offset

        if attn_state is not None:
            input_size = attn_state[0].shape[-3]
            window_size = (
                cache_kwargs.get("window_size", None)
                if cache_kwargs is not None
                else None
            )
            if not isinstance(attn_state, Tuple) or len(attn_state) != 2:
                raise ValueError(
                    "`attn_state` must be a tuple of two tensors for key/value states"
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
        else:
            state = self.states[layer_idx]
            if recurrent_state is not None:
                state["recurrent_state"] = recurrent_state
            if attn_state is not None:
                k_state, v_state = state["attn_state"]
                if window_size is not None and k_state.shape[-3] == window_size:
                    # DO NOT allocate new memory if the cache is full
                    # roll the key/value states to the left by `input_size`
                    k_state = k_state.roll(-input_size, -3)
                    v_state = v_state.roll(-input_size, -3)
                    # replace the last `input_size` tokens with the new key/value states
                    k_state[..., -input_size:, :] = attn_state[0]
                    v_state[..., -input_size:, :] = attn_state[1]
                    attn_state = (k_state, v_state)
                else:
                    attn_state = (
                        torch.cat([k_state, attn_state[0]], -3),
                        torch.cat([v_state, attn_state[1]], -3),
                    )
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
                            torch.cat([k_state, mpa_state[0]], -2),
                            torch.cat([v_state, mpa_state[1]], -2),
                            k_head_state,
                            v_head_state,
                        )
                    else:
                        mpa_state = (
                            torch.cat([k_state, mpa_state[0]], -2),
                            torch.cat([v_state, mpa_state[1]], -2),
                            torch.cat([k_head_state, mpa_state[2]], -2),
                            torch.cat([v_head_state, mpa_state[3]], -2),
                        )
                state["mpa_state"] = mpa_state

        return state

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        if len(self.states) <= layer_idx:
            return 0
        return self._seen_tokens

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
