# Copyright 2023-2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""MoE dispatch utilities."""

import torch


def moe_dispatch(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    lora_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Dispatch tokens to experts for MoE computation.

    Args:
        topk_ids: [num_tokens, top_k] - Expert IDs selected by router
        topk_weights: [num_tokens, top_k] - Router weights
        lora_indices: [num_tokens] - LoRA adapter ID for each token

    Returns:
        sorted_token_ids: Token indices sorted by expert_id
        sorted_expert_ids: Corresponding expert IDs
        sorted_topk_weights: Corresponding router weights
        sorted_lora_ids: LoRA adapter IDs for each dispatched token
        sorted_expert_slots: Original expert slot (0 to top_k-1) for each dispatched pair
    """
    num_tokens, top_k = topk_ids.shape
    device = topk_ids.device

    # Ensure lora_indices is on the same device as topk_ids
    if lora_indices.device != device:
        lora_indices = lora_indices.to(device)

    # Flatten topk dimensions: [num_tokens * top_k]
    flat_topk_ids = topk_ids.flatten()
    flat_topk_weights = topk_weights.flatten()
    flat_token_ids = torch.arange(num_tokens, device=device).repeat_interleave(top_k)
    flat_lora_ids = lora_indices.repeat_interleave(top_k)

    # Expert slot indices (0, 1, ..., top_k-1) repeated for each token
    flat_expert_slots = torch.arange(top_k, device=device).repeat(num_tokens)

    # Sort by expert_id only (each expert uses same LoRA adapter logic)
    sorted_indices = torch.argsort(flat_topk_ids)

    sorted_token_ids = flat_token_ids[sorted_indices]
    sorted_expert_ids = flat_topk_ids[sorted_indices]
    sorted_topk_weights = flat_topk_weights[sorted_indices]
    sorted_lora_ids = flat_lora_ids[sorted_indices]
    sorted_expert_slots = flat_expert_slots[sorted_indices]

    return sorted_token_ids, sorted_expert_ids, sorted_topk_weights, sorted_lora_ids, sorted_expert_slots
