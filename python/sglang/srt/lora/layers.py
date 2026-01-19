from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    split_tensor_along_last_dim,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.topk import TopKOutput
from sglang.srt.lora.backend.base_backend import BaseLoRABackend
from sglang.srt.lora.utils import LoRABatchInfo
from sglang.srt.utils import is_cuda, is_hip, is_cpu, cpu_has_amx_support

# Import activation functions for LoRA (following Triton runner pattern)
_is_cuda = is_cuda()
_is_hip = is_hip()
_is_cpu = is_cpu()
_is_cpu_amx_available = cpu_has_amx_support()

if _is_cuda:
    from sgl_kernel import gelu_and_mul, silu_and_mul
elif _is_cpu and _is_cpu_amx_available:
    pass
elif _is_hip:
    from vllm import _custom_ops as vllm_ops  # gelu_and_mul, silu_and_mul


class BaseLayerWithLoRA(nn.Module):
    def __init__(
        self,
        base_layer: nn.Module,
        lora_backend: BaseLoRABackend,
    ):
        super().__init__()
        self.base_layer: nn.Module = base_layer
        self.set_lora: bool = False
        self.lora_backend: BaseLoRABackend = lora_backend
        if hasattr(self.base_layer, "weight"):
            self.weight = self.base_layer.weight

    def forward(self, x: torch.Tensor):
        return self.base_layer.forward(x)

    def set_lora_info(self, *args):
        pass

    def slice_lora_a_weights(self, A: torch.Tensor, tp_rank: int):
        pass

    def slice_lora_b_weights(self, B: torch.Tensor, tp_rank: int):
        pass


class VocabParallelEmbeddingWithLoRA(BaseLayerWithLoRA):
    """
    Vocab parallel embedding layer with LoRA support (simplified for TP=1, no extra tokens).

    For embedding layers: output = base_embedding(x) + lora_B @ lora_A[x]
    where lora_A[x] is direct embedding lookup from lora_A weights.
    """

    def __init__(
        self,
        base_layer: VocabParallelEmbedding,
        lora_backend: BaseLoRABackend,
    ) -> None:
        super().__init__(base_layer, lora_backend)
        self.weight = base_layer.weight
        self.embed_dim = base_layer.embedding_dim
        self.vocab_size = base_layer.org_vocab_size

        self.output_offset = torch.tensor(
            [0, self.embed_dim],
            dtype=torch.int32,
            device=next(base_layer.parameters()).device,
        )

    def set_lora_info(
        self,
        new_embeddings_buffer: Optional[torch.Tensor],  # For extra tokens
        embedding_A_buffer: torch.Tensor,
        embedding_B_buffer: torch.Tensor,
    ):
        """Set LoRA buffers for embedding layer."""
        self.set_lora = True
        self.new_embeddings_buffer = new_embeddings_buffer
        self.embedding_A_buffer = embedding_A_buffer  # (num_loras, rank, vocab_size)
        self.embedding_B_buffer = embedding_B_buffer  # (num_loras, embed_dim, rank)

    def apply_lora(
        self, base_output: torch.Tensor, input_: torch.Tensor, batch_info
    ) -> torch.Tensor:
        """
        Apply LoRA to base embedding output.
        Formula: output = base_output + lora_B @ lora_A_embedding(input_)
        """

        # Efficient embedding lookup for LoRA A (already support extra token embedding process)
        lora_a_output = self.run_lora_a_embedding(input_, batch_info)

        # Apply LoRA B weights using backend
        lora_output = self.lora_backend.run_lora_b_sgemm(
            x=lora_a_output,
            weights=self.embedding_B_buffer,
            output_offset=self.output_offset,
            base_output=base_output,
        )
        return lora_output

    def run_lora_a_embedding(
        self, input_: torch.Tensor, batch_info: LoRABatchInfo
    ) -> torch.Tensor:
        """
        Apply LoRA A weights using efficient embedding lookup with CUDA graph support.
        Maps tokens to their corresponding LoRA adapters internally.
        It also includes added/extra token processing.
        """
        # Efficient embedding lookup for LoRA A (already support extra token embedding process)
        lora_a_output = self.lora_backend.run_lora_a_embedding(
            input_ids=input_,
            weights=self.embedding_A_buffer,
            vocab_size=self.vocab_size,
            extra_embeddings=(
                self.new_embeddings_buffer
                if hasattr(self, "new_embeddings_buffer")
                and self.new_embeddings_buffer is not None
                else None
            ),
        )

        return lora_a_output

    def extra_token_embedding(
        self, input_: torch.Tensor, base_output: torch.Tensor
    ) -> torch.Tensor:
        """
        Need to impl:

        Process extra tokens (tokens >= vocab_size) by looking up their embeddings
        from the new_embeddings_buffer and replacing them in base_output.

        Args:
            input_: (s,) token IDs
            base_output: (s, embed_dim) base embedding output to be modified in-place

        Returns:
            base_output: (s, embed_dim) modified input base_output (tensor[0,0,0,...]) with extra token embeddings
        """
        # return base_output
        raise NotImplementedError(
            "Error in sglang/python/sglang/srt/lora/layers.py - VocabParallelEmbeddingWithLoRA \n"
            "Current SGLang codebase did not support tuned lora with extra/added tokens. \n"
            "[TODO]: \n"
            "1. Refer to this commit: https://github.com/yushengsu-thu/sglang/commit/90415211eee8a28a316de262583d4d33fa615d10#diff-191177438bcc223837963de63c005850371f8c8a860acb153b26744b66ecc623 to complete \n"
            "2. And then you need to modified the en/decoder tokenizer - tokenizer_manager.py to support extra_token_embedding in-place. \n"
        )

    def forward(self, input_: torch.Tensor):
        """
        Forward pass with LoRA support and CUDA graph compatibility.

        Extra tokens (tokens >= vocab_size) are now handled efficiently
        in the backend's run_lora_a_embedding method.
        """
        batch_info = self.lora_backend.batch_info

        # Get base embedding output
        # For tokens >= vocab_size, base_layer will clamp or handle them
        # We mask them to 0 to avoid out-of-bounds access
        added_tokens_mask = input_ > self.vocab_size - 1
        base_output = self.base_layer.forward(input_.masked_fill(added_tokens_mask, 0))

        # [TODO] SGLang did not support extra/added token process; thus, self.extra_token_embedding only return original input_ now
        # Extra tokens - It will replace extra token embedding with self.new_embeddings_buffer's emb (Default is 0)
        if (
            hasattr(self, "new_embeddings_buffer")
            and self.new_embeddings_buffer is not None
        ):
            base_output = self.extra_token_embedding(input_, base_output)

        # Apply LoRA if configured
        if self.set_lora:
            # The backend's run_lora_a_embedding now handles both regular
            # and extra tokens efficiently with CUDA graph support
            base_output = self.apply_lora(base_output, input_, batch_info)

        return base_output

    def slice_lora_a_weights(self, A: torch.Tensor, tp_rank: int):
        # For TP=1, no slicing needed
        # LoRA A weights (rank, vocab_size) are not sliced for embedding
        # For TP>1, Need to modify code in: sglang/python/sglang/srt/lora/mem_pool.py
        # return A
        if tp_rank > 1:
            raise NotImplementedError(
                f"VocabParallelEmbeddingWithLoRA does not support tensor parallelism > 1. "
                f"Got tp_size={tp_rank}"
            )

    def slice_lora_b_weights(self, B: torch.Tensor, tp_rank: int):
        # For TP=1, no slicing needed
        # LoRA B weights (embedding_dim, rank) would be sliced along embedding dimension for TP>1
        # For TP>1, Need to modify code in: sglang/python/sglang/srt/lora/mem_pool.py
        # return B
        if tp_rank > 1:
            raise NotImplementedError(
                f"VocabParallelEmbeddingWithLoRA does not support tensor parallelism > 1. "
                f"Got tp_size={tp_rank}"
            )


class ParallelLMHeadWithLoRA(BaseLayerWithLoRA):
    """
    Parallel LM Head layer with LoRA support (simplified for TP=1).

    The LM head computes logits = hidden_states @ (W + B @ A)^T
    """

    def __init__(
        self,
        base_layer: ParallelLMHead,
        lora_backend: BaseLoRABackend,
    ) -> None:
        super().__init__(base_layer, lora_backend)
        self.weight = base_layer.weight
        self.embed_dim = base_layer.embedding_dim
        self.vocab_size = base_layer.org_vocab_size
        self.output_offset = torch.tensor(
            [0, self.vocab_size],
            dtype=torch.int32,
            device=next(base_layer.parameters()).device,
        )

    def set_lora_info(
        self,
        lm_head_A_buffer: torch.Tensor,
        lm_head_B_buffer: torch.Tensor,
    ):
        """Set LoRA buffers for LM head layer."""
        self.set_lora = True
        self.lm_head_A_buffer = lm_head_A_buffer  # (num_loras, rank, hidden_dim)
        self.lm_head_B_buffer = lm_head_B_buffer  # (num_loras, vocab_size, rank)

    def apply_lora(
        self, base_output: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply LoRA to LM head layer.

        For LM head: output = hidden @ (W + B @ A)^T
                           = hidden @ W^T + hidden @ A^T @ B^T
                           = base_output + (hidden @ A^T) @ B^T
        """
        # Apply lora_A^T: hidden_states @ A^T
        lora_a_output = self.lora_backend.run_lora_a_sgemm(
            hidden_states, self.lm_head_A_buffer
        )

        # Apply lora_B^T: lora_a_output @ B^T
        lora_output = self.lora_backend.run_lora_b_sgemm(
            x=lora_a_output,
            weights=self.lm_head_B_buffer,
            output_offset=self.output_offset,
            base_output=base_output,
        )

        return lora_output

    def forward(self, hidden_states: torch.Tensor):
        # Apply base linear transformation
        base_output = F.linear(
            hidden_states, self.weight, bias=getattr(self.base_layer, "bias", None)
        )

        # Apply LoRA if set
        if self.set_lora:
            base_output = self.apply_lora(base_output, hidden_states)

        return base_output

    def slice_lora_a_weights(self, A: torch.Tensor, tp_rank: int):
        # For TP=1, no slicing needed
        # For TP>1, need to modify code in: sglang/python/sglang/srt/lora/mem_pool.py
        # return A
        if tp_rank > 1:
            raise NotImplementedError(
                f"ParallelLMHeadWithLoRA does not support tensor parallelism > 1. "
                f"Got tp_size={tp_rank}"
            )

    def slice_lora_b_weights(self, B: torch.Tensor, tp_rank: int):
        # For TP=1, no slicing needed
        # For TP>1, would slice along vocab dimension, need to modify code in: sglang/python/sglang/srt/lora/mem_pool.py
        # return B
        if tp_rank > 1:
            raise NotImplementedError(
                f"ParallelLMHeadWithLoRA does not support tensor parallelism > 1. "
                f"Got tp_size={tp_rank}"
            )


class ColumnParallelLinearWithLoRA(BaseLayerWithLoRA):
    def __init__(
        self,
        base_layer: ColumnParallelLinear,
        lora_backend: BaseLoRABackend,
    ) -> None:
        super().__init__(base_layer, lora_backend)
        shard_size = self.base_layer.output_partition_sizes[0]
        self.output_offset = torch.tensor(
            [
                0,
                shard_size,
            ],
            dtype=torch.int32,
            device=next(self.base_layer.parameters()).device,
        )

    def set_lora_info(
        self,
        A_buffer: torch.Tensor,
        B_buffer: torch.Tensor,
    ):
        self.set_lora = True
        self.A_buffer = A_buffer
        self.B_buffer = B_buffer

    def apply_lora(self, base_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        lora_a_output = self.lora_backend.run_lora_a_sgemm(x, self.A_buffer)
        lora_output = self.lora_backend.run_lora_b_sgemm(
            x=lora_a_output,
            weights=self.B_buffer,
            output_offset=self.output_offset,
            base_output=base_output,
        )
        return lora_output

    def forward(self, input_: torch.Tensor):
        # duplicate the logic in ColumnParallelLinear
        bias = self.base_layer.bias if not self.base_layer.skip_bias_add else None
        output_parallel = self.base_layer.quant_method.apply(
            self.base_layer, input_, bias
        )

        if self.set_lora:
            output_parallel = self.apply_lora(output_parallel, input_)

        if self.base_layer.gather_output:
            output = tensor_model_parallel_all_gather(output_parallel)
        else:
            output = output_parallel
        output_bias = self.base_layer.bias if self.base_layer.skip_bias_add else None
        return output, output_bias

    def slice_lora_a_weights(self, A: torch.Tensor, tp_rank: int):
        return A

    def slice_lora_b_weights(self, B: torch.Tensor, tp_rank: int):
        shard_size = self.base_layer.output_partition_sizes[0]
        start_idx = tp_rank * shard_size
        end_idx = (tp_rank + 1) * shard_size
        B = B[start_idx:end_idx, :]
        return B


class MergedColumnParallelLinearWithLoRA(ColumnParallelLinearWithLoRA):
    def __init__(
        self,
        base_layer: MergedColumnParallelLinear,
        lora_backend: BaseLoRABackend,
    ) -> None:
        super().__init__(base_layer, lora_backend)

    def set_lora_info(
        self,
        A_buffer: torch.Tensor,
        B_buffer: torch.Tensor,
    ):
        self.set_lora = True
        self.A_buffer_gate_up = A_buffer
        self.B_buffer_gate_up = B_buffer

        shard_size = self.base_layer.output_partition_sizes[0]
        self.output_offset = torch.tensor(
            [
                0,
                shard_size,
                2 * shard_size,
            ],
            dtype=torch.int32,
            device=next(self.base_layer.parameters()).device,
        )

    def apply_lora(self, base_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        lora_output = self.lora_backend.run_gate_up_lora(
            x=x,
            gate_up_lora_a=self.A_buffer_gate_up,
            gate_up_lora_b=self.B_buffer_gate_up,
            output_offset=self.output_offset,
            base_output=base_output,
        )
        return lora_output

    def slice_lora_a_weights(self, A: torch.Tensor, tp_rank: int):
        return A

    def slice_lora_b_weights(self, B: torch.Tensor, tp_rank: int):
        # Since the outputs for both gate and up are identical, we use a random one.
        shard_size = self.base_layer.output_partition_sizes[0]
        gate_size = self.base_layer.output_sizes[0]
        start_idx = tp_rank * shard_size
        end_idx = (tp_rank + 1) * shard_size
        return torch.concat(
            (
                B[start_idx:end_idx, :],
                B[gate_size + start_idx : gate_size + end_idx],
            ),
            dim=0,
        )


class QKVParallelLinearWithLoRA(ColumnParallelLinearWithLoRA):
    def __init__(
        self,
        base_layer: QKVParallelLinear,
        lora_backend: BaseLoRABackend,
    ) -> None:
        super().__init__(base_layer, lora_backend)
        q_proj_shard_size = self.base_layer.q_proj_shard_size
        kv_proj_shard_size = self.base_layer.kv_proj_shard_size
        self.output_offset = torch.tensor(
            [
                0,
                q_proj_shard_size,
                q_proj_shard_size + kv_proj_shard_size,
                q_proj_shard_size + 2 * kv_proj_shard_size,
            ],
            dtype=torch.int32,
            device=next(self.base_layer.parameters()).device,
        )
        self.output_offset_cpu = self.output_offset.cpu()

        # For computing number of launched blocks
        self.max_qkv_out_dim = max(q_proj_shard_size, kv_proj_shard_size)

    def set_lora_info(
        self,
        A_buffer_qkv: torch.Tensor,
        B_buffer_qkv: torch.Tensor,
    ):
        self.set_lora = True
        self.A_buffer_qkv = A_buffer_qkv
        self.B_buffer_qkv = B_buffer_qkv

    def apply_lora(self, base_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        lora_output = self.lora_backend.run_qkv_lora(
            x=x,
            qkv_lora_a=self.A_buffer_qkv,
            qkv_lora_b=self.B_buffer_qkv,
            base_output=base_output,
            output_offset=self.output_offset,
            output_offset_cpu=self.output_offset_cpu,
            max_qkv_out_dim=self.max_qkv_out_dim,
        )

        return lora_output

    def slice_lora_a_weights(self, A: torch.Tensor, tp_rank: int):
        return A

    def slice_lora_b_weights(self, B: torch.Tensor, tp_rank: int) -> torch.Tensor:
        base_layer = self.base_layer
        q_proj_shard_size = base_layer.q_proj_shard_size
        kv_proj_shard_size = base_layer.kv_proj_shard_size
        num_kv_head_replicas = base_layer.num_kv_head_replicas

        q_start_idx = q_proj_shard_size * tp_rank
        q_end_idx = q_start_idx + q_proj_shard_size

        kv_shard_id = tp_rank // num_kv_head_replicas
        kv_start_idx = kv_proj_shard_size * kv_shard_id
        kv_end_idx = kv_start_idx + kv_proj_shard_size

        # Use total sizes for indexing into the LoRA B tensor (which has total dims)
        # output_sizes is buggy for GQA when tp_size >= num_kv_heads
        head_size = base_layer.head_size
        q_size = base_layer.total_num_heads * head_size
        k_size = base_layer.total_num_kv_heads * head_size

        B_q_shard = B[q_start_idx:q_end_idx, :]
        B_k_shard = B[q_size + kv_start_idx : q_size + kv_end_idx, :]
        B_v_shard = B[q_size + k_size + kv_start_idx : q_size + k_size + kv_end_idx, :]

        return torch.concat(
            (
                B_q_shard,
                B_k_shard,
                B_v_shard,
            ),
            dim=0,
        )


class RowParallelLinearWithLoRA(BaseLayerWithLoRA):
    def __init__(
        self,
        base_layer: RowParallelLinear,
        lora_backend: BaseLoRABackend,
    ) -> None:
        super().__init__(base_layer, lora_backend)

    def set_lora_info(self, A_buffer: torch.Tensor, B_buffer: torch.Tensor):
        self.set_lora = True
        self.A_buffer = A_buffer
        self.B_buffer = B_buffer
        output_size = self.base_layer.output_size
        self.output_offset = torch.tensor(
            [
                0,
                output_size,
            ],
            dtype=torch.int32,
            device=next(self.base_layer.parameters()).device,
        )

    def apply_lora(self, base_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        lora_a_output = self.lora_backend.run_lora_a_sgemm(x, self.A_buffer)
        lora_output = self.lora_backend.run_lora_b_sgemm(
            x=lora_a_output,
            weights=self.B_buffer,
            output_offset=self.output_offset,
            base_output=base_output,
        )
        return lora_output

    def forward(self, input_: torch.Tensor, skip_all_reduce=False):
        # duplicate the logic in RowParallelLinear
        if self.base_layer.input_is_parallel:
            input_parallel = input_
        else:
            tp_rank = get_tensor_model_parallel_rank()
            splitted_input = split_tensor_along_last_dim(
                input_, num_partitions=self.base_layer.tp_size
            )
            input_parallel = splitted_input[tp_rank].contiguous()
        output_parallel = self.base_layer.quant_method.apply(
            self.base_layer, input_parallel
        )

        if self.set_lora:
            output_parallel = self.apply_lora(output_parallel, input_parallel)

        if (
            self.base_layer.reduce_results
            and self.base_layer.tp_size > 1
            and not skip_all_reduce
        ):
            output_ = tensor_model_parallel_all_reduce(output_parallel)
        else:
            output_ = output_parallel

        if not self.base_layer.skip_bias_add:
            output = (
                output_ + self.base_layer.bias
                if self.base_layer.bias is not None
                else output_
            )
            output_bias = None
        else:
            output = output_
            output_bias = self.base_layer.bias
        return output, output_bias

    def slice_lora_a_weights(self, A: torch.Tensor, tp_rank: int):
        shard_size = self.base_layer.input_size_per_partition
        start_idx = tp_rank * shard_size
        end_idx = (tp_rank + 1) * shard_size
        A = A[:, start_idx:end_idx].contiguous()
        return A

    def slice_lora_b_weights(self, B: torch.Tensor, tp_rank: int):
        return B


class FusedMoEWithLoRA(BaseLayerWithLoRA):
    """
    Wrapper around FusedMoE that adds LoRA computation with proper activation handling.

    Key Design: LoRA must be injected BEFORE the activation function.
    Since silu(base + lora) != silu(base) + silu(lora), we cannot compute
    base and LoRA paths separately and add them at the end.

    Instead, we:
    1. Run gate_up GEMM (base) -> intermediate_cache1
    2. Add gate_up LoRA delta to intermediate_cache1
    3. Apply activation to (base + lora) -> intermediate_cache2
    4. Run down GEMM (base) -> intermediate_cache3
    5. Add down LoRA delta to output
    6. Combine expert outputs
    """

    USE_MERGE_WEIGHTS_PATH = True

    def __init__(
        self,
        base_layer: nn.Module,
        lora_backend: BaseLoRABackend,
    ):
        super().__init__(base_layer, lora_backend)
        # LoRA tensors will be set by LoRAManager
        self.gate_up_lora_a_weights = None
        self.gate_up_lora_b_weights = None
        self.down_lora_a_weights = None
        self.down_lora_b_weights = None

    def set_lora_info(
        self,
        gate_up_lora_a_weights: torch.Tensor,
        gate_up_lora_b_weights: torch.Tensor,
        down_lora_a_weights: torch.Tensor = None,
        down_lora_b_weights: torch.Tensor = None,
    ):
        """Set LoRA weight tensors from memory pool."""
        self.set_lora = True
        self.gate_up_lora_a_weights = gate_up_lora_a_weights
        self.gate_up_lora_b_weights = gate_up_lora_b_weights
        self.down_lora_a_weights = down_lora_a_weights
        self.down_lora_b_weights = down_lora_b_weights

    def forward(self, hidden_states: torch.Tensor, topk_output: TopKOutput, **kwargs):
        """
        Forward pass with LoRA injection BEFORE activation.

        This supports two paths:
        1. Merge weights path: When all tokens use the same LoRA adapter,
           merge LoRA weights into base weights and run base forward.
           This is faster than the per-expert kernel path.
        2. Per-expert kernel path: When multiple adapters are in the batch,
           use the Triton kernel for per-expert LoRA computation.

        The merge path is critical for correctness because:
        - silu(base + lora) != silu(base) + silu(lora)
        - We must add LoRA delta before activation
        """
        if not self.set_lora or self.gate_up_lora_a_weights is None:
            # No LoRA, use base layer directly
            return self.base_layer.forward(hidden_states, topk_output, **kwargs)

        # Check if we can use the faster merge weights path
        # This requires all tokens to use the same LoRA adapter
        forward_batch = self.lora_backend.forward_batch
        lora_indices = forward_batch.token_lora_indices
        forward_mode = forward_batch.forward_mode

        if self.USE_MERGE_WEIGHTS_PATH and forward_mode.is_prefill():
            unique_adapters = lora_indices.unique()
            if len(unique_adapters) == 1 and unique_adapters[0].item() >= 0:
                # Single LoRA adapter, merge path enabled, and prefill - use merge weights path
                # Single adapter, merge path enabled, and prefill - use merge weights path
                lora_id = unique_adapters[0].item()
                return self._forward_with_merged_weights(
                    hidden_states, topk_output, lora_id, **kwargs
                )
        # Multiple adapters, mixed batch, merge path disabled, or decode - use injection path
        return self._forward_with_lora_injection(hidden_states, topk_output, **kwargs)

    def _forward_with_lora_injection(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass with LoRA injection BEFORE activation.

        This replicates the key parts of fused_experts_impl but injects LoRA
        at the correct points to ensure silu(base + lora) instead of silu(base) + silu(lora).
        """
        from sglang.srt.layers.moe.fused_moe_triton.fused_moe import (
            get_config_dtype_str,
            invoke_fused_moe_kernel,
            moe_align_block_size,
            moe_sum_reduce_triton,
            try_get_optimal_moe_config,
        )
        from sglang.srt.lora.moe_dispatch import moe_dispatch
        from sglang.srt.lora.triton_ops.per_expert_lora_moe import (
            per_expert_lora_forward,
        )
        import triton.language as tl
        import functools

        # Get base layer attributes
        base_layer = self.base_layer
        w13 = base_layer.w13_weight
        w2 = base_layer.w2_weight

        # Get topk info
        topk_ids = topk_output.topk_ids
        topk_weights = topk_output.topk_weights

        # Get LoRA batch info
        batch_info = self.lora_backend.batch_info
        lora_ranks = batch_info.lora_ranks
        scalings = batch_info.scalings
        lora_indices = self.lora_backend.forward_batch.token_lora_indices

        # Setup computation parameters
        num_tokens, hidden_size = hidden_states.shape
        E, N, _ = w13.shape
        top_k = topk_ids.shape[1]
        num_experts = base_layer.num_experts

        compute_type = tl.bfloat16 if hidden_states.dtype == torch.bfloat16 else tl.float16

        # Get optimal config
        config_dtype = get_config_dtype_str(
            use_fp8_w8a8=False,
            use_int8_w8a8=False,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            dtype=hidden_states.dtype,
        )

        get_config_func = functools.partial(
            try_get_optimal_moe_config,
            w13.shape,
            (w2.shape[0], w2.shape[1], w2.shape[2]),
            top_k,
            config_dtype,
            block_shape=None,
            per_channel_quant=False,
            return_down_config=True,
        )

        config, (down_config, _) = get_config_func(num_tokens)

        # Align block size for efficient kernel execution
        sorted_token_ids, expert_ids_aligned, num_tokens_post_padded = moe_align_block_size(
            topk_ids, config["BLOCK_SIZE_M"], E
        )

        # Allocate intermediate caches
        # Use flat 2D shape like the base implementation for kernel compatibility
        total_tokens = num_tokens * top_k
        intermediate_cache1 = torch.empty(
            (total_tokens, N),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        # ===== Stage 1: Gate-up GEMM (base) =====
        invoke_fused_moe_kernel(
            hidden_states,
            w13,
            None,  # bias
            intermediate_cache1,
            None,  # a_scale
            None,  # w_scale
            None,  # w_zp
            topk_weights,
            topk_ids,
            sorted_token_ids,
            expert_ids_aligned,
            num_tokens_post_padded,
            False,  # apply_router_weight_on_input
            top_k,
            config,
            compute_type=compute_type,
            use_fp8_w8a8=False,
            use_int8_w8a8=False,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            per_channel_quant=False,
            block_shape=None,
        )

        # ===== Stage 2: Add gate_up LoRA delta to intermediate_cache1 BEFORE activation =====
        # This is the key difference from the incorrect approach!
        # We add LoRA to the base output before activation so that
        # activation(base + lora) is computed correctly.

        # Dispatch tokens to experts for LoRA computation
        token_ids, expert_ids, sorted_topk_weights, lora_ids, expert_slots = moe_dispatch(
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            lora_indices=lora_indices,
        )

        # Compute gate_up LoRA delta
        # Shape: (num_dispatched, gate_up_dim)
        num_dispatched = token_ids.shape[0]
        _, _, gate_up_dim, _ = self.gate_up_lora_b_weights.shape
        intermediate_dim = gate_up_dim // 2

        lora_gate_up_delta = torch.zeros(
            (num_dispatched, gate_up_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        per_expert_lora_forward(
            hidden_states=hidden_states,
            lora_a_weights=self.gate_up_lora_a_weights,
            lora_b_weights=self.gate_up_lora_b_weights,
            token_ids=token_ids,
            expert_ids=expert_ids,
            lora_ids=lora_ids,
            lora_ranks=lora_ranks,
            lora_scalings=scalings,
            num_experts=num_experts,
            base_output=lora_gate_up_delta,
            is_down_proj=False,
        )

        # Add LoRA delta to intermediate_cache1
        # intermediate_cache1 already has shape (total_tokens, N)
        intermediate_cache1_flat = intermediate_cache1

        # Compute flat indices using expert_slots directly from moe_dispatch
        # flat_indices = token_id * top_k + expert_slot
        flat_indices = token_ids * top_k + expert_slots  # (num_dispatched,)

        # Add LoRA delta to the corresponding positions in intermediate_cache1
        intermediate_cache1_flat.index_add_(
            0,
            flat_indices,
            lora_gate_up_delta,
        )

        # ===== Stage 3: Apply activation to (base + lora) =====
        intermediate_cache2 = torch.empty(
            (total_tokens, N // 2),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        activation = base_layer.moe_runner_config.activation
        if activation == "silu":
            if _is_cuda:
                silu_and_mul(intermediate_cache1_flat, intermediate_cache2)
            elif _is_hip:
                vllm_ops.silu_and_mul(intermediate_cache2, intermediate_cache1_flat)
            else:
                raise ValueError(f"Unsupported platform for activation: {activation}")
        elif activation == "gelu":
            if _is_cuda:
                gelu_and_mul(intermediate_cache1_flat, intermediate_cache2)
            elif _is_hip:
                vllm_ops.gelu_and_mul(intermediate_cache2, intermediate_cache1_flat)
            else:
                raise ValueError(f"Unsupported platform for activation: {activation}")
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # ===== Stage 4: Down GEMM (base) =====
        intermediate_cache3 = torch.empty(
            (num_tokens, top_k, w2.shape[1]),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        out_hidden_states = torch.empty_like(hidden_states)

        invoke_fused_moe_kernel(
            intermediate_cache2,
            w2,
            None,  # bias
            intermediate_cache3 if top_k != 1 else out_hidden_states.unsqueeze(0),
            None,  # a_scale
            None,  # w_scale
            None,  # w_zp
            topk_weights,
            topk_ids,
            sorted_token_ids,
            expert_ids_aligned,
            num_tokens_post_padded,
            True,  # apply_router_weight (on output for down proj)
            1,
            down_config or config,  # Use down_config if available
            compute_type=compute_type,
            use_fp8_w8a8=False,
            use_int8_w8a8=False,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            per_channel_quant=False,
            block_shape=None,
        )

        # ===== Stage 5: Add down LoRA delta =====
        if self.down_lora_a_weights is not None:
            # Compute down LoRA on the activated intermediate (after silu_and_mul)
            # We use the same dispatched pairs from earlier
            lora_down_delta = torch.zeros(
                (num_dispatched, hidden_size),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )

            # Get the activated intermediate for each dispatched pair
            # intermediate_cache2 has shape (total_tokens, N // 2)
            # We need to index it by flat_indices
            dispatched_intermediate = intermediate_cache2[flat_indices]  # (num_dispatched, N // 2)

            # IMPORTANT: For down LoRA, the hidden_states input is dispatched_intermediate
            # which has shape (num_dispatched, N // 2). The kernel uses token_ids to index
            # into this tensor. Since each row already corresponds to a dispatched pair,
            # we need to use sequential indices (0 to num_dispatched-1) as token_ids.
            sequential_token_ids = torch.arange(
                num_dispatched, device=hidden_states.device, dtype=token_ids.dtype
            )

            per_expert_lora_forward(
                hidden_states=dispatched_intermediate,
                lora_a_weights=self.down_lora_a_weights,
                lora_b_weights=self.down_lora_b_weights,
                token_ids=sequential_token_ids,  # Use sequential indices, not original token_ids
                expert_ids=expert_ids,
                lora_ids=lora_ids,
                lora_ranks=lora_ranks,
                lora_scalings=scalings,
                num_experts=num_experts,
                base_output=lora_down_delta,
                is_down_proj=True,
            )

            # Apply router weights and add to output
            # intermediate_cache3 has shape (num_tokens, top_k, hidden_size)
            # Add LoRA delta weighted by topk_weights

            # First, weight the LoRA delta by router weights
            weighted_lora_down = lora_down_delta * sorted_topk_weights.unsqueeze(-1)

            # Add to intermediate_cache3 at the correct positions
            intermediate_cache3_flat = intermediate_cache3.view(-1, hidden_size)
            intermediate_cache3_flat.index_add_(
                0,
                flat_indices,
                weighted_lora_down.to(intermediate_cache3_flat.dtype),
            )

        # ===== Stage 6: Combine expert outputs =====
        if top_k == 1:
            # Already written to out_hidden_states
            pass
        elif top_k == 2:
            torch.add(
                intermediate_cache3[:, 0],
                intermediate_cache3[:, 1],
                out=out_hidden_states,
            ).squeeze(dim=1)
        else:
            if _is_cuda:
                if num_tokens <= 32:
                    from sglang.srt.layers.moe.fused_moe_triton.fused_moe import (
                        moe_sum_reduce_torch_compile,
                    )
                    moe_sum_reduce_torch_compile(
                        intermediate_cache3,
                        out_hidden_states,
                        1.0,  # routed_scaling_factor
                    )
                else:
                    moe_sum_reduce_triton(
                        intermediate_cache3,
                        out_hidden_states,
                        1.0,  # routed_scaling_factor
                    )
            else:
                moe_sum_reduce_triton(
                    intermediate_cache3,
                    out_hidden_states,
                    1.0,
                )

        # Handle reduce_results if needed
        if base_layer.reduce_results and (base_layer.moe_tp_size > 1 or base_layer.moe_ep_size > 1):
            from sglang.srt.distributed import tensor_model_parallel_all_reduce
            out_hidden_states = tensor_model_parallel_all_reduce(out_hidden_states)

        return out_hidden_states

    def _forward_with_merged_weights(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
        lora_id: int,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass that replicates the base layer MoE computation.
        """
        from sglang.srt.layers.moe.fused_moe_triton.fused_moe import (
            get_config_dtype_str,
            invoke_fused_moe_kernel,
            moe_align_block_size,
            moe_sum_reduce_triton,
            try_get_optimal_moe_config,
        )

        import triton.language as tl
        import functools
        import einops

        # Get base layer attributes
        base_layer = self.base_layer
        w13 = base_layer.w13_weight
        w2 = base_layer.w2_weight

        # Get topk info
        topk_ids = topk_output.topk_ids
        topk_weights = topk_output.topk_weights

        # Setup computation parameters
        num_tokens, hidden_size = hidden_states.shape
        E, N, _ = w13.shape
        top_k = topk_ids.shape[1]
        # num_experts = base_layer.num_experts

        scaling_factor = self.lora_backend.batch_info.scalings[lora_id]
        compute_type = tl.bfloat16 if hidden_states.dtype == torch.bfloat16 else tl.float16

        # Get optimal config
        config_dtype = get_config_dtype_str(
            use_fp8_w8a8=False,
            use_int8_w8a8=False,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            dtype=hidden_states.dtype,
        )

        get_config_func = functools.partial(
            try_get_optimal_moe_config,
            w13.shape,
            (w2.shape[0], w2.shape[1], w2.shape[2]),
            top_k,
            config_dtype,
            block_shape=None,
            per_channel_quant=False,
            return_down_config=True,
        )

        config, (down_config, _) = get_config_func(num_tokens)

        # Align block size for efficient kernel execution
        sorted_token_ids, expert_ids_aligned, num_tokens_post_padded = moe_align_block_size(
            topk_ids, config["BLOCK_SIZE_M"], E
        )

        # Allocate intermediate caches
        # Use flat 2D shape like the base implementation for kernel compatibility
        total_tokens = num_tokens * top_k
        intermediate_cache1 = torch.empty(
            (total_tokens, N),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        # ===== Stage 1: Gate-up GEMM (base) =====
        invoke_fused_moe_kernel(
            hidden_states,
            w13 + einops.einsum(
                scaling_factor * self.gate_up_lora_a_weights[lora_id],
                self.gate_up_lora_b_weights[lora_id],
                "b r i, b o r -> b o i",
            ),
            None,  # bias
            intermediate_cache1,
            None,  # a_scale
            None,  # w_scale
            None,  # w_zp
            topk_weights,
            topk_ids,
            sorted_token_ids,
            expert_ids_aligned,
            num_tokens_post_padded,
            False,  # apply_router_weight_on_input
            top_k,
            config,
            compute_type=compute_type,
            use_fp8_w8a8=False,
            use_int8_w8a8=False,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            per_channel_quant=False,
            block_shape=None,
        )

        # ===== Stage 3: Apply activation (to base + lora) =====
        intermediate_cache2 = torch.empty(
            (total_tokens, N // 2),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        activation = base_layer.moe_runner_config.activation
        if activation == "silu":
            if _is_cuda:
                silu_and_mul(intermediate_cache1, intermediate_cache2)
            elif _is_hip:
                vllm_ops.silu_and_mul(intermediate_cache2, intermediate_cache1)
            else:
                raise ValueError(f"Unsupported platform for activation: {activation}")
        elif activation == "gelu":
            if _is_cuda:
                gelu_and_mul(intermediate_cache1, intermediate_cache2)
            elif _is_hip:
                vllm_ops.gelu_and_mul(intermediate_cache2, intermediate_cache1)
            else:
                raise ValueError(f"Unsupported platform for activation: {activation}")
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # ===== Stage 4: Down GEMM (base) =====
        intermediate_cache3 = torch.empty(
            (num_tokens, top_k, w2.shape[1]),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        out_hidden_states = torch.empty_like(hidden_states)

        invoke_fused_moe_kernel(
            intermediate_cache2,
            w2 + einops.einsum(
                scaling_factor * self.down_lora_a_weights[lora_id],
                self.down_lora_b_weights[lora_id],
                "b r i, b o r -> b o i",
            ),
            None,  # bias
            intermediate_cache3 if top_k != 1 else out_hidden_states.unsqueeze(0),
            None,  # a_scale
            None,  # w_scale
            None,  # w_zp
            topk_weights,
            topk_ids,
            sorted_token_ids,
            expert_ids_aligned,
            num_tokens_post_padded,
            True,  # apply_router_weight (on output for down proj)
            1,
            down_config or config,  # Use down_config if available
            compute_type=compute_type,
            use_fp8_w8a8=False,
            use_int8_w8a8=False,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            per_channel_quant=False,
            block_shape=None,
        )

        # ===== Stage 6: Combine expert outputs =====
        if top_k == 1:
            # Already written to out_hidden_states
            pass
        elif top_k == 2:
            torch.add(
                intermediate_cache3[:, 0],
                intermediate_cache3[:, 1],
                out=out_hidden_states,
            ).squeeze(dim=1)
        else:
            if _is_cuda:
                if num_tokens <= 32:
                    from sglang.srt.layers.moe.fused_moe_triton.fused_moe import (
                        moe_sum_reduce_torch_compile,
                    )
                    moe_sum_reduce_torch_compile(
                        intermediate_cache3,
                        out_hidden_states,
                        1.0,  # routed_scaling_factor
                    )
                else:
                    moe_sum_reduce_triton(
                        intermediate_cache3,
                        out_hidden_states,
                        1.0,  # routed_scaling_factor
                    )
            else:
                moe_sum_reduce_triton(
                    intermediate_cache3,
                    out_hidden_states,
                    1.0,
                )

        # Handle reduce_results if needed
        if base_layer.reduce_results and (base_layer.moe_tp_size > 1 or base_layer.moe_ep_size > 1):
            from sglang.srt.distributed import tensor_model_parallel_all_reduce
            out_hidden_states = tensor_model_parallel_all_reduce(out_hidden_states)

        return out_hidden_states

    def slice_lora_a_weights(self, A: torch.Tensor, tp_rank: int):
        """
        Slice LoRA A weights for tensor parallelism.

        For MoE layers:
        - gate_up_proj (column-parallel): A has shape [rank, hidden_size] - NO slicing
        - down_proj (row-parallel): A has shape [rank, intermediate_size] - SLICE along dim 1

        We detect by checking if A.shape[1] equals the full intermediate_size.
        """
        tp_size = self.base_layer.moe_tp_size
        if tp_size <= 1:
            return A

        intermediate_size_per_partition = self.base_layer.intermediate_size_per_partition
        full_intermediate_size = intermediate_size_per_partition * tp_size

        # If A's input dimension matches full intermediate_size, it's down_proj - slice it
        # gate_up_proj A has hidden_size which is different from intermediate_size
        if A.shape[1] == full_intermediate_size:
            start_idx = tp_rank * intermediate_size_per_partition
            end_idx = (tp_rank + 1) * intermediate_size_per_partition
            A = A[:, start_idx:end_idx].contiguous()

        return A

    def slice_lora_b_weights(self, B: torch.Tensor, tp_rank: int):
        """
        Slice LoRA B weights for tensor parallelism.

        For MoE layers:
        - gate_up_proj (column-parallel): B has shape [2*intermediate_size, rank] - SLICE along dim 0
        - down_proj (row-parallel): B has shape [hidden_size, rank] - NO slicing

        We detect by checking if B.shape[0] equals 2*full_intermediate_size.
        """
        tp_size = self.base_layer.moe_tp_size
        if tp_size <= 1:
            return B

        intermediate_size_per_partition = self.base_layer.intermediate_size_per_partition
        full_intermediate_size = intermediate_size_per_partition * tp_size
        full_gate_up_size = 2 * full_intermediate_size

        # If B's output dimension matches full gate_up_size, it's gate_up_proj - slice it
        # down_proj B has hidden_size which is different from 2*intermediate_size
        if B.shape[0] == full_gate_up_size:
            # gate_up is [gate, up] concatenated, need to slice both halves
            shard_size = intermediate_size_per_partition

            start_idx = tp_rank * shard_size
            end_idx = (tp_rank + 1) * shard_size

            # Slice gate and up portions separately and concatenate
            B_gate = B[start_idx:end_idx, :]
            B_up = B[full_intermediate_size + start_idx : full_intermediate_size + end_idx, :]
            B = torch.cat([B_gate, B_up], dim=0).contiguous()

        return B


def get_lora_layer(
    layer: nn.Module, lora_backend: BaseLoRABackend
) -> BaseLayerWithLoRA:
    # FusedMoE is now imported at the top of the file
    # FusedMoEWithLoRA is now defined in this file

    supported_layer_types = {
        # the order matters
        FusedMoE: FusedMoEWithLoRA,
        ParallelLMHead: ParallelLMHeadWithLoRA,
        VocabParallelEmbedding: VocabParallelEmbeddingWithLoRA,
        QKVParallelLinear: QKVParallelLinearWithLoRA,
        MergedColumnParallelLinear: MergedColumnParallelLinearWithLoRA,
        ColumnParallelLinear: ColumnParallelLinearWithLoRA,
        RowParallelLinear: RowParallelLinearWithLoRA,
    }
    for src_layer_type, lora_layer_type in supported_layer_types.items():
        if isinstance(layer, src_layer_type):  # pylint: disable=unidiomatic-typecheck
            ret = lora_layer_type(layer, lora_backend)
            return ret
    raise Exception(f"No corresponding LoRA layer supported for {type(layer)}.")
