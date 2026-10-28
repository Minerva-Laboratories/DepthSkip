"""Tiny random Qwen3.5-architecture model (3 DeltaNet : 1 attention, 8 layers) for CPU tests; random token ids."""
import pytest
import torch

from depthskip import Model, use_reference_kernels

use_reference_kernels(True)


def tiny_model(seed: int = 0, prefill_chunk: int = 0) -> Model:
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
    torch.manual_seed(seed)
    c = Qwen3_5TextConfig(
        vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=8,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=16,
        layer_types=["linear_attention"] * 3 + ["full_attention"] + ["linear_attention"] * 3 + ["full_attention"],
        tie_word_embeddings=True, pad_token_id=0)
    m = Qwen3_5ForCausalLM(c).float().eval()
    return Model.wrap(m, None, prefill_chunk)


def seqs(n: int, lens=None, seed: int = 1) -> list[list[int]]:
    g = torch.Generator().manual_seed(seed)
    lens = lens or [30] * n
    return [torch.randint(1, 512, (L,), generator=g).tolist() for L in lens]


@pytest.fixture
def model():
    return tiny_model()
