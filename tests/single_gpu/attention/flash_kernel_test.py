import os

import torch
from parameterized import parameterized
from transformers import set_seed

from dolomite_engine.hf_models.modeling_utils import flash_attention

from ...test_common import TestCommons


SEED = 1234


class FlashKernelTest(TestCommons):
    @parameterized.expand(TestCommons.make_args_matrix([torch.device("cuda")], [torch.float16, torch.bfloat16]))
    def test_flash_kernel(self, device: torch.device, torch_dtype: torch.dtype) -> None:
        self.skip_test_if_device_unavailable(device)

        set_seed(SEED)

        query_pytorch = torch.randn(4, 1024, 32, 128, device=device, requires_grad=True, dtype=torch_dtype)
        key_pytorch = torch.randn(4, 1024, 32, 128, device=device, requires_grad=True, dtype=torch_dtype)
        value_pytorch = torch.randn(4, 1024, 32, 128, device=device, requires_grad=True, dtype=torch_dtype)

        os.environ.update({"PYTORCH_NATIVE_FLASH_KERNEL": "1"})
        attention_pytorch = flash_attention(
            query_pytorch,
            key_pytorch,
            value_pytorch,
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            max_seqlen_q=None,
            max_seqlen_k=None,
            dropout_p=0,
            softmax_scale=None,
            causal=True,
        )

        query_tri_dao = query_pytorch.clone()
        key_tri_dao = key_pytorch.clone()
        value_tri_dao = value_pytorch.clone()

        os.environ.update({"PYTORCH_NATIVE_FLASH_KERNEL": "0"})
        attention_tri_dao = flash_attention(
            query_tri_dao,
            key_tri_dao,
            value_tri_dao,
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            max_seqlen_q=None,
            max_seqlen_k=None,
            dropout_p=0,
            softmax_scale=None,
            causal=True,
        )

        self.assert_equal_tensors(attention_pytorch, attention_tri_dao, exact_match=True)

        attention_pytorch.mean().backward()
        attention_tri_dao.mean().backward()

        self.assert_equal_tensors(query_pytorch.grad, query_tri_dao.grad, exact_match=True)
        self.assert_equal_tensors(query_pytorch.grad, key_tri_dao.grad, exact_match=True)
        self.assert_equal_tensors(query_pytorch.grad, value_tri_dao.grad, exact_match=True)
