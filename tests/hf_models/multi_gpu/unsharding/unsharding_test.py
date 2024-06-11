import os
import tempfile

import torch
import torch.distributed
from parameterized import parameterized

from dolomite_engine.hf_models import AttentionHeadType

from ...test_common import TestCommons


class UnshardingTest(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix(TestCommons.get_attention_head_types(), ["gelu", "geglu"], [False, True])
    )
    def test_tensor_parallel_forward(
        self, attention_head_type: AttentionHeadType, activation_function: str, tensor_parallel_embeddings: bool
    ) -> None:
        self.skip_test_if_device_unavailable(torch.device("cuda"))

        gpus_per_node = torch.cuda.device_count()

        with tempfile.TemporaryDirectory() as tmp_path:
            outfile = os.path.join(tmp_path, "out.log")

            command = (
                f"torchrun --nproc_per_node {gpus_per_node} -m tests.hf_models.multi_gpu.unsharding.unsharding "
                f"--attention-head-type {attention_head_type.value} "
                f"--activation-function {activation_function} "
                f"--tmp-path {tmp_path} "
            )

            if tensor_parallel_embeddings:
                command += "--tensor-parallel-embeddings "

            command += f"|& tee {outfile}"

            os.system(command)

            log = open(outfile, "r").readlines()
            # for i in log:
            #     print(i)
            last_line = log[-1].strip()

        error = last_line.lstrip("tensor(").rsplit(",")[0]
        error = float(error)

        assert error < 5e-4, "outputs don't match for normal and tensor parallel model"
