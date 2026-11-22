# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from transformers import PretrainedConfig


class NemotronUnifiedASRConfig(PretrainedConfig):
    model_type = "nemotron_unified_asr"

    def __init__(self, asr_config=None, packed_streaming=True, **kwargs):
        super().__init__(**kwargs)
        self.asr_config = asr_config or {}
        self.packed_streaming = packed_streaming
