# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from vllm_omni.config.stage_config import PipelineConfig, StageExecutionType, StagePipelineConfig

NEMOTRON_UNIFIED_ASR_PIPELINE = PipelineConfig(
    model_type="nemotron_unified_asr",
    model_arch="NemotronUnifiedASR",
    default_deploy_config_name="nemotron_unified_asr.yaml",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="asr",
            execution_type=StageExecutionType.LLM_GENERATION,
            supports_native_mrv2_data_plane=True,
            owns_tokenizer=True,
            input_sources=(),
            final_output=True,
            final_output_type="text",
            engine_output_type="text",
            sampling_constraints={"temperature": 0.0, "detokenize": False, "max_tokens": 1},
        ),
    ),
)
