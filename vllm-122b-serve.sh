#!/bin/bash 
## vLLM serve script for Qwen3.5-122B-A10B-NVFP4 on Jetson Thor # Runs natively (no container) with sm_110a compiled vLLM 0.19.0
# Activate venv
source ~/.vllm/bin/activate

# Thor-specific environment
export TORCH_CUDA_ARCH_LIST=11.0a
export CUDA_HOME=/usr/local/cuda-13
export TRITON_PTXAS_PATH=/usr/local/cuda-13/bin/ptxas
#export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/lib:$LD_LIBRARY_PATH

# Don't hit HuggingFace network — everything is local
#export HF_HUB_OFFLINE=1
export TORCHINDUCTOR_COMPILE_THREADS=2
export TRITON_NUM_COMPILATION_WORKERS=2

#MODEL_PATH=$HOME/.cache/huggingface/hub/models--Sehyo--Qwen3.5-122B-A10B-NVFP4/snapshots/56a6bdda33285ba2d5688e4f71f6c714649497b4

#exec vllm serve Sehyo/Qwen3.5-122B-A10B-NVFP4 \
exec vllm serve Qwen/Qwen3.5-122B-A10B-GPTQ-Int4 \
		--served-model-name Qwen/Qwen3.5-122B-A10B-GPTQ-Int4 \
		--served-model-name qwen3.5:122b \
    --host 0.0.0.0 \
    --port 9000 \
		--attention-backend FLASHINFER \
    --tensor-parallel-size 1 \
    --max-model-len 150000 \
    --trust-remote-code \
    --gpu-memory-utilization 0.95 \
    --kv-cache-dtype fp8 \
    --enable-prefix-caching \
		--max-num-batched-tokens 4224 \
		--enable-auto-tool-choice \
		--chat-template $HOME/.vllm-templates/qwen35_patched.jinja \
		--tool-parser-plugin $HOME/git/vllm-qwen35-tool-parser/vllm_qwen35_tool_parser/parser.py \
	  --tool-call-parser qwen35_coder \
		--reasoning-parser qwen3 
#    --speculative_config '{"method":"mtp","num_speculative_tokens":1}'
#		--limit-mm-per-prompt '{"image": 1}' \
#		--mm-processor-kwargs '{"max_pixels": 262144, "min_pixels": 3136}' \
#		########################
#		--chat-template $HOME/bin/qwen3.5/qwen35_merged.jinja \
#		--chat-template $HOME/.vllm-templates/qwen35_patched.jinja \
#		--tool-parser-plugin $HOME/git/vllm-qwen35-tool-parser/vllm_qwen35_tool_parser/parser.py \
#		--tool-call-parser qwen35_coder \
#		--tool-call-parser qwen3_xml \
#		--tool-call-parser qwen35_coder \
# --served-model-name Sehyo/Qwen3.5-122B-A10B-NVFP4 \
