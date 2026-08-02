# Serves the final quantized/merged LoRA variant via vLLM's OpenAI-compatible API.
# Build on a GPU host with Docker available (NOT Kaggle -- Kaggle notebooks
# have no Docker daemon; this project's actual Session 6 numbers were taken
# by running vLLM directly as a background process in the notebook instead,
# with the same flags below. This file is the "how you'd really deploy this"
# reference, not something run during this project's own benchmarking).
#
#   docker build -t absa-serve serve/
#   docker run --gpus all -p 8000:8000 -v $(pwd)/checkpoints:/models absa-serve
#
# Then point a variants/*.yaml vllm.endpoint at http://localhost:8000/v1.
# Eval and load-test code use the plain /completions endpoint, not
# /chat/completions -- the latter auto-applies a chat template that training
# never used (see eval/runner.py's run_vllm_backend).

FROM vllm/vllm-openai:latest

ARG MODEL_PATH=/models/absa-lora-merged
ENV MODEL_PATH=${MODEL_PATH}

EXPOSE 8000

ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server", \
            "--model", "/models/absa-lora-merged", \
            "--served-model-name", "absa-lora-merged", \
            "--quantization", "bitsandbytes", \
            "--load-format", "bitsandbytes", \
            "--host", "0.0.0.0", "--port", "8000"]
