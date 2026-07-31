# Serves the final quantized/merged LoRA variant via vLLM's OpenAI-compatible API.
# Build on the GPU host (Lightning/Kaggle), not locally.
#
#   docker build -t absa-serve serve/
#   docker run --gpus all -p 8000:8000 -v $(pwd)/checkpoints:/models absa-serve
#
# Then point a variants/*.yaml vllm.endpoint at http://localhost:8000/v1

FROM vllm/vllm-openai:latest

ARG MODEL_PATH=/models/absa-lora-merged
ENV MODEL_PATH=${MODEL_PATH}

EXPOSE 8000

ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server", \
            "--model", "/models/absa-lora-merged", \
            "--served-model-name", "absa-lora-merged", \
            "--host", "0.0.0.0", "--port", "8000"]
