# CPU-only PyTorch + transformers 4.57.3 — just enough to run the BreezeTTS 2
# reference and dump float32 parity fixtures. CPU on purpose: float32 on CPU is
# deterministic and is the golden the C++ port is checked against.
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.6.0
RUN pip install --no-cache-dir "transformers==4.57.3" safetensors numpy soundfile scipy accelerate
WORKDIR /work
