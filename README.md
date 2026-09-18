# GPU Orchestration Reusable Package

Files:

- `gpu_orchestrator.py` — reusable orchestration implementation.
- `GPU_ORCHESTRATION.md` — architecture and reuse documentation.

Target environment:
- Kaggle-style Linux notebook
- 2 × NVIDIA T4
- `llama-server`
- localhost HTTP endpoints

The module intentionally contains no application/research logic.

Recommended starting topology:

GPU 0:
- chat :8000
- chat :8001
- embed :8002

GPU 1:
- chat :8003
- chat :8004
- embed :8005

Before using the 2+1 topology, verify the actual model/quantization/ctx-size combination fits the available VRAM.

The orchestration layer is designed around service-parallel inference, not model-parallel inference.
