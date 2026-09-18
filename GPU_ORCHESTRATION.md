# Reusable GPU Orchestration for Kaggle 2×T4

## Purpose

This is the **infrastructure layer only** extracted from the research notebook.

It is intended to be reused whenever a Kaggle runtime provides two NVIDIA T4 GPUs and you want to run several independent `llama-server` processes concurrently.

It deliberately contains no ticker/SAR/knowledge-graph/business logic.

The abstraction is:

```text
                     Kaggle runtime
                  2 × NVIDIA T4 16 GB
                         │
          ┌──────────────┴──────────────┐
          │                             │
        GPU 0                         GPU 1
          │                             │
     ┌────┼────┐                  ┌────┼────┐
     │    │    │                  │    │    │
   chat chat embed              chat chat embed
     │    │    │                  │    │    │
     └────┴────┘                  └────┴────┘
          │                             │
          └──────────┬──────────────────┘
                     │
             Python thread scheduler
                     │
              independent tasks
```

## Why this is not simply "multi-GPU llama.cpp"

There are two different strategies:

### Model-parallel

```text
GPU 0 ──┐
        ├── one model instance
GPU 1 ──┘
```

The model itself is distributed across GPUs.

### Service-parallel

```text
GPU 0 ── chat A
      ├─ chat B
      └─ embed

GPU 1 ── chat C
      ├─ chat D
      └─ embed
```

This project uses **service parallelism**.

Each `llama-server` is an independent process. `CUDA_VISIBLE_DEVICES` controls which physical GPU each process sees.

That makes the setup useful for workloads containing many independent requests/tasks rather than one request that requires the combined VRAM of both GPUs.

---

# 1. Components

## `GPUOrchestrator`

Responsible for:

- launching servers
- assigning a server to a GPU
- assigning ports
- capturing server logs
- checking free VRAM
- waiting for `/health`
- restarting a server
- globally recycling all servers
- final cleanup

It does **not** know anything about the application.

## `ThreadedScheduler`

Responsible for:

- putting independent tasks into a shared queue
- creating one worker thread per chat server
- binding each worker to one chat port/GPU
- tracking completion
- waiting for global checkpoints
- triggering a coordinated recycle
- resuming from durable result files

## `ServerSlot`

A small declaration describing one service:

```python
ServerSlot(
    name="gpu0-a",
    gpu="0",
    port=8000,
    kind="chat",
)
```

---

# 2. Recommended 2×T4 topology

The extracted setup uses:

```text
GPU 0
  8000  chat
  8001  chat
  8002  embedding

GPU 1
  8003  chat
  8004  chat
  8005  embedding
```

So each T4 gets:

```text
2 × LLM server
1 × embedding server
```

This is a **starting topology**, not a universal guarantee.

The important feature is the sequential startup:

```text
launch server
     ↓
wait for /health
     ↓
measure free VRAM
     ↓
launch next server
     ↓
wait for /health
     ↓
measure free VRAM
```

If the topology does not fit, the failure is attributable to the server being launched rather than six processes racing to initialize simultaneously.

---

# 3. GPU isolation

Each process receives:

```python
env["CUDA_VISIBLE_DEVICES"] = slot.gpu
```

For example:

```text
gpu0-a → CUDA_VISIBLE_DEVICES=0
gpu0-b → CUDA_VISIBLE_DEVICES=0
gpu1-a → CUDA_VISIBLE_DEVICES=1
gpu1-b → CUDA_VISIBLE_DEVICES=1
```

Inside each process, its visible GPU becomes device `0`.

This is process-level isolation, not model sharding.

---

# 4. Why multiple servers?

Suppose there are 100 independent tasks.

A single server gives:

```text
task 1
  ↓
task 2
  ↓
task 3
  ↓
...
```

The orchestration layer instead provides:

```text
             task queue
          /   /   |   \
         /   /    |    \
      worker worker worker worker
        │      │      │      │
      GPU0   GPU0    GPU1   GPU1
      chat   chat    chat   chat
```

Each worker talks to its assigned localhost HTTP server.

This is particularly suitable when tasks are independent and each task does not need the other worker's KV cache.

---

# 5. Why threads instead of Python multiprocessing?

The application work is I/O-bound from the scheduler's perspective:

```text
Python worker
     │
     └── HTTP request
             │
             ↓
        llama-server
             │
             ↓
             GPU
```

The GPU inference itself happens inside the independent `llama-server` subprocess.

The Python scheduler therefore does not need to create a separate Python process for every worker.

This keeps the orchestration simpler and avoids Python multiprocessing/spawn coordination.

---

# 6. Global checkpointing

This is one of the most important parts of the design.

Do **not** recycle a server while another worker may still be using it.

Instead:

```text
worker A finishes task
        ↓
      WAIT

worker B finishes task
        ↓
      WAIT

worker C finishes task
        ↓
      WAIT

worker D finishes task
        ↓
      LAST
        │
        ▼
record checkpoint
        │
        ▼
recycle ALL servers
        │
        ▼
all servers healthy
        │
        ▼
release workers
        │
        ▼
continue
```

The checkpoint is therefore a **global quiescence point**.

This intentionally sacrifices some throughput at the checkpoint boundary in exchange for a clean restart point.

---

# 7. Why recycle all servers together?

The orchestration is designed around a practical observation:

Long-running local inference can develop resource pressure that is difficult to attribute to a single request.

Instead of attempting increasingly complicated per-request cleanup:

```text
request → restart
```

or per-worker cleanup:

```text
worker → restart
```

the reusable layer provides:

```text
N completed tasks
      ↓
all workers quiescent
      ↓
restart every server
      ↓
resume
```

This gives the process group a fresh lifecycle without killing a request in flight.

The checkpoint interval is configurable.

Default:

```python
checkpoint_every = 10
```

---

# 8. Resumability

A task is considered completed based on a durable file:

```text
/kaggle/working/gpu_orchestrator_results/
    task-001.json
    task-002.json
    task-003.json
```

On restart:

```text
load task list
     ↓
inspect result directory
     ↓
skip durable tasks
     ↓
queue remaining tasks
     ↓
continue
```

This is deliberately independent of Python memory.

A notebook/session restart therefore does not require the scheduler's in-memory state to survive.

**Important:** the task handler must write its result durably before returning if you want this guarantee.

---

# 9. Application interface

The orchestration layer does not dictate what your task does.

Provide:

```python
def task_handler(task, chat_port, gpu, worker_name):
    ...
```

For example, an application can do:

```text
task
  ↓
chat server on chat_port
  ↓
optional embedding server on same GPU
  ↓
write result JSON
  ↓
return
```

The orchestration layer remains unchanged if the application later becomes:

- translation
- document extraction
- classification
- summarization
- batch generation
- graph extraction
- entity resolution
- evaluation
- another research experiment

---

# 10. Embedding locality

If embeddings are used, keep the embedding server local to the worker's GPU:

```text
GPU 0 worker
    ├── chat → GPU 0 chat server
    └── embed → GPU 0 embedding server

GPU 1 worker
    ├── chat → GPU 1 chat server
    └── embed → GPU 1 embedding server
```

This avoids funneling all embedding traffic through one GPU.

The reusable orchestrator intentionally does not create a single global embedding client. The application can construct one client per GPU/port.

---

# 11. Model arguments

The module includes helpers for the model configuration used in the extracted notebook:

### Gemma 4 + MTP

```text
--model MODEL_PATH
--model-draft MTP_PATH
--spec-type draft-mtp
--spec-draft-n-max 2
--n-gpu-layers -1
--flash-attn on
--ctx-size 16384
--jinja
```

### Embedding

```text
--model EMBED_MODEL_PATH
--embedding
--pooling cls
--ubatch-size 8192
--ctx-size 8192
--n-gpu-layers -1
```

These are exposed as configurable builders so a future model can use different arguments without changing the orchestration code.

---

# 12. Minimal usage

```python
from gpu_orchestrator import (
    GPUOrchestrator,
    OrchestratorConfig,
    ThreadedScheduler,
    make_kaggle_2xt4_slots,
    make_gemma4_chat_args,
    make_embedding_args,
)

MODEL_PATH = "/kaggle/working/model.gguf"
MTP_PATH = "/kaggle/working/model-mtp.gguf"
EMBED_MODEL_PATH = "/kaggle/working/bge-m3.gguf"

slots = make_kaggle_2xt4_slots()

config = OrchestratorConfig(
    llama_server="/kaggle/working/llama_bin/llama-server",
    checkpoint_every=10,
)

orchestrator = GPUOrchestrator(
    config=config,
    slots=slots,
    chat_args=make_gemma4_chat_args(MODEL_PATH, MTP_PATH),
    embed_args=make_embedding_args(EMBED_MODEL_PATH),
)

orchestrator.start_all()
```

Then define your application task:

```python
def task_handler(task, chat_port, gpu, worker_name):
    # Your application logic goes here.
    # Use chat_port for this worker's llama-server.
    # Use gpu to select the corresponding embedding server.
    #
    # IMPORTANT:
    # persist the result before returning.
    return {"task": task, "chat_port": chat_port, "gpu": gpu}
```

Run:

```python
tasks = ["task-001", "task-002", "task-003", ...]

scheduler = ThreadedScheduler(
    orchestrator=orchestrator,
    tasks=tasks,
    task_id=lambda x: x,
    task_handler=task_handler,
)

results = scheduler.run()
```

Finally:

```python
orchestrator.stop_all()
```

---

# 13. Recommended notebook structure

For future projects, keep the orchestration cells separate from application cells:

```text
00  Install / environment
01  GPU inventory
02  Model paths
03  Server topology
04  Server launch
05  Server health / VRAM verification
06  Client definitions
07  Application logic
08  Task definition
09  Threaded scheduler
10  Run
11  Results
12  Cleanup
```

The reusable infrastructure should ideally end at:

```text
06  Client definitions
```

Everything after that should belong to the specific project.

---

# 14. What should NOT go into this reusable layer

Do not put these into the orchestration module:

- ticker logic
- SEC/EDGAR retrieval
- SAR logic
- knowledge graph schemas
- prompts specific to a research question
- prediction arms
- evaluation metrics
- experiment metadata
- Hugging Face experiment uploads
- domain-specific result formats

Those belong to the application.

The orchestration module should answer only:

> **"How do I turn these GPUs into a pool of reusable local inference services and safely dispatch independent work to them?"**

---

# 15. Future extensions

The architecture can later grow without changing the core model:

```text
Current

2 × T4
 ├── 4 chat
 └── 2 embed


Possible future

2 × T4
 ├── chat pool
 ├── embedding pool
 ├── vision pool
 └── specialized model pool
```

The important abstraction is the `ServerSlot`:

```python
ServerSlot(
    name="gpu1-vision",
    gpu="1",
    port=8006,
    kind="vision",
)
```

The scheduler can then be extended to route tasks according to capability rather than hard-coding research logic.

---

# 16. Design principle

The entire reusable setup can be reduced to five ideas:

```text
1. ISOLATE
   One llama-server process → one GPU assignment.

2. PACK
   Multiple independent services can share a T4 when VRAM allows.

3. DISPATCH
   One Python worker → one persistent chat endpoint.

4. QUIESCE
   Finish in-flight tasks before global maintenance.

5. RECOVER
   Persist completed tasks and resume from disk.
```

That is the infrastructure worth carrying from project to project.
