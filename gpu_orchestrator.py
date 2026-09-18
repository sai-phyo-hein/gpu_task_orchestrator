"""
Reusable multi-GPU llama.cpp orchestration for Kaggle-style 2xT4 runtimes.

The module deliberately contains NO research/application logic.
It provides:
  - GPU-pinned llama-server subprocesses
  - multiple chat servers per GPU
  - optional embedding server per GPU
  - sequential startup with VRAM checkpoints
  - /health readiness checks
  - thread-based task dispatch
  - coordinated global checkpoints
  - full server recycle at checkpoints
  - disk-based resumability
  - optional checkpoint hooks

Typical topology on 2xT4:
    GPU 0: chat, chat, embed
    GPU 1: chat, chat, embed

The model-specific command-line arguments are configurable through ServerSlot.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional, Any

import requests


@dataclass(frozen=True)
class ServerSlot:
    name: str
    gpu: str
    port: int
    kind: str = "chat"  # "chat" or "embed"


@dataclass
class ServerInfo:
    slot: ServerSlot
    proc: subprocess.Popen
    log_path: Path


@dataclass
class OrchestratorConfig:
    llama_server: str
    log_dir: str = "/kaggle/working/gpu_orchestrator_logs"
    host: str = "127.0.0.1"
    health_timeout: int = 180
    health_poll_interval: float = 3.0
    terminate_timeout: int = 15
    checkpoint_every: int = 10
    result_dir: str = "/kaggle/working/gpu_orchestrator_results"


class GPUOrchestrator:
    """
    Owns the lifecycle of all llama-server processes.

    The orchestrator does not know what a task means. The caller supplies a
    task_handler(task, chat_port, gpu, worker_name), making the infrastructure
    reusable for translation, extraction, classification, batch generation,
    embedding-assisted pipelines, etc.
    """

    def __init__(
        self,
        config: OrchestratorConfig,
        slots: list[ServerSlot],
        chat_args: Callable[[ServerSlot], list[str]],
        embed_args: Optional[Callable[[ServerSlot], list[str]]] = None,
    ):
        self.config = config
        self.slots = slots
        self.chat_args = chat_args
        self.embed_args = embed_args or (lambda slot: [])

        self.log_dir = Path(config.log_dir)
        self.result_dir = Path(config.result_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.result_dir.mkdir(parents=True, exist_ok=True)

        self.server_procs: dict[str, ServerInfo] = {}
        self._lifecycle_lock = threading.RLock()

    # ---------- GPU / process lifecycle ----------

    @staticmethod
    def free_mem_mb(gpu_index: int | str) -> int:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={gpu_index}",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ]
        )
        return int(out.decode().strip())

    def _args_for(self, slot: ServerSlot) -> list[str]:
        if slot.kind == "chat":
            return self.chat_args(slot)
        if slot.kind == "embed":
            return self.embed_args(slot)
        raise ValueError(f"unknown server kind: {slot.kind!r}")

    def launch_server(self, slot: ServerSlot) -> ServerInfo:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(slot.gpu)

        log_path = self.log_dir / f"server_{slot.name}.log"
        log_file = open(log_path, "w", buffering=1)

        proc = subprocess.Popen(
            [
                self.config.llama_server,
                *self._args_for(slot),
                "--port",
                str(slot.port),
                "--host",
                self.config.host,
            ],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
        )

        return ServerInfo(slot=slot, proc=proc, log_path=log_path)

    def wait_ready(self, info: ServerInfo) -> bool:
        start = time.time()
        url = f"http://{self.config.host}:{info.slot.port}/health"

        while time.time() - start < self.config.health_timeout:
            if info.proc.poll() is not None:
                tail = info.log_path.read_text(errors="replace")[-3000:]
                print(
                    f"[{info.slot.name}] exited early "
                    f"(code={info.proc.returncode})\n{tail}"
                )
                return False

            try:
                response = requests.get(url, timeout=3)
                if response.status_code == 200:
                    return True
            except requests.RequestException:
                pass

            time.sleep(self.config.health_poll_interval)

        print(f"[{info.slot.name}] health check timed out")
        return False

    def start_all(self) -> None:
        """
        Start slots one at a time.

        Sequential startup is intentional: it makes VRAM pressure attributable
        to the server being launched and avoids six processes competing for
        memory initialization simultaneously.
        """
        with self._lifecycle_lock:
            for slot in self.slots:
                gpu = int(slot.gpu)
                before = self.free_mem_mb(gpu)
                print(
                    f"Launching {slot.name}: GPU {slot.gpu}, "
                    f"port {slot.port}, kind={slot.kind}; "
                    f"free VRAM before={before} MiB"
                )

                info = self.launch_server(slot)

                if not self.wait_ready(info):
                    self._stop_info(info)
                    raise RuntimeError(
                        f"{slot.name} failed to start; inspect {info.log_path}"
                    )

                after = self.free_mem_mb(gpu)
                self.server_procs[slot.name] = info
                print(
                    f"  {slot.name} ready; free VRAM after={after} MiB"
                )

        print(self.status())

    def _stop_info(self, info: ServerInfo) -> None:
        if info.proc.poll() is None:
            info.proc.terminate()
            try:
                info.proc.wait(timeout=self.config.terminate_timeout)
            except subprocess.TimeoutExpired:
                info.proc.kill()
                info.proc.wait()

    def recycle(self, slot: ServerSlot) -> None:
        """
        Restart one exact slot on the same GPU and port.
        """
        with self._lifecycle_lock:
            old = self.server_procs.get(slot.name)
            if old is not None:
                self._stop_info(old)

            info = self.launch_server(slot)
            if not self.wait_ready(info):
                self._stop_info(info)
                raise RuntimeError(
                    f"{slot.name} failed after recycle; inspect {info.log_path}"
                )

            self.server_procs[slot.name] = info

    def recycle_all(self) -> None:
        """
        Globally quiesce and rebuild every server.

        Call this only when no task is currently using a server.
        """
        print(f"[recycle] restarting {len(self.slots)} servers")
        for slot in self.slots:
            self.recycle(slot)
        print("[recycle] all servers ready")

    def stop_all(self) -> None:
        with self._lifecycle_lock:
            for info in list(self.server_procs.values()):
                self._stop_info(info)
            self.server_procs.clear()
        print("[cleanup] all servers stopped")

    def status(self) -> str:
        rows = []
        for name, info in self.server_procs.items():
            state = "running" if info.proc.poll() is None else "stopped"
            rows.append(
                f"  {name}: GPU {info.slot.gpu}, "
                f"port {info.slot.port}, {info.slot.kind}, {state}"
            )
        return "\n".join(["[servers]", *rows])


# ---------- Generic resumable threaded scheduler ----------

TaskHandler = Callable[[Any, int, str, str], Any]
CheckpointHook = Callable[[int, list[Any], list[Any]], None]


class ThreadedScheduler:
    """
    One worker thread per chat server.

    Each worker owns a fixed chat port/GPU association. Tasks are distributed
    from a shared queue. A completed task is considered durable only after the
    handler has written its own result file (or otherwise made it persistent).

    Checkpoints are global, not per worker:
      1. workers finish their current task
      2. workers wait at a barrier
      3. one worker performs the checkpoint hook
      4. all servers are recycled
      5. workers resume

    This deliberately sacrifices some throughput at checkpoint boundaries for
    a clean, fully-quiesced restart point.
    """

    def __init__(
        self,
        orchestrator: GPUOrchestrator,
        tasks: Iterable[Any],
        task_id: Callable[[Any], str],
        task_handler: TaskHandler,
        checkpoint_every: Optional[int] = None,
        checkpoint_hook: Optional[CheckpointHook] = None,
    ):
        self.orchestrator = orchestrator
        self.tasks = list(tasks)
        self.task_id = task_id
        self.task_handler = task_handler
        self.checkpoint_every = (
            checkpoint_every
            if checkpoint_every is not None
            else orchestrator.config.checkpoint_every
        )
        self.checkpoint_hook = checkpoint_hook

        self.work_queue: queue.Queue = queue.Queue()
        self.outcomes: dict[str, Any] = {}
        self.outcomes_lock = threading.Lock()
        self.print_lock = threading.Lock()

        self.done_count = 0
        self.done_lock = threading.Lock()

        self.checkpoint_condition = threading.Condition()
        self.waiting_workers: set[str] = set()
        self.active_workers: set[str] = set()
        self.next_checkpoint_target = self.checkpoint_every
        self.checkpoint_seq = 0

    def _result_path(self, task: Any) -> Path:
        return self.orchestrator.result_dir / f"{self.task_id(task)}.json"

    def _already_done(self, task: Any) -> bool:
        return self._result_path(task).exists()

    def _durable_completed(self) -> list[Any]:
        return [
            task for task in self.tasks
            if self._already_done(task)
        ]

    def _remaining(self) -> list[Any]:
        completed_ids = {self.task_id(t) for t in self._durable_completed()}
        return [
            task for task in self.tasks
            if self.task_id(task) not in completed_ids
        ]

    def _checkpoint_if_due(self, worker_name: str) -> None:
        with self.checkpoint_condition:
            if self.done_count < self.next_checkpoint_target:
                return

            # No need to wait for workers that already drained the queue.
            self.waiting_workers.add(worker_name)

            while worker_name in self.waiting_workers:
                if self.waiting_workers >= self.active_workers:
                    self.checkpoint_seq += 1
                    seq = self.checkpoint_seq

                    # Perform slow lifecycle operations outside the condition.
                    self.checkpoint_condition.release()
                    try:
                        completed = self._durable_completed()
                        remaining = self._remaining()

                        print(
                            f"[checkpoint {seq}] "
                            f"{len(completed)}/{len(self.tasks)} durable"
                        )

                        if self.checkpoint_hook:
                            self.checkpoint_hook(seq, completed, remaining)

                        self.orchestrator.recycle_all()

                    finally:
                        self.checkpoint_condition.acquire()

                    self.next_checkpoint_target += self.checkpoint_every
                    self.waiting_workers.clear()
                    self.checkpoint_condition.notify_all()
                else:
                    self.checkpoint_condition.wait()

    def _worker_loop(
        self,
        port: int,
        gpu: str,
        worker_name: str,
    ) -> None:
        with self.checkpoint_condition:
            self.active_workers.add(worker_name)

        while True:
            try:
                task = self.work_queue.get_nowait()
            except queue.Empty:
                with self.checkpoint_condition:
                    self.active_workers.discard(worker_name)
                    self.checkpoint_condition.notify_all()
                return

            task_number = self.done_count + 1
            with self.print_lock:
                print(
                    f"[{worker_name}] starting {self.task_id(task)} "
                    f"(queue={self.work_queue.qsize()})"
                )

            try:
                result = self.task_handler(task, port, gpu, worker_name)
                with self.outcomes_lock:
                    self.outcomes[self.task_id(task)] = result
            except Exception as exc:
                with self.outcomes_lock:
                    self.outcomes[self.task_id(task)] = {
                        "error": repr(exc),
                    }
                with self.print_lock:
                    print(
                        f"[{worker_name}] task {self.task_id(task)} failed: {exc}"
                    )
            finally:
                self.work_queue.task_done()

            with self.done_lock:
                self.done_count += 1
                done = self.done_count

            with self.print_lock:
                print(
                    f"[{worker_name}] finished {self.task_id(task)} "
                    f"({done}/{len(self.tasks)})"
                )

            self._checkpoint_if_due(worker_name)

    def run(self) -> dict[str, Any]:
        remaining = self._remaining()

        print(
            f"[scheduler] {len(self.tasks)} total tasks; "
            f"{len(self.tasks) - len(remaining)} already durable; "
            f"{len(remaining)} remaining"
        )

        if not remaining:
            return self.outcomes

        chat_slots = [
            info.slot
            for info in self.orchestrator.server_procs.values()
            if info.slot.kind == "chat"
        ]
        if not chat_slots:
            raise RuntimeError("No chat servers are running")

        for task in remaining:
            self.work_queue.put(task)

        self.active_workers = {
            f"worker-{slot.port}" for slot in chat_slots
        }

        threads = [
            threading.Thread(
                target=self._worker_loop,
                args=(slot.port, slot.gpu, f"worker-{slot.port}"),
                daemon=True,
            )
            for slot in chat_slots
        ]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        return self.outcomes


# ---------- Recommended 2xT4 topology ----------

def make_kaggle_2xt4_slots(
    gpu0_chat_ports=(8000, 8001),
    gpu0_embed_port=8002,
    gpu1_chat_ports=(8003, 8004),
    gpu1_embed_port=8005,
) -> list[ServerSlot]:
    """
    Recommended starting topology:

        GPU 0: 2 chat + 1 embed
        GPU 1: 2 chat + 1 embed

    Adjust after measuring VRAM on the actual runtime.
    """
    return [
        ServerSlot("gpu0-a", "0", gpu0_chat_ports[0], "chat"),
        ServerSlot("gpu0-b", "0", gpu0_chat_ports[1], "chat"),
        ServerSlot("gpu0-embed", "0", gpu0_embed_port, "embed"),
        ServerSlot("gpu1-a", "1", gpu1_chat_ports[0], "chat"),
        ServerSlot("gpu1-b", "1", gpu1_chat_ports[1], "chat"),
        ServerSlot("gpu1-embed", "1", gpu1_embed_port, "embed"),
    ]


def make_gemma4_chat_args(
    model_path: str,
    mtp_path: str,
    ctx_size: int = 16384,
) -> Callable[[ServerSlot], list[str]]:
    def builder(slot: ServerSlot) -> list[str]:
        return [
            "--model", model_path,
            "--model-draft", mtp_path,
            "--spec-type", "draft-mtp",
            "--spec-draft-n-max", "2",
            "--n-gpu-layers", "-1",
            "--flash-attn", "on",
            "--ctx-size", str(ctx_size),
            "--jinja",
        ]
    return builder


def make_embedding_args(
    model_path: str,
    ctx_size: int = 8192,
    ubatch_size: int = 8192,
) -> Callable[[ServerSlot], list[str]]:
    def builder(slot: ServerSlot) -> list[str]:
        return [
            "--model", model_path,
            "--embedding",
            "--pooling", "cls",
            "--ubatch-size", str(ubatch_size),
            "--ctx-size", str(ctx_size),
            "--n-gpu-layers", "-1",
        ]
    return builder
