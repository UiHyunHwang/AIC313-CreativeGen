"""Watch available GPU memory in a separate CPU process; stop only our parent.

This is a best-effort safety margin, not GPU memory isolation or a reservation
for other jobs. The PyTorch allocator limit is configured separately.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


class MemoryGuardStop(RuntimeError):
    """The generation process was stopped by its own memory guard."""


def free_gpu_mib() -> float:
    result = subprocess.run(
        ["nvidia-smi", "--id=0", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True, timeout=5,
    )
    return float(result.stdout.strip())


def _identity(pid: int) -> str | None:
    try:
        # A PID can be reused. Include the process start time before signalling.
        stat = Path(f"/proc/{pid}/stat").read_text()
        return stat[stat.rfind(")") + 2:].split()[19]
    except (OSError, IndexError):
        return None


def _save(path: Path, record: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _watch(pid: int, identity: str, threshold_gib: float, path: Path) -> None:
    record = {"status": "monitoring", "pid": pid, "min_free_gpu_gib": threshold_gib, "samples": 0}
    minimum = None
    while _identity(pid) == identity:
        try:
            free = free_gpu_mib() / 1024
            minimum = free if minimum is None else min(minimum, free)
            record.update(last_free_gib=free, minimum_observed_free_gib=minimum)
            reason = f"GPU free memory {free:.3f} GiB < {threshold_gib:.3f} GiB" if free < threshold_gib else None
        except Exception as error:
            # Stop our process if monitoring itself fails; do not keep running
            # without the requested protection.
            reason = f"GPU monitoring failed: {error}"
        record["samples"] += 1
        if reason is not None:
            record.update(status="stopped_generation", reason=reason)
            _save(path, record)
            if _identity(pid) == identity:
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    return
                deadline = time.monotonic() + 3
                while _identity(pid) == identity and time.monotonic() < deadline:
                    time.sleep(0.1)
                if _identity(pid) == identity:
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            return
        _save(path, record)
        time.sleep(0.5)
    record["status"] = "parent_exited"
    _save(path, record)


class GpuMemoryGuard:
    def __init__(self, threshold_gib: float, out_dir: Path):
        self.threshold_gib = threshold_gib
        self.path = out_dir / "memory_guard.json"
        self.process = None
        self.previous_handler = None

    def __enter__(self):
        if self.threshold_gib <= 0:
            return self
        free = free_gpu_mib() / 1024
        if free < self.threshold_gib:
            raise MemoryGuardStop(f"Not starting: only {free:.3f} GiB GPU memory is free")
        pid = os.getpid()
        identity = _identity(pid)
        if identity is None:
            raise MemoryGuardStop("Cannot identify this generation process for safe monitoring")

        def stop_generation(signum, frame):
            raise MemoryGuardStop("Generation stopped by SIGTERM; see memory_guard.json")

        self.previous_handler = signal.signal(signal.SIGTERM, stop_generation)
        self.process = subprocess.Popen(
            [sys.executable, "-m", "sdxl.memory_guard", "--pid", str(pid),
             "--identity", identity, "--threshold-gib", str(self.threshold_gib), "--log", str(self.path)],
            start_new_session=True,
        )
        return self

    def close(self):
        if self.process is None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        if self.previous_handler is not None:
            signal.signal(signal.SIGTERM, self.previous_handler)
        if self.path.exists():
            record = json.loads(self.path.read_text())
            if record.get("status") == "monitoring":
                record["status"] = "monitoring_finished"
                _save(self.path, record)

    def __exit__(self, *args):
        self.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--identity", required=True)
    parser.add_argument("--threshold-gib", type=float, required=True)
    parser.add_argument("--log", type=Path, required=True)
    arguments = parser.parse_args()
    _watch(arguments.pid, arguments.identity, arguments.threshold_gib, arguments.log)
