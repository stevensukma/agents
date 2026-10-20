import os

import prometheus_client
import psutil

from .. import utils

PROC_INITIALIZE_TIME = prometheus_client.Histogram(
    "lk_agents_proc_initialize_duration_seconds",
    "Time taken to initialize a process",
    ["nodename"],
    buckets=[0.1, 0.5, 1, 2, 5, 10],
)

# Use 'livesum' mode to aggregate active jobs across all processes
# This sums the values from processes that are still running
RUNNING_JOB_GAUGE = prometheus_client.Gauge(
    "lk_agents_active_job_count",
    "Active jobs",
    ["nodename"],
    multiprocess_mode="livesum",
)

# Use 'max' mode for child process count since we want the total across all processes
CHILD_PROC_GAUGE = prometheus_client.Gauge(
    "lk_agents_child_process_count",
    "Total number of child processes",
    ["nodename"],
    multiprocess_mode="max",
)

CPU_LOAD_GAUGE = prometheus_client.Gauge(
    "lk_agents_worker_load",
    "Worker load percentage",
    ["nodename"],
)

MEMORY_AVAILABLE_GAUGE = prometheus_client.Gauge(
    "lk_agents_worker_memory_available_bytes",
    "Available worker memory bounded by visible cgroup limits; NaN on read failure",
    ["nodename"],
)
MEMORY_REQUIRED_GAUGE = prometheus_client.Gauge(
    "lk_agents_worker_memory_required_bytes",
    "Memory required for pending assignments plus one new job",
    ["nodename"],
)
MEMORY_ADMISSION_BLOCKED_GAUGE = prometheus_client.Gauge(
    "lk_agents_worker_memory_admission_blocked",
    "Whether memory admission is blocked by insufficient headroom or a read failure",
    ["nodename"],
)


# Note: set_function() is not supported in multiprocess mode.# We need to update this metric explicitly.
def _update_child_proc_count() -> None:
    """Update child process count metric. Must be called periodically in the main process."""
    try:
        count = len(psutil.Process(os.getpid()).children(recursive=True))
        CHILD_PROC_GAUGE.labels(nodename=utils.nodename()).set(count)
    except Exception:
        # Process might not exist anymore or access denied
        pass


def _update_worker_load(worker_load: float) -> None:
    CPU_LOAD_GAUGE.labels(nodename=utils.nodename()).set(worker_load)


def _update_worker_memory(*, available_bytes: int | None, required_bytes: float) -> None:
    nodename = utils.nodename()
    MEMORY_AVAILABLE_GAUGE.labels(nodename=nodename).set(
        available_bytes if available_bytes is not None else float("nan")
    )
    MEMORY_REQUIRED_GAUGE.labels(nodename=nodename).set(required_bytes)
    MEMORY_ADMISSION_BLOCKED_GAUGE.labels(nodename=nodename).set(
        int(available_bytes is None or available_bytes < required_bytes)
    )


def job_started() -> None:
    RUNNING_JOB_GAUGE.labels(nodename=utils.nodename()).inc()


def job_ended() -> None:
    RUNNING_JOB_GAUGE.labels(nodename=utils.nodename()).dec()


def proc_initialized(*, time_elapsed: float) -> None:
    PROC_INITIALIZE_TIME.labels(nodename=utils.nodename()).observe(time_elapsed)
