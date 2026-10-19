import asyncio
import multiprocessing as mp
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from livekit.agents import JobExecutorType
from livekit.agents.ipc import proc_pool

pytestmark = pytest.mark.unit


@pytest.fixture
async def pool(monkeypatch):
    def executor(**kwargs):
        closed = asyncio.Event()
        proc = MagicMock()
        proc.running_job = None
        proc.start = AsyncMock()
        proc.initialize = AsyncMock()
        proc.join = AsyncMock(side_effect=closed.wait)
        proc.aclose = AsyncMock(side_effect=closed.set)

        async def launch(info):
            proc.running_job = info

        proc.launch_job = AsyncMock(side_effect=launch)
        return proc

    monkeypatch.setattr(proc_pool.job_thread_executor, "ThreadJobExecutor", executor)
    pool = proc_pool.ProcPool(
        initialize_process_fnc=lambda proc: None,
        job_entrypoint_fnc=AsyncMock(),
        session_end_fnc=None,
        simulation_end_fnc=None,
        num_idle_processes=4,
        initialize_timeout=1,
        close_timeout=1,
        session_end_timeout=1,
        inference_executor=None,
        job_executor_type=JobExecutorType.THREAD,
        mp_ctx=mp.get_context("spawn"),
        memory_warn_mb=0,
        memory_limit_mb=0,
        http_proxy=None,
        loop=asyncio.get_running_loop(),
    )
    yield pool
    await pool.aclose()


@pytest.mark.parametrize("target", [0, 1, 2])
async def test_start_waits_for_current_target_instead_of_default(pool, target, caplog):
    pool.set_target_idle_processes(target)
    await asyncio.wait_for(pool.start(), timeout=0.5)
    await asyncio.sleep(0)
    assert len(pool.processes) == target
    assert pool._warmed_proc_queue.qsize() == target
    assert "timed out waiting for idle processes" not in caplog.text


async def test_target_recovers_and_existing_warm_processes_are_retained(pool):
    pool.set_target_idle_processes(0)
    await pool.start()
    pool.set_target_idle_processes(2)
    await asyncio.wait_for(pool._idle_ready.wait(), timeout=1)
    assert len(pool.processes) == 2
    pool.set_target_idle_processes(0)
    for process in pool.processes:
        process.aclose.assert_not_called()
    assert len(pool.processes) == 2


async def test_assigned_job_can_launch_with_zero_idle_target(pool):
    pool.set_target_idle_processes(0)
    await pool.start()
    info = SimpleNamespace(job=SimpleNamespace(id="assigned-job"))
    await asyncio.wait_for(pool.launch_job(info), timeout=1)
    assert len(pool.processes) == 1
    assert pool.processes[0].running_job is info
