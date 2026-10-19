import asyncio
import math
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from livekit.agents import AgentServer, WorkerOptions, worker as worker_module
from livekit.protocol import agent

pytestmark = pytest.mark.unit
MIB = 1024 * 1024


async def entrypoint(ctx):
    pass


@pytest.fixture
async def server():
    server = AgentServer(load_fnc=lambda: 0.0, load_threshold=math.inf)
    server.rtc_session(entrypoint)
    server._request_fnc = worker_module._default_request_fnc
    server._loop = asyncio.get_running_loop()
    server._devmode = False
    server._worker_load = 0.0
    server._reserved_slots = 0
    server._proc_pool = MagicMock()
    server._proc_pool.processes = []
    server._pending_assignments = {}
    server._job_lifecycle_tasks = set()
    server._previous_status = None

    async def launch_job(info):
        server._proc_pool.processes.append(SimpleNamespace(running_job=info))

    server._proc_pool.launch_job = AsyncMock(side_effect=launch_job)

    async def send(msg):
        if msg.HasField("availability") and msg.availability.available:
            server._handle_assignment(
                agent.JobAssignment(job=agent.Job(id=msg.availability.job_id), token="test-token")
            )

    server._queue_msg = AsyncMock(side_effect=send)
    yield server
    tasks = list(server._job_lifecycle_tasks)
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


def request(job_id="job-1"):
    return agent.AvailabilityRequest(job=agent.Job(id=job_id))


async def finish_requests(server):
    await asyncio.gather(*server._job_lifecycle_tasks)
    await asyncio.sleep(0)  # run the existing reservation-release callbacks


def response(server, job_id):
    return next(
        call.args[0].availability
        for call in server._queue_msg.call_args_list
        if call.args[0].HasField("availability") and call.args[0].availability.job_id == job_id
    )


@pytest.mark.parametrize("available,accepted", [(767 * MIB, False), (768 * MIB, True)])
async def test_memory_boundary_and_retryable_rejection(server, monkeypatch, available, accepted):
    server.update_options(worker_memory_reserve_mb=768)
    monkeypatch.setattr(worker_module, "get_available_memory", lambda: available)
    await server._answer_availability(request())
    await finish_requests(server)
    result = response(server, "job-1")
    assert result.available is accepted
    assert not result.terminate
    assert len(server.active_jobs) == int(accepted)


async def test_memory_status_recovers_without_killing_running_job(server, monkeypatch):
    server.update_options(worker_memory_reserve_mb=768)
    available = 1000 * MIB
    monkeypatch.setattr(worker_module, "get_available_memory", lambda: available)
    await server._answer_availability(request())
    await finish_requests(server)
    available = 400 * MIB
    await server._update_worker_status()
    assert server._queue_msg.call_args.args[0].update_worker.status == agent.WorkerStatus.WS_FULL
    assert len(server.active_jobs) == 1
    available = 1000 * MIB
    await server._update_worker_status()
    assert (
        server._queue_msg.call_args.args[0].update_worker.status == agent.WorkerStatus.WS_AVAILABLE
    )
    assert len(server.active_jobs) == 1


async def test_count_limit_and_job_completion(server):
    server.update_options(max_active_jobs=1)
    await server._answer_availability(request())
    await finish_requests(server)
    await server._answer_availability(request("job-2"))
    assert not response(server, "job-2").available
    assert not response(server, "job-2").terminate
    await server._update_worker_status()
    assert server._queue_msg.call_args.args[0].update_worker.status == agent.WorkerStatus.WS_FULL
    server._proc_pool.processes.clear()  # the first job exits
    await server._update_worker_status()
    assert (
        server._queue_msg.call_args.args[0].update_worker.status == agent.WorkerStatus.WS_AVAILABLE
    )
    await server._answer_availability(request("job-3"))
    await finish_requests(server)
    assert response(server, "job-3").available


@pytest.mark.parametrize("gate", ["memory", "count"])
async def test_concurrent_requests_reserve_capacity_before_callback(server, monkeypatch, gate):
    if gate == "memory":
        server.update_options(worker_memory_reserve_mb=768)
        monkeypatch.setattr(worker_module, "get_available_memory", lambda: 1000 * MIB)
    else:
        server.update_options(max_active_jobs=1)
    started = asyncio.Event()
    release = asyncio.Event()

    async def delayed_request(job):
        started.set()
        await release.wait()
        await job.reject(terminate=False)

    server._request_fnc = delayed_request
    await asyncio.gather(
        server._answer_availability(request()), server._answer_availability(request("job-2"))
    )
    await started.wait()
    assert server._reserved_slots == 1
    rejected = [
        call.args[0].availability
        for call in server._queue_msg.call_args_list
        if call.args[0].HasField("availability")
    ]
    assert len(rejected) == 1
    assert not rejected[0].available and not rejected[0].terminate
    await server._update_worker_status()
    assert server._queue_msg.call_args.args[0].update_worker.status == agent.WorkerStatus.WS_FULL
    release.set()
    await finish_requests(server)
    assert server._reserved_slots == 0
    assert server._is_available()


async def test_assignment_timeout_releases_capacity(server, monkeypatch):
    server.update_options(max_active_jobs=1)
    server._queue_msg.side_effect = None  # never deliver assignment
    monkeypatch.setattr(worker_module, "ASSIGNMENT_TIMEOUT", 0.001)
    await server._answer_availability(request())
    await finish_requests(server)
    assert server._reserved_slots == 0
    assert server._pending_assignments == {}
    assert server._is_available()


async def test_callback_failure_releases_capacity(server):
    server.update_options(max_active_jobs=1)
    server._request_fnc = AsyncMock(side_effect=RuntimeError("callback failed"))
    await server._answer_availability(request())
    await finish_requests(server)
    assert server._reserved_slots == 0
    assert not response(server, "job-1").terminate
    assert server._is_available()


async def test_memory_read_failure_rejects_and_recovers(server, monkeypatch, caplog):
    server.update_options(worker_memory_reserve_mb=768)
    reader = MagicMock(side_effect=[OSError("unreadable"), ValueError("invalid"), 1000 * MIB])
    monkeypatch.setattr(worker_module, "get_available_memory", reader)
    await server._answer_availability(request())
    await server._update_worker_status()
    assert not response(server, "job-1").available
    assert not response(server, "job-1").terminate
    assert server._queue_msg.call_args.args[0].update_worker.status == agent.WorkerStatus.WS_FULL
    assert caplog.text.count("cannot read worker memory capacity") == 1
    await server._answer_availability(request("job-2"))
    await finish_requests(server)
    assert response(server, "job-2").available
    assert not server._memory_read_failed


@pytest.mark.parametrize("load,accepted", [(0.2, True), (0.8, False)])
async def test_disabled_defaults_preserve_custom_load(server, monkeypatch, load, accepted):
    server.update_options(load_threshold=0.7)
    server.load_fnc = lambda: load
    reader = MagicMock(side_effect=AssertionError("disabled memory admission must not read memory"))
    monkeypatch.setattr(worker_module, "get_available_memory", reader)
    await server._answer_availability(request())
    await finish_requests(server)
    assert response(server, "job-1").available is accepted
    reader.assert_not_called()


async def test_memory_applies_in_dev_with_infinite_load_threshold(server, monkeypatch):
    server._devmode = True
    server.update_options(worker_memory_reserve_mb=768)
    monkeypatch.setattr(worker_module, "get_available_memory", lambda: 0)
    await server._answer_availability(request())
    assert not response(server, "job-1").available


async def test_cpu_still_gates_when_memory_is_healthy(server, monkeypatch):
    server.update_options(worker_memory_reserve_mb=768, load_threshold=0.7)
    server.load_fnc = lambda: 0.8
    monkeypatch.setattr(worker_module, "get_available_memory", lambda: 1000 * MIB)
    await server._answer_availability(request())
    assert not response(server, "job-1").available


async def test_simulation_bypasses_resources_but_not_draining(server, monkeypatch):
    server.update_options(worker_memory_reserve_mb=768, max_active_jobs=1)
    server._simulation = True
    server._proc_pool.processes = [SimpleNamespace(running_job=object())]
    reader = MagicMock(side_effect=AssertionError("simulation must not read memory"))
    monkeypatch.setattr(worker_module, "get_available_memory", reader)
    await server._update_worker_status()
    assert server._is_available()
    assert (
        server._queue_msg.call_args.args[0].update_worker.status == agent.WorkerStatus.WS_AVAILABLE
    )
    server._draining = True
    assert not server._is_available()
    reader.assert_not_called()


@pytest.mark.parametrize(
    "options",
    [
        {"worker_memory_reserve_mb": -1},
        {"worker_memory_reserve_mb": math.inf},
        {"worker_memory_reserve_mb": math.nan},
        {"max_active_jobs": -1},
        {"max_active_jobs": 1.5},
    ],
)
def test_invalid_options_rejected_at_all_entry_points(options):
    with pytest.raises(ValueError):
        AgentServer(**options)
    with pytest.raises(ValueError):
        WorkerOptions(entrypoint_fnc=entrypoint, **options)
    server = AgentServer()
    with pytest.raises(ValueError):
        server.update_options(**options)


def test_legacy_options_propagate_and_are_pickleable():
    import pickle

    options = WorkerOptions(
        entrypoint_fnc=entrypoint, worker_memory_reserve_mb=768, max_active_jobs=2
    )
    server = AgentServer.from_server_options(pickle.loads(pickle.dumps(options)))
    assert server._worker_memory_reserve_mb == 768
    assert server._max_active_jobs == 2
    server.update_options(worker_memory_reserve_mb=512)
    assert server._worker_memory_reserve_mb == 512
    assert server._max_active_jobs == 2
