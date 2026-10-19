# LiveKit Agents for Python

Realtime framework for production-grade multimodal and voice AI agents.

See [https://docs.livekit.io/agents/](https://docs.livekit.io/agents/) for quickstarts, documentation, and examples.

```python
from dotenv import load_dotenv

from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.plugins import openai

load_dotenv()

async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()

    session = AgentSession(
        llm=openai.realtime.RealtimeModel(
            voice="coral"
        )
    )

    await session.start(
        room=ctx.room,
        agent=Agent(instructions="You are a helpful voice AI assistant.")
    )

    await session.generate_reply(
        instructions="Greet the user and offer your assistance."
    )


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
```

### Optional memory and concurrency admission

For self-hosted workers, enable additional admission limits on `AgentServer`:

```python
server = agents.AgentServer(
    worker_memory_reserve_mb=768,
    max_active_jobs=2,
)
```

Both settings default to `0` (disabled) and are also accepted by `WorkerOptions`
and `AgentServer.update_options()` before startup. They apply alongside the
existing `load_fnc` and `load_threshold`, without changing the default CPU policy.

`worker_memory_reserve_mb` requires that much available memory for the next job,
in MiB (1,048,576 bytes). Linux workers check the tightest visible cgroup v1/v2
headroom, including ancestor limits, bounded by host available memory. Cgroup
usage includes all charged processes and cache. Other platforms and unlimited
cgroups use host available memory. Limits outside the mounted cgroup hierarchy
cannot be inspected. Unreadable or malformed counters block new jobs until
readings recover.

Each pending assignment reserves the same amount of memory. For example, with
1,000 MiB available and a 768 MiB reserve, one pending job prevents another
admission until it launches or releases its reservation. Running jobs are already
represented in measured memory usage. `max_active_jobs` separately caps the sum
of running jobs and pending assignments.

A worker at capacity advertises itself as full and answers availability requests
with `available=False, terminate=False`, allowing the server to try another
worker. It becomes available again when capacity recovers. These limits apply in
both development and production; simulation mode bypasses them. They never kill
running jobs. The separate `job_memory_warn_mb` and `job_memory_limit_mb` options
still control per-process warnings and termination.

Choose the reserve and job cap for your workload, accounting for prewarmed
processes and startup spikes. Admission is a headroom check, not a guarantee
against later memory growth or an OOM in an existing job.
