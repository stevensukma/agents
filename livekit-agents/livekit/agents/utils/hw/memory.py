from __future__ import annotations

import re
import sys
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path, PurePosixPath

import psutil

from ...log import logger

_CGROUP_FILE = Path("/proc/self/cgroup")
_MOUNTINFO_FILE = Path("/proc/self/mountinfo")


@lru_cache(maxsize=16)
def _warn_unresolved_cgroups(paths: tuple[str, ...]) -> None:
    logger.warning(
        "could not resolve worker memory cgroup; using host memory availability "
        "without the unresolved cgroup limits",
        extra={"cgroup_paths": paths},
    )


def _unescape_mount_path(value: str) -> str:
    return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), value)


def _cgroup_memory_dirs() -> Iterator[tuple[Path, bool]]:
    memberships = {}
    mapped = set[bool]()
    for line in _CGROUP_FILE.read_text().splitlines():
        _, controllers, path = line.split(":", 2)
        if not controllers or "memory" in controllers.split(","):
            memberships[bool(controllers)] = PurePosixPath(path)

    for line in _MOUNTINFO_FILE.read_text().splitlines():
        mount, filesystem = line.split(" - ", 1)
        fields, fs_fields = mount.split(), filesystem.split()
        is_v1 = fs_fields[0] == "cgroup"
        if fs_fields[0] != "cgroup2" and not (is_v1 and "memory" in fs_fields[2].split(",")):
            continue
        membership = memberships.get(is_v1)
        if membership is None:
            continue

        mount_root = PurePosixPath(_unescape_mount_path(fields[3]))
        mount_point = Path(_unescape_mount_path(fields[4]))
        try:
            relative = membership.relative_to(mount_root)
        except ValueError:
            # A cgroup namespace can expose its root as / even when the mount
            # retains the host's original root path.
            if membership != PurePosixPath("/"):
                continue
            relative = PurePosixPath(".")
        # Some container runtimes report a path outside the visible namespace.
        # Only the mounted root (and its visible limits) is accessible then.
        current = mount_point if ".." in relative.parts else mount_point / relative
        mapped.add(is_v1)
        while True:
            yield current, is_v1
            if current == mount_point:
                break
            current = current.parent

    unresolved = tuple(str(path) for is_v1, path in memberships.items() if is_v1 not in mapped)
    if unresolved:
        _warn_unresolved_cgroups(unresolved)


def get_available_memory() -> int:
    """Return available bytes, bounded by visible Linux cgroup memory limits.

    Includes all processes and charged cache in the cgroup, not just the worker's
    RSS. Checks the process's cgroup and each visible ancestor, including custom
    mount points. Unlimited cgroups and other platforms use host availability.
    Unreadable or malformed counters raise rather than overstate capacity.
    Unresolved cgroup paths warn and fall back to host availability for those
    limits. Warnings are suppressed for repeated occurrences of the same paths.
    """
    available = int(psutil.virtual_memory().available)
    if not sys.platform.startswith("linux"):
        return available

    for directory, is_v1 in _cgroup_memory_dirs():
        limit_path = directory / ("memory.limit_in_bytes" if is_v1 else "memory.max")
        usage_path = directory / ("memory.usage_in_bytes" if is_v1 else "memory.current")
        # The memory controller may not be enabled on a cgroup v2 mount.
        if not limit_path.exists() and not usage_path.exists():
            continue
        limit_text = limit_path.read_text().strip()
        if limit_text == "max" and not is_v1:
            continue
        limit = int(limit_text)
        if is_v1 and limit >= 1 << 60:  # kernel's unlimited sentinel
            continue
        usage = int(usage_path.read_text().strip())
        if limit <= 0 or usage < 0:
            raise ValueError("invalid cgroup memory counters")
        available = min(available, max(0, limit - usage))

    return available
