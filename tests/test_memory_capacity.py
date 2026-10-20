from pathlib import Path
from types import SimpleNamespace

import pytest

from livekit.agents.utils.hw import memory

pytestmark = pytest.mark.unit


@pytest.fixture
def cgroup(tmp_path, monkeypatch):
    memory._warn_unresolved_cgroups.cache_clear()
    membership = tmp_path / "cgroup"
    mountinfo = tmp_path / "mountinfo"
    monkeypatch.setattr(memory, "_CGROUP_FILE", membership)
    monkeypatch.setattr(memory, "_MOUNTINFO_FILE", mountinfo)
    monkeypatch.setattr(memory.sys, "platform", "linux")
    monkeypatch.setattr(memory.psutil, "virtual_memory", lambda: SimpleNamespace(available=10_000))

    def setup(*, v1=False, root="/", member="/", mount_name="memory", limit="2000", usage="500"):
        mount = tmp_path / mount_name
        mount.mkdir()
        membership.write_text(f"5:cpu,memory:{member}\n" if v1 else f"0::{member}\n")
        escaped = str(mount).replace(" ", r"\040")
        mountinfo.write_text(
            f"29 23 0:26 {root} {escaped} rw - "
            + ("cgroup cgroup rw,memory\n" if v1 else "cgroup2 cgroup rw\n")
        )
        relative = Path(member).relative_to(root) if member.startswith(root) else Path(".")
        current = mount / relative
        current.mkdir(parents=True, exist_ok=True)
        write_counters(current, v1=v1, limit=limit, usage=usage)
        return mount, current

    yield setup
    memory._warn_unresolved_cgroups.cache_clear()


def write_counters(path, *, v1=False, limit="2000", usage="500"):
    (path / ("memory.limit_in_bytes" if v1 else "memory.max")).write_text(limit)
    (path / ("memory.usage_in_bytes" if v1 else "memory.current")).write_text(usage)


@pytest.mark.parametrize("v1", [False, True])
def test_finite_cgroup_limits_include_all_charged_memory(cgroup, v1):
    cgroup(v1=v1)
    assert memory.get_available_memory() == 1500


@pytest.mark.parametrize("v1", [False, True])
def test_tightest_visible_ancestor_wins(cgroup, v1):
    mount, current = cgroup(v1=v1, member="/pod/worker", limit="10000", usage="1000")
    write_counters(current.parent, v1=v1, limit="4000", usage="3500")
    write_counters(mount, v1=v1, limit="20000", usage="1000")
    assert memory.get_available_memory() == 500


@pytest.mark.parametrize("root,member", [("/pod", "/pod/worker"), ("/pod", "/")])
def test_mount_root_and_cgroup_namespaces(cgroup, root, member):
    cgroup(root=root, member=member)
    assert memory.get_available_memory() == 1500


def test_escaped_custom_mount_point(cgroup):
    cgroup(mount_name="custom memory mount")
    assert memory.get_available_memory() == 1500


@pytest.mark.parametrize("v1,limit", [(False, "max"), (True, "9223372036854771712")])
def test_unlimited_cgroup_uses_host_availability(cgroup, v1, limit):
    cgroup(v1=v1, limit=limit)
    assert memory.get_available_memory() == 10_000


def test_host_availability_is_also_a_bound(cgroup):
    cgroup(limit="20000", usage="500")
    assert memory.get_available_memory() == 10_000


def test_usage_over_limit_has_no_headroom(cgroup):
    cgroup(limit="500", usage="600")
    assert memory.get_available_memory() == 0


@pytest.mark.parametrize(
    "limit,usage", [("broken", "500"), ("2000", "broken"), ("0", "0"), ("2000", "-1")]
)
def test_invalid_counters_raise(cgroup, limit, usage):
    cgroup(limit=limit, usage=usage)
    with pytest.raises(ValueError):
        memory.get_available_memory()


def test_missing_usage_raises(cgroup):
    _, current = cgroup()
    (current / "memory.current").unlink()
    with pytest.raises(OSError):
        memory.get_available_memory()


def test_missing_limit_with_usage_present_raises(cgroup):
    _, current = cgroup()
    (current / "memory.max").unlink()
    with pytest.raises(OSError):
        memory.get_available_memory()


def test_container_membership_outside_visible_namespace(cgroup):
    cgroup()
    memory._CGROUP_FILE.write_text("0::/../../host/container\n")
    assert memory.get_available_memory() == 1500


def test_disabled_memory_controller_uses_host(cgroup):
    _, current = cgroup()
    (current / "memory.max").unlink()
    (current / "memory.current").unlink()
    assert memory.get_available_memory() == 10_000


def test_non_linux_uses_host_without_proc_files(monkeypatch):
    monkeypatch.setattr(memory.sys, "platform", "darwin")
    monkeypatch.setattr(memory.psutil, "virtual_memory", lambda: SimpleNamespace(available=1234))
    assert memory.get_available_memory() == 1234


@pytest.mark.parametrize("v1", [False, True])
@pytest.mark.parametrize("member", ["/unmapped-worker", "/../../host/container"])
def test_unmapped_cgroup_warns_once_and_uses_host(cgroup, caplog, v1, member):
    cgroup(v1=v1, root="/pod", member=member)
    assert memory.get_available_memory() == 10_000
    assert memory.get_available_memory() == 10_000
    warnings = [record for record in caplog.records if "could not resolve" in record.message]
    assert len(warnings) == 1
    assert "using host memory availability" in warnings[0].message
    assert warnings[0].cgroup_paths == (member,)


def test_unrelated_mount_does_not_warn_when_membership_is_resolved(cgroup, caplog):
    cgroup(member="/worker")
    original = memory._MOUNTINFO_FILE.read_text()
    memory._MOUNTINFO_FILE.write_text(
        "30 23 0:26 /unrelated /unrelated-mount rw - cgroup2 cgroup rw\n" + original
    )
    assert memory.get_available_memory() == 1500
    assert "could not resolve" not in caplog.text


def test_missing_cgroup_mount_warns_and_uses_host(cgroup, caplog):
    cgroup()
    memory._MOUNTINFO_FILE.write_text("29 23 0:26 / /tmp rw - tmpfs tmpfs rw\n")
    assert memory.get_available_memory() == 10_000
    assert "using host memory availability" in caplog.text
