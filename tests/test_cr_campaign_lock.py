"""Campaign lock: OS-held exclusive ownership; no stale-file reclamation race (V3 P0-1)."""
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from scripts import cr_common as cc

SCRIPTS = str(Path(cc.__file__).parent)


def holder(lock_path, hold_s=60.0, release=False):
    """Child process that acquires the lock, prints READY, then holds it (or exits without release)."""
    code = ("import sys, time; sys.path.insert(0, %r); import cr_common as cc\n"
            "lk = cc.ExclusiveLock(%r)\n"
            "try:\n    lk.acquire()\nexcept cc.LockHeld:\n    print('LOSE', flush=True); sys.exit(3)\n"
            "print('READY', flush=True)\n"
            "time.sleep(%r)\n") % (SCRIPTS, str(lock_path), hold_s)
    p = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    line = p.stdout.readline().strip()
    return p, line


def test_acquire_release_and_reacquire(tmp_path):
    lk = cc.ExclusiveLock(tmp_path / "c.lock")
    assert lk.acquire()["acquired"] and lk.held
    own = json.loads((tmp_path / "c.lock.owner.json").read_text())
    assert own["pid"] == os.getpid() and own["create_time"] == cc.process_create_time(os.getpid())
    with pytest.raises(cc.LockHeld):
        cc.ExclusiveLock(tmp_path / "c.lock").acquire()
    lk.release()
    assert not lk.held and not (tmp_path / "c.lock.owner.json").exists()
    lk2 = cc.ExclusiveLock(tmp_path / "c.lock")
    assert lk2.acquire()["acquired"]
    lk2.release()


def test_live_foreign_owner_is_never_taken_over_or_killed(tmp_path):
    p, line = holder(tmp_path / "c.lock")
    try:
        assert line == "READY"
        lk = cc.ExclusiveLock(tmp_path / "c.lock")
        assert lk.is_locked_by_other()
        with pytest.raises(cc.LockHeld, match="held by another live process"):
            lk.acquire()
        assert p.poll() is None, "lock contention must never kill the owner"
    finally:
        p.kill()
        p.wait()


def test_dead_owner_releases_via_os(tmp_path):
    p, line = holder(tmp_path / "c.lock", hold_s=0.0)
    assert line == "READY"
    p.wait(timeout=60)
    assert (tmp_path / "c.lock.owner.json").exists()  # metadata left behind by the dead owner
    lk = cc.ExclusiveLock(tmp_path / "c.lock")
    info = lk.acquire()
    assert info["previous_owner"]["pid"] not in (None, os.getpid())  # the dead holder (venv launcher's child)
    assert json.loads((tmp_path / "c.lock.owner.json").read_text())["pid"] == os.getpid()
    lk.release()


def test_metadata_alone_never_blocks_or_grants(tmp_path):
    # A metadata/lock file naming a live process does not block: only the OS lock does.
    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        (tmp_path / "c.lock").write_text(json.dumps({"pid": sleeper.pid}))
        (tmp_path / "c.lock.owner.json").write_text("{not json")
        lk = cc.ExclusiveLock(tmp_path / "c.lock")
        assert lk.acquire()["acquired"]
        assert sleeper.poll() is None
        lk.release()
    finally:
        sleeper.kill()
        sleeper.wait()


def test_stale_reclamation_race_has_one_winner_threads(tmp_path):
    # V3 P0-1 interleaving: two contenders both observe the same dead owner's metadata.
    lock_path = tmp_path / "race.lock"
    lock_path.write_text(json.dumps({"pid": 2147483647, "create_time": 1.0}))
    (tmp_path / "race.lock.owner.json").write_text(json.dumps({"pid": 2147483647, "create_time": 1.0}))
    both_read = threading.Barrier(2)

    class RacingLock(cc.ExclusiveLock):
        def owner(self):
            observed = super().owner()
            try:
                both_read.wait(timeout=2)
            except threading.BrokenBarrierError:
                pass
            return observed

    def attempt(_):
        lk = RacingLock(lock_path)
        try:
            lk.acquire()
            return lk
        except cc.LockHeld:
            return None
    with ThreadPoolExecutor(2) as ex:
        res = list(ex.map(attempt, range(2)))
    winners = [r for r in res if r is not None]
    assert len(winners) == 1 and winners[0].held
    winners[0].release()


def test_stale_reclamation_race_has_one_winner_processes(tmp_path):
    lock_path = tmp_path / "c.lock"
    p, line = holder(lock_path, hold_s=0.0)  # dead previous owner leaves its metadata
    p.wait(timeout=60)
    code = ("import sys, time; sys.path.insert(0, %r); import cr_common as cc\n"
            "t0 = float(sys.argv[1])\n"
            "while time.time() < t0: time.sleep(0.001)\n"
            "try:\n    cc.ExclusiveLock(%r).acquire(); print('WIN', flush=True); time.sleep(3)\n"
            "except cc.LockHeld:\n    print('LOSE', flush=True)\n") % (SCRIPTS, str(lock_path))
    start = str(time.time() + 3.0)
    procs = [subprocess.Popen([sys.executable, "-c", code, start], stdout=subprocess.PIPE, text=True)
             for _ in range(6)]
    outs = [q.communicate(timeout=90)[0].strip() for q in procs]
    assert outs.count("WIN") == 1, outs


def test_concurrent_acquire_single_winner_threads(tmp_path):
    path = tmp_path / "c.lock"

    def attempt(_):
        lk = cc.ExclusiveLock(path)
        try:
            lk.acquire()
            return lk
        except cc.LockHeld:
            return None
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(attempt, range(16)))
    winners = [r for r in res if r is not None]
    assert len(winners) == 1
    winners[0].release()


def test_same_process_alive_checks_create_time():
    me = os.getpid()
    ct = cc.process_create_time(me)
    assert cc.same_process_alive(me, ct)
    assert not cc.same_process_alive(me, ct - 50)
    assert not cc.same_process_alive(None, ct)
