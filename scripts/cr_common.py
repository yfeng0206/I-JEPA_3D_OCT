"""Shared, dependency-light helpers for the cr_seed_v1 campaign tools.

Used by scripts/cr_campaign.py, scripts/cr_watchdog.py and scripts/cr_gate_check.py.
Only the standard library is imported at module level (the watchdog must stay
small); psutil is used when installed and replaced by ctypes calls otherwise.

Nothing in this module ever terminates a process.
"""
from __future__ import annotations

import ctypes
import datetime as _dt
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

try:  # optional
    import psutil  # type: ignore
except Exception:  # noqa: BLE001
    psutil = None

IS_WINDOWS = os.name == "nt"
GIB = 1024 ** 3
MIB = 1024 ** 2


def now_iso() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Atomic files
# ---------------------------------------------------------------------------

def _replace_with_retry(src: str, dst: str, attempts: int = 20) -> None:
    # Windows refuses os.replace while another process holds dst open
    # (a status reader, an editor, antivirus); retry briefly.
    for i in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(0.1 * (i + 1))


def atomic_write_text(path, text: str) -> None:
    path = str(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = "%s.tmp.%d" % (path, os.getpid())
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    _replace_with_retry(tmp, path)


def atomic_write_json(path, obj) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=False, default=str) + "\n")


def read_json(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        return default


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def sha256_file(path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def cached_sha256(path, cache_path, write: bool = True) -> tuple[str, bool]:
    """SHA-256 of ``path``, cached by (abs path, size, mtime_ns).

    Returns (hexdigest, from_cache).  A changed size or mtime forces a re-hash.
    """
    path = os.path.abspath(str(path))
    st = os.stat(path)
    cache = read_json(cache_path, {}) or {}
    ent = cache.get(path)
    if ent and ent.get("size") == st.st_size and ent.get("mtime_ns") == st.st_mtime_ns:
        return ent["sha256"], True
    digest = sha256_file(path)
    st2 = os.stat(path)
    if (st2.st_size, st2.st_mtime_ns) != (st.st_size, st.st_mtime_ns):
        raise RuntimeError("file changed while hashing: %s" % path)
    if not write:
        return digest, False
    cache = read_json(cache_path, {}) or {}
    cache[path] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": digest,
                   "hashed_at": now_iso()}
    atomic_write_json(cache_path, cache)
    return digest, False


# ---------------------------------------------------------------------------
# Processes (identity = PID + process creation time; never kills anything)
# ---------------------------------------------------------------------------

if IS_WINDOWS:
    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    _STILL_ACTIVE = 259

    class _FILETIME(ctypes.Structure):
        _fields_ = [("lo", ctypes.c_uint32), ("hi", ctypes.c_uint32)]

    _k32.OpenProcess.restype = ctypes.c_void_p
    _k32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    _k32.CloseHandle.argtypes = [ctypes.c_void_p]
    _k32.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
    _k32.GetProcessTimes.argtypes = [ctypes.c_void_p] + [ctypes.POINTER(_FILETIME)] * 4


def process_create_time(pid: int):
    """Process creation time as float epoch seconds, or None if not running.

    Uses only OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION)/GetProcessTimes on
    Windows: never reads another process's memory.
    """
    if pid is None or int(pid) <= 0:
        return None
    pid = int(pid)
    if IS_WINDOWS:
        h = _k32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not h:
            return None
        try:
            code = ctypes.c_uint32()
            if not _k32.GetExitCodeProcess(h, ctypes.byref(code)) or code.value != _STILL_ACTIVE:
                return None
            c, e, k, u = _FILETIME(), _FILETIME(), _FILETIME(), _FILETIME()
            if not _k32.GetProcessTimes(h, ctypes.byref(c), ctypes.byref(e),
                                        ctypes.byref(k), ctypes.byref(u)):
                return None
            ticks = (c.hi << 32) | c.lo
            return round(ticks / 1e7 - 11644473600.0, 3)
        finally:
            _k32.CloseHandle(h)
    if psutil is not None:
        try:
            p = psutil.Process(pid)
            if p.status() == getattr(psutil, "STATUS_ZOMBIE", "zombie"):
                return None
            return round(float(p.create_time()), 3)
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            return None
        except psutil.AccessDenied:
            pass
    try:
        os.kill(pid, 0)
    except OSError:
        return None
    return 0.0


def same_process_alive(pid, create_time, tol: float = 1.0) -> bool:
    """True iff ``pid`` is running AND was created at ``create_time`` (PID-reuse safe)."""
    if pid is None or create_time is None:
        return False
    ct = process_create_time(pid)
    return ct is not None and abs(ct - float(create_time)) <= tol


def own_identity() -> dict:
    return {"pid": os.getpid(), "create_time": process_create_time(os.getpid()),
            "host": socket.gethostname(), "started_at": now_iso(),
            "argv": list(sys.argv)}


# ---------------------------------------------------------------------------
# Exclusive lock: OS-held byte-range lock for the owner's lifetime
# ---------------------------------------------------------------------------

class LockHeld(RuntimeError):
    pass


if IS_WINDOWS:
    import msvcrt

    def _os_trylock(fd) -> bool:
        os.lseek(fd, 0, os.SEEK_SET)
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False

    def _os_unlock(fd) -> None:
        os.lseek(fd, 0, os.SEEK_SET)
        try:
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
else:
    import fcntl

    def _os_trylock(fd) -> bool:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            return False

    def _os_unlock(fd) -> None:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass


class ExclusiveLock:
    """Single-instance lock held by the operating system for the owner's lifetime.

    Ownership is an exclusive byte-range lock (Windows ``LockFile`` via msvcrt;
    ``flock`` elsewhere) on ``path``, kept until ``release()`` or process death, when
    the OS drops it.  There is no stale-file reclamation, so two contenders can never
    both own it.  Owner metadata (pid, creation time, argv) is informational only and
    lives in ``<path>.owner.json``.  No process is ever killed.
    """

    def __init__(self, path):
        self.path = str(path)
        self.meta_path = self.path + ".owner.json"
        self.held = False
        self.info = None
        self._fd = None

    def owner(self):
        try:
            with open(self.meta_path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except FileNotFoundError:
            return None
        except (ValueError, OSError):
            return {"unreadable": True}

    def is_locked_by_other(self) -> bool:
        """Probe without keeping ownership (read-only status)."""
        if self.held:
            return False
        if not os.path.exists(self.path):
            return False
        try:
            fd = os.open(self.path, os.O_RDWR)
        except OSError:
            return True
        try:
            if _os_trylock(fd):
                _os_unlock(fd)
                return False
            return True
        finally:
            os.close(fd)

    def acquire(self) -> dict:
        if self.held:
            return {"acquired": True, "previous_owner": None}
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT)
        if not _os_trylock(fd):
            os.close(fd)
            own = self.owner() or {}
            raise LockHeld("lock %s is held by another live process (last recorded owner pid=%s started %s)"
                           % (self.path, own.get("pid"), own.get("started_at")))
        self._fd = fd
        self.held = True
        previous = self.owner()
        self.info = own_identity()
        atomic_write_json(self.meta_path, self.info)
        return {"acquired": True, "previous_owner": previous}

    def release(self) -> None:
        if not self.held:
            return
        try:
            own = self.owner()
            if own and own.get("pid") == os.getpid():
                try:
                    os.remove(self.meta_path)
                except OSError:
                    pass
        finally:
            _os_unlock(self._fd)
            os.close(self._fd)
            self._fd = None
            self.held = False

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()


# ---------------------------------------------------------------------------
# Machine probes (all read-only)
# ---------------------------------------------------------------------------

if IS_WINDOWS:
    class _MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_uint32), ("dwMemoryLoad", ctypes.c_uint32),
                    ("ullTotalPhys", ctypes.c_uint64), ("ullAvailPhys", ctypes.c_uint64),
                    ("ullTotalPageFile", ctypes.c_uint64), ("ullAvailPageFile", ctypes.c_uint64),
                    ("ullTotalVirtual", ctypes.c_uint64), ("ullAvailVirtual", ctypes.c_uint64),
                    ("ullAvailExtendedVirtual", ctypes.c_uint64)]


def memory_status() -> dict:
    """RAM and commit charge in MB. Commit limit = RAM + pagefiles (Windows)."""
    out = {"ram_total_mb": None, "ram_avail_mb": None, "commit_used_mb": None,
           "commit_limit_mb": None, "pagefile_used_mb": None, "pagefile_total_mb": None}
    if IS_WINDOWS:
        ms = _MEMORYSTATUSEX()
        ms.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
        if _k32.GlobalMemoryStatusEx(ctypes.byref(ms)):
            out["ram_total_mb"] = ms.ullTotalPhys / MIB
            out["ram_avail_mb"] = ms.ullAvailPhys / MIB
            out["commit_limit_mb"] = ms.ullTotalPageFile / MIB
            out["commit_used_mb"] = (ms.ullTotalPageFile - ms.ullAvailPageFile) / MIB
    elif psutil is not None:
        vm = psutil.virtual_memory()
        out["ram_total_mb"] = vm.total / MIB
        out["ram_avail_mb"] = vm.available / MIB
    if psutil is not None:
        try:
            sw = psutil.swap_memory()
            out["pagefile_used_mb"] = sw.used / MIB
            out["pagefile_total_mb"] = sw.total / MIB
        except Exception:  # noqa: BLE001
            pass
    for k, v in list(out.items()):
        if isinstance(v, float):
            out[k] = round(v, 1)
    return out


def disk_free_gib(path) -> float | None:
    try:
        return round(shutil.disk_usage(str(path)).free / GIB, 2)
    except OSError:
        return None


def _run_nvidia_smi(args, timeout=20):
    exe = shutil.which("nvidia-smi")
    if exe is None:
        raise FileNotFoundError("nvidia-smi not found on PATH")
    kw = {}
    if IS_WINDOWS:
        kw["creationflags"] = 0x08000000  # CREATE_NO_WINDOW
    r = subprocess.run([exe] + list(args), capture_output=True, text=True, timeout=timeout, **kw)
    if r.returncode != 0:
        raise RuntimeError("nvidia-smi rc=%d: %s" % (r.returncode, (r.stderr or r.stdout).strip()))
    return r.stdout


def gpu_compute_apps() -> list[dict]:
    """Processes holding a CUDA compute context (nvidia-smi). Raises if unavailable."""
    out = _run_nvidia_smi(["--query-compute-apps=pid,process_name,used_memory",
                           "--format=csv,noheader,nounits"])
    apps = []
    for line in out.splitlines():
        line = line.strip()
        if not line or "No running" in line:
            continue
        parts = [p.strip() for p in line.split(",")]
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        apps.append({"pid": pid, "name": parts[1] if len(parts) > 1 else "",
                     "used_mb": parts[2] if len(parts) > 2 else ""})
    return apps


GPU_FIELDS = ("memory.used", "memory.total", "utilization.gpu", "temperature.gpu",
              "power.draw", "power.limit")


def gpu_status() -> dict:
    out = _run_nvidia_smi(["--query-gpu=" + ",".join(GPU_FIELDS), "--format=csv,noheader,nounits"])
    line = out.strip().splitlines()[0]
    vals = [v.strip() for v in line.split(",")]

    def num(v):
        try:
            return float(v)
        except ValueError:
            return None
    d = dict(zip(GPU_FIELDS, (num(v) for v in vals)))
    return {"gpu_mem_used_mb": d.get("memory.used"), "gpu_mem_total_mb": d.get("memory.total"),
            "gpu_util_pct": d.get("utilization.gpu"), "gpu_temp_c": d.get("temperature.gpu"),
            "gpu_power_w": d.get("power.draw"), "gpu_power_limit_w": d.get("power.limit")}


TRAINING_CMD_MARKERS = ("train_patch.py", "eval_downstream.py", "torchrun",
                        "run_guarded_probe.py", "campaign_supervisor.py", "chain_replication.py",
                        "campaign_chain.py")
PYTHONISH = ("python", "torchrun")


def _is_pythonish(name: str) -> bool:
    base = os.path.basename(str(name or "")).lower()
    return any(base.startswith(p) for p in PYTHONISH)


def foreign_compute_python(exclude_pids=()) -> list[dict]:
    """Running python processes whose command line looks like training/probing.

    Only python-named processes are ever opened: reading the command line of
    other processes (e.g. an anti-cheat-protected game) can freeze the caller.
    """
    if psutil is None:
        return []
    found = []
    excl = {int(p) for p in exclude_pids if p}
    for p in psutil.process_iter(["pid", "name"]):
        try:
            if p.info["pid"] in excl or p.info["pid"] == os.getpid():
                continue
            if not _is_pythonish(p.info.get("name")):
                continue
            cmd = " ".join(p.cmdline() or [])
            if any(m in cmd for m in TRAINING_CMD_MARKERS):
                found.append({"pid": p.info["pid"], "name": p.info.get("name"), "cmdline": cmd[:300]})
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return found


def split_gpu_apps(apps) -> tuple[list, list]:
    """(python/torch processes, everything else) from gpu_compute_apps().

    On Windows/WDDM nvidia-smi lists every process with a GPU context (dwm,
    explorer, browsers, games) and reports no per-process memory, so only
    python/torch processes are treated as competing compute jobs.
    """
    py, other = [], []
    for a in apps:
        (py if _is_pythonish(a.get("name")) else other).append(a)
    return py, other


def suspended_in_tree(pid, create_time) -> list[int] | None:
    """PIDs in the process tree rooted at our own ``pid`` whose threads are all
    suspended (psutil status 'stopped'); None if unknown.  Reads only thread
    states of that tree, never process memory."""
    if psutil is None or not same_process_alive(pid, create_time):
        return None
    try:
        root = psutil.Process(int(pid))
        procs = [root] + root.children(recursive=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None
    out = []
    for p in procs:
        try:
            if p.status() == psutil.STATUS_STOPPED:
                out.append(p.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return out


def tail_text(path, max_bytes: int = 262144) -> str:
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - max_bytes))
            return fh.read().decode("utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def write_alert(alert_dir, kind: str, text: str) -> str:
    os.makedirs(str(alert_dir), exist_ok=True)
    name = "ALERT_%s_%s.txt" % (kind, time.strftime("%Y%m%d_%H%M%S"))
    path = os.path.join(str(alert_dir), name)
    n = 1
    while os.path.exists(path):
        path = os.path.join(str(alert_dir), name.replace(".txt", "_%d.txt" % n))
        n += 1
    atomic_write_text(path, "[%s] %s\n%s\n" % (now_iso(), kind, text))
    return path


def as_path(p) -> Path:
    return Path(str(p))
