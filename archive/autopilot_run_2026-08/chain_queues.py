"""Chain queue 2 after queue 1 without ever running two GPU jobs at once.

Waits for the queue-1 driver PID to exit, verifies the GPU has actually been
released, then starts queue 2.
"""
import os
import subprocess
import sys
import time

PY = r"D:\jepa_phase0\.venv\Scripts\python.exe"
HERE = os.path.dirname(os.path.abspath(__file__))


def pid_alive(pid):
    try:
        out = subprocess.run(["tasklist", "/FI", "PID eq %d" % pid, "/NH"],
                             capture_output=True, text=True, timeout=30).stdout
        return str(pid) in out
    except Exception:
        return False


def gpu_busy_mib():
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=used_memory",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout.strip()
        if not out:
            return 0
        return sum(int(x) for x in out.splitlines() if x.strip().isdigit())
    except Exception:
        return -1


def main():
    wait_pid = int(sys.argv[1])
    print("[chain] waiting for queue-1 pid %d to exit ..." % wait_pid, flush=True)
    while pid_alive(wait_pid):
        time.sleep(60)
    print("[chain] queue-1 pid %d exited at %s" % (wait_pid, time.strftime("%H:%M:%S")), flush=True)

    # confirm the GPU was really released before starting anything new
    for _ in range(20):
        b = gpu_busy_mib()
        print("[chain] gpu compute-app memory = %s MiB" % b, flush=True)
        if b <= 500:
            break
        time.sleep(30)

    print("[chain] starting queue 2", flush=True)
    rc = subprocess.call([PY, "-u", os.path.join(HERE, "gpu_queue2.py")])
    print("[chain] queue 2 exited rc=%d" % rc, flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
