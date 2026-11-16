"""Shared process measurements and source/environment records."""
from __future__ import annotations

import ctypes
import hashlib
from datetime import datetime, timezone
from pathlib import Path
import platform
import subprocess
import sys

def peak_rss_bytes() -> int | None:
    """Process lifetime peak working set, including imported/native libraries."""
    if sys.platform == "win32":
        class MemoryCounters(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong)] + [
                (name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize",
                "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
        counters = MemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        handle = ctypes.windll.kernel32.GetCurrentProcess
        handle.restype = ctypes.c_void_p
        memory = ctypes.windll.psapi.GetProcessMemoryInfo
        memory.argtypes = [ctypes.c_void_p, ctypes.POINTER(MemoryCounters), ctypes.c_ulong]
        if memory(handle(), ctypes.byref(counters), counters.cb):
            return counters.PeakWorkingSetSize
        return None
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value if sys.platform == "darwin" else value * 1024)
    except ImportError:
        return None


def environment_record() -> dict:
    import numpy
    import scipy
    import sklearn
    root = Path(__file__).resolve().parents[3]
    def git(*args):
        try:
            return subprocess.check_output(["git", *args], cwd=root, text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    status = git("status", "--porcelain")
    paths = sorted((root / 'src/afs_rgmnb').rglob('*.py'))
    paths += sorted((root / 'scripts').glob('*.py'))
    code_hashes = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in paths}
    return {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(), "platform": platform.platform(),
            "numpy": numpy.__version__, "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__, "source_commit": git("rev-parse", "HEAD"),
            "source_worktree_dirty": None if status is None else bool(status),
            "code_sha256": code_hashes}
