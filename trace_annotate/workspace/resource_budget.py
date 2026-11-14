"""Host-memory admission and ownership of one isolated model process tree.

The default 2 GiB model allocation plus 1 GiB host reserve is a conservative
local deployment policy based on measured CPU model runs. It is not a promise
about every model or a platform-wide memory reservation.
"""
from __future__ import annotations

from contextlib import contextmanager
import ctypes
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import uuid

from .errors import WorkspaceError
from .slots import get_prepare_slot, SlotClosed, SlotTimeout

GIB = 1024 ** 3
CONTRACT = "isolated-model-memory-v1"
HOST_SLOT_CONTRACT = "traceannotate-host-model-slot-v1"
HOST_SLOT_FILENAME = "model-resource-v1.lock"
HOST_SLOT_POLL_SECONDS = .05


def memory_available():
    """Return physical bytes reported available by the host operating system."""
    if os.name == "nt":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong) for name in ("total_phys", "avail_phys", "total_page",
                "avail_page", "total_virtual", "avail_virtual", "avail_extended")]
        value = MemoryStatus()
        value.length = ctypes.sizeof(value)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value)):
            raise OSError("GlobalMemoryStatusEx failed")
        return int(value.avail_phys)
    try:
        for line in Path("/proc/meminfo").read_text("ascii").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    raise OSError("OS available physical memory is unavailable")


def policy():
    def amount(name, default, minimum, maximum):
        raw = os.environ.get(name, str(default))
        try:
            value = float(raw)
        except ValueError as exc:
            raise WorkspaceError("INVALID_MODEL_RESOURCE_POLICY", name + " must be numeric") from exc
        if not math.isfinite(value) or not minimum <= value <= maximum:
            raise WorkspaceError("INVALID_MODEL_RESOURCE_POLICY", name + " is outside its supported range")
        return value
    configured_lock_root = os.environ.get("TRACE_ANNOTATE_MODEL_LOCK_ROOT")
    if configured_lock_root:
        lock_root = Path(configured_lock_root).expanduser().absolute()
        host_lock = {"contract": HOST_SLOT_CONTRACT, "kind": "file",
                     "identity": str(lock_root / HOST_SLOT_FILENAME)}
    elif os.name == "nt":
        host_lock = {"contract": HOST_SLOT_CONTRACT, "kind": "windows_named_mutex",
                     "identity": "Local\\TraceAnnotateHostModelSlot-v1"}
    elif sys.platform.startswith("linux"):
        host_lock = {"contract": HOST_SLOT_CONTRACT, "kind": "linux_abstract_socket",
                     "identity": "traceannotate-host-model-slot-v1-" + str(os.getuid())}
    else:
        raise WorkspaceError(
            "INVALID_MODEL_RESOURCE_POLICY",
            "TRACE_ANNOTATE_MODEL_LOCK_ROOT must name an application-owned project path on this platform")
    return {"contract": CONTRACT,
            "model_bytes": int(amount("TRACE_ANNOTATE_MODEL_BUDGET_MIB", 2048, 2048, 65536) * 1024 ** 2),
            "reserve_bytes": int(amount("TRACE_ANNOTATE_MODEL_RESERVE_MIB", 1024, 1024, 65536) * 1024 ** 2),
            "wait_seconds": amount("TRACE_ANNOTATE_MODEL_WAIT_SECONDS", 60, 0, 600),
            "frame_seconds": amount("TRACE_ANNOTATE_MODEL_FRAME_SECONDS", 60, 5, 600),
            "phase_seconds": amount("TRACE_ANNOTATE_MODEL_PHASE_SECONDS", 3600, 60, 86400),
            "host_lock": host_lock}


def _available():
    try:
        value = memory_available()
    except (OSError, ValueError, AttributeError) as exc:
        raise WorkspaceError("MEMORY_STATUS_UNAVAILABLE", "Cannot read available host memory",
                             retryable=True) from exc
    if type(value) is not int or value < 0:
        raise WorkspaceError("MEMORY_STATUS_UNAVAILABLE", "Invalid host memory measurement", retryable=True)
    return value


class HostSlotTimeout(Exception):
    """The OS-backed host slot remained owned for the complete wait budget."""

    def __init__(self, waited_seconds):
        super().__init__("host model slot wait budget exhausted")
        self.waited_seconds = waited_seconds


class HostModelSlot:
    """One auto-released cross-process slot backed by an OS primitive.

    Windows uses a named mutex and Linux uses an abstract Unix socket, neither
    of which creates a filesystem artifact. Other POSIX platforms require an
    explicit project-local lock root and use ``flock``. In every case the open
    kernel handle is the owner, so a process crash releases the slot.
    """

    def __init__(self, specification):
        if (not isinstance(specification, dict)
                or set(specification) != {"contract", "kind", "identity"}
                or specification.get("contract") != HOST_SLOT_CONTRACT
                or specification.get("kind") not in {"file", "windows_named_mutex", "linux_abstract_socket"}
                or not isinstance(specification.get("identity"), str)
                or not specification["identity"]):
            raise WorkspaceError("INVALID_MODEL_RESOURCE_POLICY", "Invalid host model lock specification")
        self.specification = dict(specification)
        self.kind = specification["kind"]
        self.identity = specification["identity"]
        self.path = Path(self.identity).absolute() if self.kind == "file" else None
        self.handle = None
        self.kernel = None
        self.acquired = False
        self.recovered_abandoned_owner = False

    @property
    def scope_sha256(self):
        value = self.kind + ":" + self.identity
        if self.kind == "file" and os.name == "nt":
            value = value.lower()
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def _open(self):
        if self.kind == "windows_named_mutex":
            if os.name != "nt" or not self.identity.startswith("Local\\TraceAnnotateHostModelSlot-"):
                raise WorkspaceError("INVALID_MODEL_RESOURCE_POLICY", "Invalid Windows host model mutex")
            from ctypes import wintypes
            self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            self.kernel.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
            self.kernel.CreateMutexW.restype = wintypes.HANDLE
            self.kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
            self.kernel.WaitForSingleObject.restype = wintypes.DWORD
            self.kernel.ReleaseMutex.argtypes = [wintypes.HANDLE]
            self.kernel.ReleaseMutex.restype = wintypes.BOOL
            self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
            self.handle = self.kernel.CreateMutexW(None, False, self.identity)
            if not self.handle:
                raise OSError(ctypes.get_last_error(), "Cannot create host model mutex")
            return
        if self.kind == "linux_abstract_socket":
            if not sys.platform.startswith("linux") or not self.identity.startswith("traceannotate-host-model-slot-"):
                raise WorkspaceError("INVALID_MODEL_RESOURCE_POLICY", "Invalid Linux host model socket")
            self.handle = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
            return
        if self.path is None or not self.path.is_absolute() or self.path.name != HOST_SLOT_FILENAME:
            raise WorkspaceError("INVALID_MODEL_RESOURCE_POLICY", "Invalid host model lock path")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.is_symlink():
            raise WorkspaceError("INVALID_MODEL_RESOURCE_POLICY", "Host model lock must not be a symbolic link")
        self.handle = self.path.open("a+b")
        self.handle.seek(0, os.SEEK_END)
        if self.handle.tell() == 0:
            self.handle.write(b"\0")
            self.handle.flush()
            os.fsync(self.handle.fileno())

    def _try_acquire(self):
        if self.kind == "windows_named_mutex":
            result = self.kernel.WaitForSingleObject(self.handle, 0)
            if result == 0:  # WAIT_OBJECT_0
                return True
            if result == 0x80:  # WAIT_ABANDONED
                self.recovered_abandoned_owner = True
                return True
            if result == 0x102:  # WAIT_TIMEOUT
                return False
            raise OSError(ctypes.get_last_error(), "Cannot wait for host model mutex")
        if self.kind == "linux_abstract_socket":
            try:
                self.handle.bind("\0" + self.identity)
                return True
            except OSError as exc:
                if exc.errno == errno.EADDRINUSE:
                    return False
                raise
        self.handle.seek(0)
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
                return True
            except OSError as exc:
                if getattr(exc, "winerror", None) in {33, 36} or exc.errno in {errno.EACCES, errno.EAGAIN}:
                    return False
                raise
        import fcntl
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError as exc:
            if exc.errno in {errno.EACCES, errno.EAGAIN}:
                return False
            raise

    def acquire(self, *, cancelled, timeout):
        self._open()
        started = time.monotonic()
        try:
            while True:
                if cancelled():
                    raise InterruptedError("host model slot wait was cancelled")
                if self._try_acquire():
                    self.acquired = True
                    return time.monotonic() - started
                elapsed = time.monotonic() - started
                if elapsed >= timeout:
                    raise HostSlotTimeout(elapsed)
                time.sleep(min(HOST_SLOT_POLL_SECONDS, max(0, timeout - elapsed)))
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.handle is None:
            return
        try:
            if self.acquired:
                if self.kind == "windows_named_mutex":
                    if not self.kernel.ReleaseMutex(self.handle):
                        raise OSError(ctypes.get_last_error(), "Cannot release host model mutex")
                elif self.kind == "file":
                    self.handle.seek(0)
                    if os.name == "nt":
                        import msvcrt
                        msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
                    else:
                        import fcntl
                        fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        finally:
            self.acquired = False
            if self.kind == "windows_named_mutex":
                self.kernel.CloseHandle(self.handle)
            else:
                self.handle.close()
            self.handle = None
            self.kernel = None

    @contextmanager
    def hold(self, *, cancelled, timeout):
        waited = self.acquire(cancelled=cancelled, timeout=timeout)
        try:
            yield {"contract": HOST_SLOT_CONTRACT, "kind": self.kind,
                   "identity": self.identity, "scope_sha256": self.scope_sha256,
                   "owner_pid": os.getpid(), "wait_seconds": waited,
                   "recovered_abandoned_owner": self.recovered_abandoned_owner}
        finally:
            self.close()


@contextmanager
def admit(settings, *, cancelled):
    """Serialise heavy models across processes and admit against real memory."""
    started = time.monotonic()
    try:
        deadline = started + settings["wait_seconds"]
        with get_prepare_slot().hold(cancelled=cancelled, timeout=max(0, deadline - time.monotonic())):
            host_slot = HostModelSlot(settings["host_lock"])
            with host_slot.hold(cancelled=cancelled,
                                timeout=max(0, deadline - time.monotonic())) as lock_record:
                required = settings["model_bytes"] + settings["reserve_bytes"]
                while True:
                    if cancelled():
                        raise WorkspaceError("CANCELLED", "Model admission cancelled")
                    available = _available()
                    if available >= required:
                        yield {"available_at_admission": available, "required_bytes": required,
                               "wait_seconds": time.monotonic() - started,
                               "host_lock": lock_record}
                        return
                    if time.monotonic() >= deadline:
                        raise WorkspaceError("INSUFFICIENT_MEMORY", "Insufficient available host memory for the model",
                            retryable=True, details={"available_bytes": available, "required_bytes": required,
                                                     "host_lock": lock_record})
                    time.sleep(min(.05, max(0, deadline - time.monotonic())))
    except (SlotClosed, SlotTimeout, HostSlotTimeout) as exc:
        details = {"waited_seconds": getattr(exc, "waited_seconds", time.monotonic() - started)}
        if isinstance(exc, HostSlotTimeout):
            lock = HostModelSlot(settings["host_lock"])
            details["host_lock"] = {"contract": HOST_SLOT_CONTRACT, "kind": lock.kind,
                                    "identity": lock.identity, "scope_sha256": lock.scope_sha256}
        raise WorkspaceError("MODEL_RESOURCE_BUSY", "Another local model is using the resource slot",
                             retryable=True, details=details) from exc
    except InterruptedError as exc:
        raise WorkspaceError("CANCELLED", "Model resource wait cancelled") from exc


def _windows_api():
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel.CreateJobObjectW.restype = wintypes.HANDLE
    kernel.OpenJobObjectW.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR]
    kernel.OpenJobObjectW.restype = wintypes.HANDLE
    kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    return kernel


class ProcessTree:
    """An owned Windows Job Object or POSIX session, including venv children."""
    def __init__(self):
        self.handle = None
        self.name = "Local\\TraceAnnotateModel-" + uuid.uuid4().hex
        if os.name == "nt":
            from ctypes import wintypes
            class BasicLimits(ctypes.Structure):
                _fields_ = [("process_time", ctypes.c_longlong), ("job_time", ctypes.c_longlong),
                    ("flags", wintypes.DWORD), ("min_working_set", ctypes.c_size_t),
                    ("max_working_set", ctypes.c_size_t), ("active_processes", wintypes.DWORD),
                    ("affinity", ctypes.c_size_t), ("priority", wintypes.DWORD), ("scheduling", wintypes.DWORD)]
            class IoCounters(ctypes.Structure):
                _fields_ = [(name, ctypes.c_ulonglong) for name in
                            ("read_ops", "write_ops", "other_ops", "read_bytes", "write_bytes", "other_bytes")]
            class Limits(ctypes.Structure):
                _fields_ = [("basic", BasicLimits), ("io", IoCounters), ("process_memory", ctypes.c_size_t),
                            ("job_memory", ctypes.c_size_t), ("peak_process", ctypes.c_size_t),
                            ("peak_job", ctypes.c_size_t)]
            self.kernel = _windows_api()
            self.handle = self.kernel.CreateJobObjectW(None, self.name)
            if not self.handle:
                raise OSError(ctypes.get_last_error(), "Cannot create owned model Job Object")
            limits = Limits()
            limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            if not self.kernel.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
                self.close()
                raise OSError("Cannot configure owned model Job Object")

    def terminate(self, process):
        if os.name == "nt":
            if self.handle:
                self.kernel.TerminateJobObject(self.handle, 1)
        elif process is not None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)

    def close(self):
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


def join_owned_job(name):
    """Called by the real worker before any model imports or child launches."""
    if os.name != "nt":
        return
    kernel = _windows_api()
    handle = kernel.OpenJobObjectW(0x0001 | 0x0004, False, name)
    if not handle:
        raise OSError(ctypes.get_last_error(), "Cannot open owned model Job Object")
    try:
        if not kernel.AssignProcessToJobObject(handle, kernel.GetCurrentProcess()):
            raise OSError(ctypes.get_last_error(), "Cannot join owned model Job Object")
    finally:
        kernel.CloseHandle(handle)


def run_owned(command, directory, *, settings, cancelled, timeout, progress_path=None):
    """Run one model phase, bounded by memory, duration, logs and cancellation."""
    directory.mkdir(parents=True, exist_ok=False)
    receipt = {"argv": command, "parent_pid": os.getpid(), "worker_group": None,
               "started_at": time.time(), "status": "starting"}
    process = None
    tree = None
    started = time.monotonic()
    try:
        tree = ProcessTree()
        with admit(settings, cancelled=cancelled) as admitted:
            receipt.update(admitted)
            env = os.environ.copy()
            env["TRACE_ANNOTATE_OWNED_MODEL_JOB"] = tree.name if os.name == "nt" else ""
            with (directory / "worker.log").open("xb") as log:
                process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                    env=env, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    start_new_session=os.name != "nt")
                receipt["wrapper_pid"] = process.pid
                receipt["worker_group"] = tree.name if os.name == "nt" else process.pid
                phase_started = time.monotonic()
                minimum = admitted["available_at_admission"]
                while process.poll() is None:
                    if cancelled():
                        raise WorkspaceError("CANCELLED", "Model worker cancelled")
                    if time.monotonic() - phase_started >= timeout:
                        raise WorkspaceError("MODEL_TIMEOUT", "Model phase exceeded its execution budget", retryable=True)
                    if progress_path is not None and progress_path.is_file():
                        if progress_path.stat().st_size > 4096:
                            raise WorkspaceError("MODEL_WORKER_PROTOCOL", "Model progress exceeds its byte budget")
                        try:
                            progress = json.loads(progress_path.read_text("utf-8"))
                        except (OSError, ValueError):
                            progress = None
                        if progress and progress.get("phase") == "frame" and (
                                time.time() - progress["started_at"] > settings["frame_seconds"]):
                            raise WorkspaceError("MODEL_TIMEOUT", "One model frame exceeded its execution budget",
                                                 retryable=True)
                    available = _available()
                    minimum = min(minimum, available)
                    receipt["minimum_available_bytes"] = minimum
                    if available < settings["reserve_bytes"]:
                        raise WorkspaceError("INSUFFICIENT_MEMORY", "Host memory reserve exhausted during model execution",
                            retryable=True, details={"available_bytes": available, "reserve_bytes": settings["reserve_bytes"]})
                    if (directory / "worker.log").stat().st_size > 1024 * 1024:
                        raise WorkspaceError("MODEL_LOG_LIMIT", "Model worker diagnostic limit exceeded")
                    time.sleep(.05)
                receipt["actual_exit_code"] = process.returncode
                if process.returncode != 0:
                    code = {75: "MODEL_MEMORY_FAILED", 76: "MODEL_TIMEOUT", 77: "MODEL_LOAD_FAILED",
                            78: "MODEL_PROGRESS_FAILED"}.get(
                        process.returncode, "MODEL_WORKER_FAILED")
                    raise WorkspaceError(code, "Model worker exited unsuccessfully", retryable=True,
                                         details={"exit_code": process.returncode})
                receipt["status"] = "completed"
    except BaseException as exc:
        receipt.update(status="failed", error_code=getattr(exc, "code", type(exc).__name__), error=str(exc))
        details = getattr(exc, "details", None)
        if details:
            receipt["error_details"] = details
        raise
    finally:
        # Termination includes descendants even if a Windows venv launcher has
        # already exited. Closing the last Job handle is also a safety barrier.
        if tree is not None:
            tree.terminate(process)
            tree.close()
        receipt.update(actual_exit_code=None if process is None else process.returncode,
                       elapsed_seconds=time.monotonic() - started, finished_at=time.time())
        (directory / "process-receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    return receipt
