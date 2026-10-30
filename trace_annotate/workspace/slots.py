"""One process-wide physical prepare slot, cancellable and observable (P04/P06).

The workspace performs at most one ffmpeg decode at a time. Every entry point
(synchronous CLI import, synchronous HTTP import, the async drainer and any
manual ``run_pending`` caller) funnels through this single object so that the
limit is a property of the process rather than of one call path.

Two properties matter for correctness and are enforced here:

* **Re-entrancy.** The async worker takes the slot *before* it claims an
  intake, so a queued intake never burns its lease while waiting. The media
  layer then takes the same slot again while it actually decodes. The second
  acquisition must not deadlock, so ownership is tracked per thread.
* **Cancellable waiting.** A waiter must be able to give up when the workspace
  is closing or when its own deadline passes, and must never spin on an
  unbounded blocking lock. Every wait iteration re-checks the supplied
  predicates, so a shutdown cannot be blocked by a slot that never frees.

The slot also records peak concurrency and wait durations. Those counters are
diagnostic evidence, not a substitute for observing real child processes.
"""
from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Optional

# A short poll keeps cancel/close latency well below any human-visible budget
# while adding negligible CPU: the wait is a condition wait, not a spin.
DEFAULT_POLL_SECONDS = 0.02


class SlotTimeout(Exception):
    """Raised when a slot could not be acquired inside its budget."""

    def __init__(self, message: str, *, waited_seconds: float):
        super().__init__(message)
        self.waited_seconds = waited_seconds


class SlotClosed(Exception):
    """Raised when the owning workspace started closing during a wait."""


class PrepareSlot:
    """A re-entrant, cancellable, single-capacity prepare slot."""

    def __init__(self, *, poll_seconds: float = DEFAULT_POLL_SECONDS):
        if not poll_seconds > 0:
            raise ValueError("poll_seconds must be positive")
        self._poll = float(poll_seconds)
        self._condition = threading.Condition(threading.Lock())
        self._owner: Optional[int] = None
        self._depth = 0
        self._active = 0
        self._peak = 0
        self._acquisitions = 0
        self._waits = 0
        self._total_wait_seconds = 0.0
        self._max_wait_seconds = 0.0
        self._last_owner_released_at: Optional[float] = None

    # -- introspection -----------------------------------------------------
    @property
    def depth(self) -> int:
        """Nesting depth for the current thread (0 when it holds nothing)."""
        with self._condition:
            return self._depth if self._owner == threading.get_ident() else 0

    def stats(self) -> Dict[str, Any]:
        with self._condition:
            return {
                "capacity": 1,
                "active": self._active,
                "peak": self._peak,
                "acquisitions": self._acquisitions,
                "waits": self._waits,
                "total_wait_seconds": round(self._total_wait_seconds, 6),
                "max_wait_seconds": round(self._max_wait_seconds, 6),
                "held": self._owner is not None,
                "last_released_at": self._last_owner_released_at,
            }

    def reset_stats(self) -> None:
        with self._condition:
            self._peak = self._active
            self._acquisitions = 0
            self._waits = 0
            self._total_wait_seconds = 0.0
            self._max_wait_seconds = 0.0

    # -- acquisition -------------------------------------------------------
    def acquire(self, *, cancelled: Optional[Callable[[], bool]] = None,
                closing: Optional[Callable[[], bool]] = None,
                timeout: Optional[float] = None,
                clock: Callable[[], float] = time.monotonic) -> float:
        """Take the slot; returns the seconds spent waiting.

        ``cancelled`` and ``closing`` are re-evaluated on every poll. A truthy
        ``closing`` raises :class:`SlotClosed`; a truthy ``cancelled`` raises
        ``InterruptedError`` so callers can map it onto their own cancel code.
        ``timeout`` is a wait budget in seconds, never a total operation budget.
        """
        if cancelled is not None and not callable(cancelled):
            raise TypeError("cancelled must be callable or None")
        if closing is not None and not callable(closing):
            raise TypeError("closing must be callable or None")
        if timeout is not None and timeout < 0:
            raise ValueError("timeout must not be negative")

        me = threading.get_ident()
        started = clock()
        with self._condition:
            if self._owner == me:
                # Re-entrant: the worker already reserved the slot for this
                # intake, so the media layer must not wait on itself.
                self._depth += 1
                return 0.0
            waited = 0.0
            while self._owner is not None:
                if closing is not None and closing():
                    raise SlotClosed("workspace is closing")
                if cancelled is not None and cancelled():
                    raise InterruptedError("slot wait was cancelled")
                if timeout is not None:
                    elapsed = clock() - started
                    if elapsed >= timeout:
                        self._waits += 1
                        self._total_wait_seconds += elapsed
                        self._max_wait_seconds = max(self._max_wait_seconds, elapsed)
                        raise SlotTimeout("prepare slot wait budget exhausted",
                                          waited_seconds=elapsed)
                # Condition wait is interruptible and never busy-spins.
                self._condition.wait(self._poll)
                waited = clock() - started
            if closing is not None and closing():
                raise SlotClosed("workspace is closing")
            self._owner = me
            self._depth = 1
            self._active = 1
            self._acquisitions += 1
            self._peak = max(self._peak, self._active)
            if waited > 0:
                self._waits += 1
                self._total_wait_seconds += waited
                self._max_wait_seconds = max(self._max_wait_seconds, waited)
            return waited

    def release(self) -> None:
        me = threading.get_ident()
        with self._condition:
            if self._owner != me:
                raise RuntimeError("prepare slot released by a non-owner thread")
            self._depth -= 1
            if self._depth > 0:
                return
            self._owner = None
            self._active = 0
            self._last_owner_released_at = time.time()
            self._condition.notify_all()

    @contextmanager
    def hold(self, *, cancelled: Optional[Callable[[], bool]] = None,
             closing: Optional[Callable[[], bool]] = None,
             timeout: Optional[float] = None,
             clock: Callable[[], float] = time.monotonic) -> Iterator[float]:
        waited = self.acquire(cancelled=cancelled, closing=closing, timeout=timeout, clock=clock)
        try:
            yield waited
        finally:
            self.release()


# One slot per workspace root, not one per interpreter. Two independent
# workspaces opened in the same process must be able to prepare concurrently;
# only a shared workspace is serialised. The key is the resolved absolute root
# so the same workspace always maps to the same slot.
_SLOTS: Dict[str, PrepareSlot] = {}
_SLOTS_LOCK = threading.Lock()
_PROCESS_KEY = "<process>"


def get_prepare_slot(workspace_root: Any = None) -> PrepareSlot:
    """Return the single prepare slot for a workspace root."""
    if workspace_root is None:
        key = _PROCESS_KEY
    else:
        key = str(Path(workspace_root).absolute())
        if os.name == "nt":
            key = key.lower()
    with _SLOTS_LOCK:
        slot = _SLOTS.get(key)
        if slot is None:
            slot = PrepareSlot()
            _SLOTS[key] = slot
        return slot


def reset_prepare_slots() -> None:
    """Drop the registry (tests only; never called by product code)."""
    with _SLOTS_LOCK:
        _SLOTS.clear()
