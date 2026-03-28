"""Thread-safe token usage tracker for RLM runs.

The :class:`UsageTracker` accumulates ``pydantic_ai.usage.RunUsage`` from every
LLM sub-call (action agent, extract agent, ``llm_query`` / ``llm_query_batched``)
into a single total.  Because ``llm_query_batched`` dispatches work to a
:class:`~concurrent.futures.ThreadPoolExecutor` **and** the main RLM loop is
async, the tracker uses a :class:`threading.Lock` to protect concurrent writes.

A ``threading.Lock`` (rather than ``asyncio.Lock``) is chosen deliberately:
the lock is only held for the brief duration of ``RunUsage.incr()`` (integer
additions), so it never meaningfully blocks the event loop, and it is the only
primitive that also protects against concurrent OS threads.
"""

from __future__ import annotations

import threading
from copy import copy

from pydantic_ai.usage import RequestUsage, RunUsage


class UsageTracker:
    """Thread-safe and async-safe accumulator for pydantic-ai ``RunUsage``.

    All ``incr`` calls are serialised behind a single ``threading.Lock`` so the
    tracker is safe to use from:

    * the main ``asyncio`` event loop (action / extract agent calls),
    * ``ThreadPoolExecutor`` threads (``llm_query_batched`` sub-calls).

    Example:
        >>> tracker = UsageTracker()
        >>> tracker.incr(RunUsage(requests=1, input_tokens=100, output_tokens=50))
        >>> tracker.usage.total_tokens
        150
    """

    def __init__(self) -> None:
        self._usage = RunUsage()
        self._lock = threading.Lock()

    # -- mutation -----------------------------------------------------------

    def incr(self, usage: RunUsage | RequestUsage) -> None:
        """Increment accumulated usage in a thread-safe manner.

        Args:
            usage: The usage delta to add.  Accepts both ``RunUsage``
                (from ``agent.run().usage()``) and ``RequestUsage``.
        """
        with self._lock:
            self._usage.incr(usage)

    # -- read ---------------------------------------------------------------

    @property
    def usage(self) -> RunUsage:
        """Return a snapshot of the current accumulated usage.

        The returned object is a **copy** so callers cannot accidentally
        mutate internal state.
        """
        with self._lock:
            return copy(self._usage)

    # -- reset --------------------------------------------------------------

    def reset(self) -> None:
        """Reset all counters to zero.

        Useful when reusing the same :class:`MontyRLM` instance for
        multiple ``run()`` calls and you want per-run usage.
        """
        with self._lock:
            self._usage = RunUsage()
