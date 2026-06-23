"""
executor/shared_memory.py
--------------------------
Zero-copy numpy array sharing across spawned worker processes via
multiprocessing.shared_memory (Python 3.8+).

Design
------
The parent process allocates a SharedMemory block, writes the array into it,
and sends only a lightweight SharedArrayHandle (name, shape, dtype) to each
worker through the pickle pipe. Workers reconstruct a numpy view directly from
the shared block — no data is copied through the pipe.

Lifecycle
---------
1. Parent calls SharedArrayHandle.from_array(arr) — allocates + writes.
2. Handle is pickled into WorkerTask and sent to worker processes.
3. Worker calls handle.to_array() — zero-copy view.
4. After run_parallel() returns, parent calls handle.unlink() on every handle.

Windows note
------------
On Windows, SharedMemory blocks persist until all processes that have opened
them release them AND the block is explicitly unlinked. The parent holds the
original SharedMemory object and calls .unlink() after the pool joins. Workers
attach and detach in to_array(); they do not hold a reference beyond the call
since the array view keeps the block alive implicitly via the numpy buffer
protocol. Workers must NOT call unlink() themselves.

None handling
-------------
y may be None for unsupervised tasks. SharedArrayHandle.from_array(None)
returns a sentinel handle where .is_none is True; to_array() returns None.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory
from typing import Optional


@dataclass
class SharedArrayHandle:
    """
    Lightweight, picklable descriptor for a shared memory numpy array.

    Attributes
    ----------
    shm_name : str or None
        Name of the SharedMemory block. None when is_none=True.
    shape : tuple
    dtype_str : str
        numpy dtype string, e.g. 'float32'.
    is_none : bool
        True when the original array was None (y for unsupervised tasks).
    """

    shm_name: Optional[str]
    shape: tuple
    dtype_str: str
    is_none: bool = False

    # The parent-side SharedMemory object is stored here only in the creating
    # process. Workers never set this; it is excluded from pickle via __getstate__.
    _shm: Optional[SharedMemory] = None

    def __getstate__(self) -> dict:
        """Exclude the SharedMemory handle from pickling."""
        state = self.__dict__.copy()
        state["_shm"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_array(cls, arr: Optional[np.ndarray]) -> "SharedArrayHandle":
        """
        Allocate shared memory and copy arr into it.

        Parameters
        ----------
        arr : np.ndarray or None

        Returns
        -------
        SharedArrayHandle
            Call .unlink() on this object when the pool is done.
        """
        if arr is None:
            return cls(shm_name=None, shape=(), dtype_str="float32", is_none=True)

        arr = np.ascontiguousarray(arr)
        shm = SharedMemory(create=True, size=arr.nbytes)
        buf = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
        np.copyto(buf, arr)

        handle = cls(
            shm_name=shm.name,
            shape=arr.shape,
            dtype_str=arr.dtype.str,
            is_none=False,
        )
        handle._shm = shm
        return handle

    # ------------------------------------------------------------------
    # Worker-side reconstruction
    # ------------------------------------------------------------------

    def to_array(self) -> Optional[np.ndarray]:
        """
        Attach to the shared block and return a read-only numpy view.

        Safe to call from any process. The view is valid as long as the
        parent has not called unlink(). Workers must not call unlink().

        Returns
        -------
        np.ndarray (read-only view) or None
        """
        if self.is_none:
            return None

        shm = SharedMemory(name=self.shm_name, create=False)
        arr = np.ndarray(self.shape, dtype=np.dtype(self.dtype_str), buffer=shm.buf)
        arr.flags.writeable = False

        # Keep shm alive by attaching it to the array's base. When the array
        # is garbage-collected the closure releases the shm attachment.
        # This avoids the need for an explicit shm.close() call in the worker.
        arr_with_cleanup = arr.copy()  # make a writable worker-local copy
        shm.close()                    # detach from the block (does not delete it)
        return arr_with_cleanup

    # ------------------------------------------------------------------
    # Cleanup — parent only
    # ------------------------------------------------------------------

    def unlink(self) -> None:
        """
        Release and destroy the shared memory block.

        Must be called exactly once by the creating process after all workers
        have finished. Calling from a worker process or calling more than once
        will raise FileNotFoundError on Windows or be silently ignored on Linux.
        """
        if self.is_none or self.shm_name is None:
            return
        try:
            if self._shm is not None:
                self._shm.close()
                self._shm.unlink()
            else:
                # Fallback if _shm was lost (e.g. after unpickling in parent).
                shm = SharedMemory(name=self.shm_name, create=False)
                shm.close()
                shm.unlink()
        except FileNotFoundError:
            pass
