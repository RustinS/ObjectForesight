from __future__ import annotations

import gc
from collections import OrderedDict


class _LRU(OrderedDict):
    def __init__(self, cap: int = 128, gc_on_evict: bool = False) -> None:
        super().__init__()
        self.cap = max(1, cap)
        self.gc_on_evict = gc_on_evict

    def get(self, k, default=None):
        if k in self:
            self.move_to_end(k)
            return self[k]
        return default

    def put(self, k, v):
        if k in self:
            self.move_to_end(k)
        self[k] = v
        if len(self) > self.cap:
            self.popitem(last=False)
            if self.gc_on_evict:
                gc.collect()

    def __getstate__(self):
        return {"cap": self.cap, "gc_on_evict": self.gc_on_evict}

    def __setstate__(self, state):
        self.cap = state.get("cap", 128) if isinstance(state, dict) else 128
        self.gc_on_evict = state.get("gc_on_evict", False) if isinstance(state, dict) else False
