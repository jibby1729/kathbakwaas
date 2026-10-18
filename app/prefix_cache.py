"""Bounded full-prompt snapshots. Partial prefixes are deliberately never reused."""
from collections import OrderedDict
from copy import deepcopy
from time import monotonic


class PromptCache:
    def __init__(self, capacity=4, ttl=120):
        self.entries = OrderedDict()
        self.capacity = capacity
        self.ttl = ttl

    def get(self, session):
        now = monotonic()
        for key in list(self.entries):
            if now - self.entries[key][2] > self.ttl:
                del self.entries[key]
        entry = self.entries.get(session)
        if entry:
            self.entries.move_to_end(session)
            return entry[:2]
        return None

    def put(self, session, ids, cache):
        self.entries[session] = (tuple(ids), cache, monotonic())
        self.entries.move_to_end(session)
        while len(self.entries) > self.capacity:
            self.entries.popitem(last=False)

    def discard(self, session):
        self.entries.pop(session, None)

    def fork(self, session, ids):
        """Reuse only the complete same prompt; isolate the stored snapshot."""
        entry = self.get(session)
        if not entry:
            return None
        previous_ids, output = entry
        if tuple(ids) != previous_ids:
            return None
        return deepcopy(output)
