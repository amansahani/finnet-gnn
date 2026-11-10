"""Bounded incremental connected-component tracking for entity fraud rings."""

from __future__ import annotations

import threading


class UnionFind:
    """Union-find for the static-ish analyst view of active entity links.

    Removes nodes approximately by decrementing the node's active count. It
    cannot recompute split components without a full graph scan, so feature
    consumers should treat these values as community-scale signals rather than
    exact current connectivity.
    """

    def __init__(self) -> None:
        self._parent: dict[int, int] = {}
        self._size: dict[int, int] = {}
        self._active: dict[int, int] = {}
        self._lock = threading.RLock()

    def add(self, node_id: int) -> None:
        with self._lock:
            if node_id not in self._parent:
                self._parent[node_id] = node_id
                self._size[node_id] = 1
                self._active[node_id] = 1

    def find(self, node_id: int) -> int:
        with self._lock:
            root = node_id
            parent = self._parent
            while parent.get(root) != root:
                root = parent.get(root, root)
            cursor = node_id
            while parent.get(cursor) != root:
                parent[cursor], cursor = root, parent[cursor]
            return root

    def union(self, a: int, b: int) -> int:
        with self._lock:
            self.add(a)
            self.add(b)
            root_a = self.find(a)
            root_b = self.find(b)
            if root_a == root_b:
                return root_a
            if self._active.get(root_a, 1) < self._active.get(root_b, 1):
                root_a, root_b = root_b, root_a
            self._parent[root_b] = root_a
            self._size[root_a] = self._size[root_a] + self._size[root_b]
            self._active[root_a] = self._active.get(root_a, 0) + self._active.get(root_b, 0)
            self._size.pop(root_b, None)
            self._active.pop(root_b, None)
            return root_a

    def remove(self, node_id: int) -> None:
        with self._lock:
            if node_id not in self._parent:
                return
            root = self.find(node_id)
            if self._active.get(root, 0) > 0:
                self._active[root] -= 1
            self._parent.pop(node_id, None)

    def component_size(self, node_id: int) -> int:
        with self._lock:
            root = self.find(node_id)
            return self._active.get(root, 0)

    def component_count(self) -> int:
        with self._lock:
            return sum(1 for node, parent in self._parent.items() if parent == node)

    def stats(self) -> dict[str, int]:
        with self._lock:
            roots = {self.find(node_id) for node_id in self._parent}
            largest = max((self._active.get(root, 0) for root in roots), default=0)
            return {
                "communities": len(roots),
                "largest_component_active": largest,
                "tracked_entities": len(self._parent),
            }
