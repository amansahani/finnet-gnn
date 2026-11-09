"""Small probabilistic counters used to bound online graph cardinality work."""

from __future__ import annotations

import hashlib
import math


def _hash_64(value: int | str | bytes) -> int:
    if isinstance(value, int):
        value = str(value).encode("ascii")
    elif isinstance(value, str):
        value = value.encode("utf-8", "ignore")
    return int.from_bytes(hashlib.blake2b(value, digest_size=8).digest(), "big")


class HyperLogLog:
    """Approximate distinct-count sketch with add-only semantics."""

    def __init__(self, precision: int = 6) -> None:
        if not 4 <= precision <= 16:
            raise ValueError("precision must be between 4 and 16")
        self.precision = precision
        self.num_registers = 1 << precision
        self.registers = [0] * self.num_registers
        self._base = Bias(self.precision, self.num_registers)

    def add(self, value: int | str | bytes) -> None:
        digest = _hash_64(value)
        index = digest & (self.num_registers - 1)
        remaining = digest >> self.precision
        width = 64 - self.precision
        if remaining == 0:
            rank = width + 1
        else:
            rank = width - remaining.bit_length() + 1
        self.registers[index] = max(self.registers[index], rank)

    def estimate(self) -> float:
        raw = self._base.alpha * self.num_registers**2 / sum(
            2.0 ** (-register) for register in self.registers
        )
        zeros = self.registers.count(0)
        if raw <= 2.5 * self.num_registers and zeros:
            raw = self.num_registers * math.log(self.num_registers / zeros)
        elif raw > (1 / 30) * (1 << 64):
            raw = -(1 << 64) * math.log(1.0 - raw / (1 << 64))
        return float(raw)

    def __len__(self) -> int:
        return round(self.estimate())

    def merge(self, other: HyperLogLog) -> HyperLogLog:
        if other.precision != self.precision:
            raise ValueError("cannot merge HyperLogLog sketches with different precision")
        self.registers = [max(a, b) for a, b in zip(self.registers, other.registers, strict=True)]
        return self


class Bias:
    def __init__(self, precision: int, num_registers: int) -> None:
        self.alpha = _alpha(num_registers)


def _alpha(num_registers: int) -> float:
    if num_registers == 16:
        return 0.673
    if num_registers == 32:
        return 0.697
    if num_registers == 64:
        return 0.709
    return 0.7213 / (1.0 + 1.079 / num_registers)


class CountMinSketch:
    """Frequency sketch with conservative update for hot transactions."""

    def __init__(self, width: int = 1024, depth: int = 4) -> None:
        if width < 8 or depth < 2:
            raise ValueError("width >= 8 and depth >= 2 required")
        self.width = int(width)
        self.depth = int(depth)
        self.table = [[0] * self.width for _ in range(self.depth)]
        self.seeds = [index + 1 for index in range(self.depth)]

    def add(self, value: int | str | bytes, amount: int = 1) -> None:
        for row, seed in zip(self.table, self.seeds, strict=True):
            index = _hash_with_seed(value, seed) % self.width
            row[index] += amount

    def estimate(self, value: int | str | bytes) -> int:
        estimates = []
        for row, seed in zip(self.table, self.seeds, strict=True):
            estimates.append(row[_hash_with_seed(value, seed) % self.width])
        return int(min(estimates))


def _hash_with_seed(value: int | str | bytes, seed: int) -> int:
    if isinstance(value, int):
        value = str(value).encode("ascii")
    elif isinstance(value, str):
        value = value.encode("utf-8", "ignore")
    return int.from_bytes(hashlib.blake2b(value, digest_size=8, key=seed.to_bytes(8, "big")).digest(), "big")


class SpaceSaving:
    """Bounded top-k frequency tracker."""

    def __init__(self, capacity: int = 128) -> None:
        self.capacity = int(capacity)
        self.counts: dict[int | str, int] = {}
        self.minimum = 0

    def add(self, value: int | str) -> None:
        if value in self.counts:
            self.counts[value] += 1
            self.minimum = min(self.minimum, self.counts[value])
            return
        if len(self.counts) < self.capacity:
            self.counts[value] = 1
            return
        victim = min(self.counts, key=self.counts.get)
        lower = self.counts.pop(victim)
        self.counts[value] = lower + 1
        self.minimum = min(self.counts.values())

    def top(self, limit: int = 10) -> list[tuple[int | str, int]]:
        return sorted(self.counts.items(), key=lambda item: item[1], reverse=True)[:limit]
