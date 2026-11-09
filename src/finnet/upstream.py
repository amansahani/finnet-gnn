"""Optional upstream ISO-8583 relay for genuinely forwarding allowed traffic."""

from __future__ import annotations

import asyncio
import struct

from .config import ServerConfig


class UpstreamISOClient:
    def __init__(self, config: ServerConfig) -> None:
        self.config = config
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._lock = asyncio.Lock()

    @property
    def enabled(self) -> bool:
        return bool(self.config.upstream_host and self.config.upstream_port)

    async def exchange(self, payload: bytes) -> bytes | None:
        if not self.enabled:
            return None
        async with self._lock:
            try:
                await self._ensure_connected()
                assert self._reader is not None and self._writer is not None
                self._writer.write(struct.pack(">H", len(payload)) + payload)
                await self._writer.drain()
                header = await asyncio.wait_for(
                    self._reader.readexactly(2), timeout=self.config.upstream_timeout_s
                )
                (length,) = struct.unpack(">H", header)
                if length <= 0 or length > self.config.max_frame_bytes:
                    raise ValueError(f"invalid upstream frame length {length}")
                return await asyncio.wait_for(
                    self._reader.readexactly(length), timeout=self.config.upstream_timeout_s
                )
            except Exception:  # noqa: BLE001 - relay errors must not crash the gateway
                await self._drop()
                return None

    async def _ensure_connected(self) -> None:
        if self._writer is not None and not self._writer.is_closing():
            return
        self._reader, self._writer = await asyncio.wait_for(
            asyncio.open_connection(self.config.upstream_host, self.config.upstream_port),
            timeout=self.config.upstream_timeout_s,
        )

    async def _drop(self) -> None:
        if self._writer is not None:
            self._writer.close()
            try:
                await self._writer.wait_closed()
            except ConnectionError:
                pass
        self._reader = None
        self._writer = None

    async def close(self) -> None:
        async with self._lock:
            await self._drop()
