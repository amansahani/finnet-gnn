"""Asynchronous L4 ISO-8583 gateway server."""

from __future__ import annotations

import asyncio
import contextlib
import struct

from .config import AppConfig
from .engine import EngineResult, GatewayEngine
from .iso8583 import ISO8583Codec, ISO8583Error
from .metrics import start_metrics_server
from .security import build_tls_context
from .upstream import UpstreamISOClient


class GatewayServer:
    def __init__(self, config: AppConfig, engine: GatewayEngine | None = None) -> None:
        self.config = config
        self.engine = engine or GatewayEngine(config)
        self.codec = ISO8583Codec()
        self.upstream = UpstreamISOClient(config.server)
        self._server: asyncio.AbstractServer | None = None
        self._metrics_server: asyncio.AbstractServer | None = None
        self._slots = asyncio.Semaphore(config.server.max_connections)
        if config.server.tls_enabled:
            self._tls_context = build_tls_context(
                cert_file=config.server.tls_cert_file,
                key_file=config.server.tls_key_file,
                ca_file=config.server.tls_ca_file,
                require_client_certificate=config.server.require_client_certificate,
            )
        else:
            self._tls_context = None
        if config.server.mac_required:
            from .security import load_mac_secret

            self._mac_secret = load_mac_secret(
                mac_secret=config.server.mac_secret,
                mac_secret_file=config.server.mac_secret_file,
            )
        else:
            self._mac_secret = None
        self._active = 0
        self._stop_event = asyncio.Event()
        self._closed = False

    @property
    def port(self) -> int:
        if self._server is None or not self._server.sockets:
            return self.config.server.port
        return int(self._server.sockets[0].getsockname()[1])

    @property
    def metrics_port(self) -> int:
        if self._metrics_server is None or not self._metrics_server.sockets:
            return self.config.server.metrics_port
        return int(self._metrics_server.sockets[0].getsockname()[1])

    async def start(self) -> None:
        self._server = await asyncio.start_server(
            self._handle_client,
            self.config.server.host,
            self.config.server.port,
            backlog=512,
            ssl=self._tls_context,
        )
        self._metrics_server = await start_metrics_server(
            self.engine.metrics,
            self.config.server.metrics_host,
            self.config.server.metrics_port,
            extra_status=self.engine.status,
        )
        self.engine.metrics.gauge("finnet_active_connections", 0)

    async def serve_forever(self) -> None:
        if self._server is None:
            await self.start()
        await self._stop_event.wait()

    def request_stop(self) -> None:
        self._stop_event.set()

    async def stop(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
        if self._metrics_server is not None:
            self._metrics_server.close()
            await self._metrics_server.wait_closed()
        await self.upstream.close()
        await self.engine.aclose()

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        async with self._slots:
            self._active += 1
            self.engine.metrics.inc("finnet_connections_total")
            self.engine.metrics.gauge("finnet_active_connections", self._active)
            peer = writer.get_extra_info("peername")
            peer_ip = str(peer[0]) if peer else "0.0.0.0"
            if "%" in peer_ip:
                peer_ip = peer_ip.split("%", 1)[0]
            try:
                blocked = self.engine.precheck_connection(peer_ip)
                if blocked is not None:
                    await self._close(writer)
                    return
                await self._message_loop(reader, writer, peer_ip)
            except (asyncio.TimeoutError, ConnectionError, BrokenPipeError):
                pass
            except Exception:  # noqa: BLE001 - isolate one bad socket
                self.engine.metrics.inc("finnet_errors_total", stage="connection")
            finally:
                self._active -= 1
                self.engine.metrics.gauge("finnet_active_connections", self._active)
                await self._close(writer)

    async def _message_loop(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        peer_ip: str,
    ) -> None:
        for _ in range(self.config.server.max_messages_per_connection):
            try:
                header = await asyncio.wait_for(
                    reader.readexactly(2), timeout=self.config.server.idle_timeout_s
                )
            except asyncio.IncompleteReadError:
                return
            (length,) = struct.unpack(">H", header)
            if length <= 0 or length > self.config.server.max_frame_bytes:
                self.engine.metrics.inc("finnet_protocol_errors_total")
                return
            try:
                payload = await asyncio.wait_for(
                    reader.readexactly(length), timeout=self.config.server.idle_timeout_s
                )
                if self._mac_secret is not None:
                    from .security import verify_mac

                    mac = verify_mac(payload, self._mac_secret)
                    if not mac.valid:
                        self.engine.metrics.inc("finnet_mac_failures_total")
                        self.engine.metrics.inc("finnet_protocol_errors_total")
                        return
                message = self.codec.decode(payload)
            except (asyncio.IncompleteReadError, asyncio.TimeoutError):
                return
            except ISO8583Error:
                self.engine.metrics.inc("finnet_protocol_errors_total")
                return

            try:
                result = await asyncio.wait_for(
                    self.engine.process(message, peer_ip=peer_ip, raw_payload=payload),
                    timeout=self.config.server.request_timeout_s,
                )
            except asyncio.TimeoutError:
                self.engine.metrics.inc("finnet_errors_total", stage="deadline")
                self.engine.metrics.inc("finnet_actions_total", action="TERMINATE")
                return

            if not result.send_response:
                return
            if result.delay_ms > 0:
                await asyncio.sleep(result.delay_ms / 1000.0)

            response_bytes = await self._response_bytes(result, message, payload)
            writer.write(struct.pack(">H", len(response_bytes)) + response_bytes)
            await writer.drain()
            if result.close_socket:
                return

    async def _response_bytes(
        self,
        result: EngineResult,
        message,
        payload: bytes,
    ) -> bytes:
        if result.action.value == "ALLOW" and self.upstream.enabled:
            relayed = await self.upstream.exchange(payload)
            if relayed is not None:
                return relayed
        response = self.codec.make_response(
            message,
            result.response_code,
            action=result.action.value,
            score=result.score,
            reason=result.reason,
        )
        return self.codec.encode(response)

    async def _close(self, writer: asyncio.StreamWriter) -> None:
        writer.close()
        with contextlib.suppress(ConnectionError, BrokenPipeError):
            await writer.wait_closed()
