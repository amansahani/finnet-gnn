from __future__ import annotations

import asyncio
import struct
import time
from dataclasses import replace

from finnet.config import load_config
from finnet.engine import GatewayEngine
from finnet.iso8583 import ISO8583Codec, ISO8583Message
from finnet.server import GatewayServer


def base_config(tmp_path):
    config = load_config("configs/default.yaml")
    return replace(
        config,
        server=replace(
            config.server,
            port=0,
            metrics_port=0,
            use_uvloop=False,
            trust_client_subnet=True,
        ),
        model=replace(config.model, checkpoint=str(tmp_path / "missing.pt")),
        policy=replace(
            config.policy,
            allow_threshold=0.99,
            throttle_threshold=0.995,
            quarantine_threshold=0.998,
            terminate_threshold=0.999,
            firewall_audit_path=str(tmp_path / "firewall.jsonl"),
        ),
        log_path=str(tmp_path / "decisions.jsonl"),
    )


def make_message(index: int = 1) -> ISO8583Message:
    return ISO8583Message(
        "0200",
        {
            2: "4111111111111111",
            3: "000000",
            4: "000000001000",
            7: time.strftime("%m%d%H%M%S"),
            11: f"{index:06d}",
            22: "051",
            41: "T0000001",
            42: "M00000000000001",
            49: "356",
            61: "subnet=127.50.0.0/24;src_ip=127.50.0.10",
            62: f"ts_ms={time.time() * 1000:.3f}",
        },
    )


async def test_engine_processes_message(tmp_path) -> None:
    config = base_config(tmp_path)
    engine = GatewayEngine(config)
    codec = ISO8583Codec()
    message = make_message()
    result = await engine.process(
        message,
        peer_ip="127.0.0.1",
        raw_payload=codec.encode(message),
    )
    assert result.action.value == "ALLOW"
    assert result.features is not None
    assert len(result.features) == 24
    assert result.tx_node_id is not None
    assert engine.graph.stats()["nodes"] >= 5
    await engine.aclose()


async def test_server_roundtrip(tmp_path) -> None:
    config = base_config(tmp_path)
    server = GatewayServer(config)
    await server.start()
    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", server.port)
        codec = ISO8583Codec()
        writer.write(codec.encode_frame(make_message()))
        await writer.drain()
        header = await asyncio.wait_for(reader.readexactly(2), timeout=2)
        (length,) = struct.unpack(">H", header)
        payload = await asyncio.wait_for(reader.readexactly(length), timeout=2)
        response = codec.decode(payload)
        assert response.mti == "0210"
        assert response.response_code == "00"
        assert "action=ALLOW" in str(response.fields[60])
        writer.close()
        await writer.wait_closed()
        assert server.engine.metrics.counter_value("finnet_messages_total") >= 1
    finally:
        await server.stop()
