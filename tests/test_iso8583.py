from __future__ import annotations

import pytest

from finnet.iso8583 import (
    ISO8583Codec,
    ISO8583Error,
    ISO8583Message,
    client_network_claim,
    subnet_for_ip,
    tokenize_pan,
)


@pytest.fixture()
def codec() -> ISO8583Codec:
    return ISO8583Codec()


def sample_message() -> ISO8583Message:
    return ISO8583Message(
        mti="0200",
        fields={
            2: "4111111111111111",
            3: "000000",
            4: "000000001000",
            7: "1110123456",
            11: "123456",
            22: "051",
            41: "T0000001",
            42: "M00000000000001",
            49: "356",
            52: b"\x01\x02\x03\x04\x05\x06\x07\x08",
            55: "9F2608AABBCCDD",
            62: "ts_ms=1700000000000",
        },
    )


def test_roundtrip_primary_secondary_and_binary(codec: ISO8583Codec) -> None:
    payload = codec.encode(sample_message())
    decoded = codec.decode(payload)
    assert decoded.mti == "0200"
    assert decoded.pan == "4111111111111111"
    assert decoded.masked_pan == "411111******1111"
    assert str(decoded.amount) == "10"
    assert decoded.fields[52] == b"\x01\x02\x03\x04\x05\x06\x07\x08"
    assert decoded.fields[55] == "9F2608AABBCCDD"
    assert decoded.client_timestamp_ms == 1700000000000


def test_framing(codec: ISO8583Codec) -> None:
    frame = codec.encode_frame(sample_message())
    assert int.from_bytes(frame[:2], "big") == len(frame) - 2
    decoded = codec.decode_frame(frame)
    assert decoded.stan == "123456"


def test_response_construction(codec: ISO8583Codec) -> None:
    request = sample_message()
    response = codec.make_response(
        request, "00", action="ALLOW", score=0.1, reason="ok"
    )
    assert response.mti == "0210"
    assert response.response_code == "00"
    assert response.fields[2] == request.fields[2]
    assert "action=ALLOW" in str(response.fields[60])
    codec.decode(codec.encode(response))


def test_fixed_length_validation(codec: ISO8583Codec) -> None:
    with pytest.raises(ISO8583Error):
        codec.encode("0200", {11: "123"})


def test_trailing_bytes_rejected(codec: ISO8583Codec) -> None:
    payload = codec.encode(sample_message())
    with pytest.raises(ISO8583Error, match="trailing"):
        codec.decode(payload + b"x")


def test_tokenization_is_stable_and_not_reversible() -> None:
    token = tokenize_pan("4111111111111111", "salt")
    assert token == tokenize_pan("4111111111111111", "salt")
    assert token != tokenize_pan("4111111111111111", "other")
    assert "411111" not in token


def test_subnet_mapping() -> None:
    assert subnet_for_ip("192.0.2.44") == "192.0.2.0/24"
    assert subnet_for_ip("2001:db8::1") == "2001:db8::/64"
    assert subnet_for_ip("not-an-ip") == "0.0.0.0/24"


def test_client_network_claim() -> None:
    message = ISO8583Message(
        "0200", {61: "subnet=127.100.1.0/24;src_ip=127.100.1.22"}
    )
    src_ip, subnet = client_network_claim(message)
    assert src_ip == "127.100.1.22"
    assert subnet == "127.100.1.0/24"


def test_codec_random_bytes_do_not_crash_uncontrolled() -> None:
    import random

    random_codec = ISO8583Codec()
    rng = random.Random(1234)
    for _ in range(1_000):
        payload = bytes(rng.randrange(256) for _ in range(rng.randrange(0, 256)))
        try:
            random_codec.decode(payload)
        except ISO8583Error:
            pass
