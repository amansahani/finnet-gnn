FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build
COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch==2.15.0+cpu" \
    && /opt/venv/bin/pip install --no-cache-dir .

FROM python:3.12-slim AS runtime

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN groupadd --system finnet \
    && useradd --system --gid finnet --home /app --no-create-home finnet

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY configs ./configs

RUN mkdir -p /app/artifacts \
    && chown -R finnet:finnet /app/artifacts /app/configs

USER finnet
VOLUME ["/app/artifacts"]
EXPOSE 8583 9101

HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:9101/healthz', timeout=2).read()"

ENTRYPOINT ["finnet"]
CMD ["serve", "--config", "configs/default.yaml", "--metrics-port", "9101"]
