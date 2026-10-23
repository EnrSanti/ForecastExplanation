ARG PYTHON_IMAGE=python:3.14.7-slim-trixie

FROM ${PYTHON_IMAGE} AS builder
COPY --from=ghcr.io/astral-sh/uv:0.13.4 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /app
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --locked --no-dev --no-install-project

FROM ${PYTHON_IMAGE} AS runtime
LABEL authors="elius"

# libeccodes0: GRIB reading (cfgrib; the Linux eccodes wheel has no library)
# libexpat1: udunits2 of cf_units (tobac -> iris)
RUN apt-get update && \
    apt-get install -y --no-install-recommends libeccodes0 libexpat1 && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY src ./src

CMD ["python", "src/main.py"]
