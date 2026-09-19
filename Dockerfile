FROM condaforge/miniforge3:latest AS builder

WORKDIR /app
COPY environment.yaml .

RUN --mount=type=cache,target=/opt/conda/pkgs \
    conda config --add pkgs_dirs /opt/conda/pkgs && \
    conda env create -f environment.yaml

FROM debian:trixie-slim AS runtime
LABEL authors="elius"

COPY --from=builder /opt/conda/envs/weather /opt/conda/envs/weather
ENV PATH=/opt/conda/envs/weather/bin:$PATH

WORKDIR /app
COPY src ./src

CMD ["python", "src/main.py"]
