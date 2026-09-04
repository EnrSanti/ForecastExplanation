FROM condaforge/miniforge3:latest
LABEL authors="elius"

WORKDIR /app
COPY environment.yml .

RUN --mount=type=cache,target=/opt/conda/pkgs \
    conda config --add pkgs_dirs /opt/conda/pkgs && \
    conda env create -f environment.yml
ENV PATH=/opt/conda/envs/weather/bin:$PATH

COPY src ./src

# Run the application
CMD ["python", "src/main.py"]