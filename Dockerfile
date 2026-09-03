FROM condaforge/miniforge3:latest
LABEL authors="elius"

WORKDIR /app
COPY environment.yml .

RUN --mount=type=cache,target=/opt/conda/pkgs conda env create -f environment.yml
ENV PATH=/opt/conda/envs/weather/bin:$PATH

COPY main.py .
COPY region.py .
COPY data_extraction ./data_extraction
COPY image_processing ./image_processing
COPY reasoning ./reasoning

# Run the application
CMD ["python", "main.py"]