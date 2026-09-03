FROM condaforge/miniforge3:latest
LABEL authors="elius"

WORKDIR /app

# Copy environment definition first for better Docker layer caching
COPY environment.yml .

# Create the conda environment
RUN --mount=type=cache,target=/opt/conda/pkgs conda env create -f environment.yml

# Make the environment the default for subsequent commands
ENV PATH=/opt/conda/envs/weather/bin:$PATH

# Copy application
COPY main.py .
COPY data_extraction ./data_extraction
COPY image_processing ./image_processing
COPY reasoning ./reasoning

# Run the application
CMD ["python", "main.py"]