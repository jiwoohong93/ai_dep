FROM condaforge/miniforge3:24.9.2-0

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_PROGRESS_BAR=off \
    CONDA_NO_PLUGINS=true \
    CONDA_NUMBER_CHANNEL_NOTICES=0 \
    CONDA_OVERRIDE_CUDA=11.8 \
    CONDA_REPODATA_THREADS=1 \
    CONDA_FETCH_THREADS=1 \
    CONDA_VERIFY_THREADS=1 \
    CONDA_EXECUTE_THREADS=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    XDG_CACHE_HOME=/tmp/.cache \
    PROJECT_ROOT=/app \
    DATA_DIR=/app/data/processed \
    MODEL_DIR=/app/models \
    OUTPUT_DIR=/app/outputs \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /app

COPY environment-linux-64.lock environment.yml requirements.txt requirements-docker.txt ./
RUN conda --no-plugins config --set solver classic \
    && conda --no-plugins config --set notify_outdated_conda false \
    && conda --no-plugins config --set repodata_threads 1 \
    && conda --no-plugins config --set fetch_threads 1 \
    && conda --no-plugins config --set verify_threads 1 \
    && conda --no-plugins config --set execute_threads 1 \
    && conda --no-plugins create --json -y -n smdart --file environment-linux-64.lock \
    && conda --no-plugins run -n smdart python -m pip install --no-cache-dir --progress-bar off -r requirements-docker.txt \
    && conda --no-plugins clean -afy

ENV PATH=/opt/conda/envs/smdart/bin:/opt/conda/bin:$PATH \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1

COPY src ./src
COPY models ./models
COPY data/processed ./data/processed
COPY README.md ./

RUN mkdir -p /app/outputs /app/data/raw /tmp/matplotlib /tmp/.cache \
    && chmod -R 777 /app/outputs /tmp/matplotlib /tmp/.cache

CMD ["python", "src/smdart/inference_test.py"]
