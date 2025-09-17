# ========================================
# Dockerfile for HAR (MHEALTH, PAMAP2, UCI-HAR)
# ========================================
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget unzip curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create workspace
WORKDIR /workspace

# Install Python packages
RUN pip install --no-cache-dir \
    torchvision==0.17.2 \
    torchaudio==2.2.2 \
    scikit-learn \
    pandas \
    numpy \
    matplotlib \
    seaborn \
    jupyterlab \
    scipy

# Download datasets
RUN mkdir -p /workspace/data && cd /workspace/data && \
    # UCI-HAR
    wget -q https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip -O UCI-HAR.zip && \
    unzip -q UCI-HAR.zip && rm UCI-HAR.zip && \
    # PAMAP2
    wget -q https://archive.ics.uci.edu/ml/machine-learning-databases/00231/PAMAP2_Dataset.zip -O PAMAP2.zip && \
    unzip -q PAMAP2.zip && rm PAMAP2.zip && \
    # MHEALTH
    wget -q https://archive.ics.uci.edu/ml/machine-learning-databases/00319/MHEALTHDATASET.zip -O MHEALTH.zip && \
    unzip -q MHEALTH.zip && rm MHEALTH.zip

# Default command
CMD ["bash"]
