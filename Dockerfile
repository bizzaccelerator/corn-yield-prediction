FROM python:3.11.10-slim

# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    ca-certificates \
    libglib2.0-0 \
    libx11-6 \
    coreutils \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Miniconda
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    /opt/conda/bin/conda clean -afy

# Add conda to PATH
# ENV PATH="$CONDA_DIR/bin:$PATH"
ENV PATH="/opt/conda/bin:$PATH"
# Copy the environment.yml file
COPY ml-env.yml /tmp/ml-env.yml
# Create the Conda environment and activate it
RUN conda env create -f /tmp/ml-env.yml --verbose && \
    conda clean -afy
# Set the default path to include your Conda environment
ENV PATH="/opt/conda/envs/ml-env/bin:$PATH"

# Set the working directory
WORKDIR /app

# Copy the application code into the container
COPY ["predict.py", "model_Grid_GBT_learnig=0.1_depth=3.bin", "./"]

# Expose the application port
EXPOSE 9696

# Set the default command to run the application
CMD ["waitress-serve", "--listen=0.0.0.0:9696", "predict:app"]