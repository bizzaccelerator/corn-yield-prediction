FROM continuumio/miniconda3

# Copy the environment.yml file into the container
COPY environment.yml /tmp/environment.yml
# Create the Conda environment and activate it
RUN conda env create -f /tmp/environment.yml && conda clean -afy
# Make the environment the default
ENV PATH /opt/conda/envs/ml-zoomcamp/bin:$PATH

# Set the working directory
WORKDIR /app

# Copy the application code into the container
COPY ["predict.py", "model_Grid_GBT_learnig=0.1_depth=3.bin", "./"]

# Expose the application port
EXPOSE 9696

# Set the default command to run the application
CMD ["waitress-serve", "--listen=0.0.0.0:9696", "predict:app"]