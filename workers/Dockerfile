# Start with PyTorch (includes CUDA support)
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Install system dependencies
# torchxrayvision often uses OpenCV/scikit-image which needs libgl
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

# Install Python libraries
# ADDED: torchxrayvision and torchvision here
RUN pip install "dask[complete]" distributed minio pymongo scikit-learn pandas pillow torchxrayvision torchvision

# Set the working directory inside the container
WORKDIR /app

# COPY THE MODEL
# Source: ./model -> Dest: /app/model
COPY models /app/models

# COPY THE WORKER LOGIC
# Source: ./workers/worker_logic.py -> Dest: /app/worker_logic.py
COPY workers/worker_logic.py /app/worker_logic.py

# Add current directory to python path
ENV PYTHONPATH="${PYTHONPATH}:/app"

# Start the worker
CMD ["dask-worker", "tcp://scheduler:8786", "--nthreads", "1", "--memory-limit", "4GB"]