# NVIDIA CUDA image as a base
FROM nvidia/cuda:11.3-runtime AS tcp-base
# Install Python and its tools
RUN apt update && apt install -y --no-install-recommends \
    git \
    build-essential \
    python3-dev \
    python3-pip \
    python3-setuptools
RUN pip3 -q install pip --upgrade
WORKDIR /app
COPY environment.yml .
RUN pip install -r environment.yml
ENV PYTHONPATH $PYTHONPATH:/app/home/e17072/Documents/TCP_Refactored/TCP
COPY . .


CMD ["python", "TCP/train.py", "--gpus", "all"]
