FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubi8


RUN ["/bin/bash", "-c", "echo I am now using bash!"]
SHELL ["/bin/bash", "-c"]

SHELL ["apt-get", "install", "-y", "wget"]
SHELL ["wget", "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh"]

SHELL ["bash", "Miniconda3-latest-Linux-x86_64.sh"]




COPY environment.yml .

SHELL ["conda", "env", "create", "-f", "environment.yml", "--name", "TCP"]

SHELL ["conda", "run", "-n", "TCP", "/bin/bash", "-c"]

COPY docker-config.env /app/docker-config.env
ENV PYTHONPATH $PYTHONPATH:/app/TCP
COPY . .


CMD ["python", "TCP/train.py", "--epochs=$EPOCHS", "--lr=$LR", "--val_every=$VAL_EVERY", "--batch_size=$BATCH_SIZE", "--logdir=$LOGDIR", "--gpus=$GPUS", "--transferloading=$TRANSFERLOADING"]