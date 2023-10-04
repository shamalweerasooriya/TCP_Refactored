FROM nvidia/cuda:11.3-base
RUN apt-get update && apt-get install -y wget && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy
ENV PATH /opt/conda/bin:$PATH
WORKDIR /app
COPY environment.yml .
RUN conda env create -f environment.yml --name TCP
SHELL ["conda", "run", "-n", "TCP", "/bin/bash", "-c"]
COPY docker-config.env /app/docker-config.env
ENV PYTHONPATH $PYTHONPATH:/app/TCP
COPY . .


CMD ["python", "TCP/train.py", "--epochs=$EPOCHS", "--lr=$LR", "--val_every=$VAL_EVERY", "--batch_size=$BATCH_SIZE", "--logdir=$LOGDIR", "--gpus=$GPUS", "--transferloading=$TRANSFERLOADING"]