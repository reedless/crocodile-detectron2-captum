# sudo docker build -t captum -f Dockerfile .

FROM nvidia/cuda:10.1-devel-ubuntu18.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y \
    libgl1 libsm6 libxrender1 libglib2.0-0 \
	python3-pip \
    && apt-get clean && rm -rf /tmp/* /var/tmp/*

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -r requirements.txt
RUN python3 -m pip install torch==1.8.1+cu101 torchvision==0.9.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
RUN python3 -m pip install 'git+https://github.com/facebookresearch/detectron2.git'