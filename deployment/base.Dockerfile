# CUDA基础镜像
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04
# CPU版本
# FROM ubuntu:20.04

# 安装基础包
RUN apt update && \
apt install -y \
    wget build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev \
    libreadline-dev libffi-dev libsqlite3-dev libbz2-dev liblzma-dev libopenmpi-dev \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 zbar-tools&& \
apt clean && \
rm -rf /var/lib/apt/lists/*

WORKDIR /temp

# 下载python
RUN wget https://www.python.org/ftp/python/3.8.5/Python-3.8.5.tgz && \
    tar -xvf Python-3.8.5.tgz

# 编译&安装python
RUN cd Python-3.8.5 && \
    ./configure --enable-optimizations && \
    make && \
    make install

WORKDIR /workspace

RUN rm -r /temp && \
    ln -s /usr/local/bin/python3 /usr/local/bin/python && \
    ln -s /usr/local/bin/pip3 /usr/local/bin/pip

# 安装pytorch
# https://pytorch.org/get-started/locally/
RUN pip3 install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
RUN rm -r /root/.cache/pip