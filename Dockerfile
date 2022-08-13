FROM nvidia/cuda:11.3.1-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get -y install \
    python3 \
    python3-pip \
    python-is-python3 \
    default-jre
RUN pip install wheel

ADD requirements.txt setup.py ./
RUN pip install -r requirements.txt

ADD . ./
# make sure to install the current package
RUN pip install -r requirements.txt
