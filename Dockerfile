# syntax = docker/dockerfile:1.2

# create an image that has everything we need to build
# we can build only this image and work interactively
FROM ubuntu:20.04 AS builddep_light
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    curl \
    python3 \
    python3-numpy \
    python3-distutils \
    python3-dev \
    python3-pyaudio \
    python-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py

RUN pip3 install --upgrade requests grpcio-tools notebook librosa editdistance "ipython>=7.31.1"
RUN pip3 uninstall -y click

# Dependencies for building
FROM builddep_light AS builddep
ARG BAZEL_VERSION=3.7.2

RUN apt-get update && apt-get install -y \
    unzip \
    zip \
    wget \
    flac \
    libflac++-dev \
    parallel \
    alsa-base \
    libasound2-dev \
    linux-libc-dev \
    alsa-utils \
    bc \
    build-essential \
    cmake \
    libboost-test-dev \
    zlib1g-dev \
    libbz2-dev \
    liblzma-dev \
    git \
    vim \
    sox \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade sklearn transformers

#Install NGC client.
RUN wget https://ngc.nvidia.com/downloads/ngccli_bat_linux.zip && unzip ngccli_bat_linux.zip && chmod u+x ngc && \
    echo "PATH=/:$PATH\n" >> /root/.bashrc
RUN md5sum -c ngc.md5
ENV PATH="/:${PATH}"


# copy the source and run build
FROM builddep as builder

WORKDIR /work
COPY VERSION ./
COPY .git /work/.git
COPY ./riva/proto /work/riva/proto
COPY third_party /work/third_party
COPY ./scripts/ scripts

COPY python /work/python
RUN python3 python/clients/setup.py bdist_wheel

# create lightweight client image
FROM builddep_light AS riva-api-client

ENV PYTHONPATH="${PYTHONPATH}:/work/"

WORKDIR /work
COPY --from=builder /work/riva/proto/ /work/riva/proto/
COPY --from=builder /work/dist /work
RUN pip install *.whl
RUN python3 -m pip uninstall -y pip
COPY ./python/clients/asr/*.py ./python/
COPY ./python/clients/nlp/riva_nlp/test_qa.py ./python/
COPY ./python/clients/tts/talk_stream.py ./python/
COPY ./python/clients/tts/talk.py ./python/

# create client image for CI and devel
FROM builddep AS riva-api-client-dev

ENV PYTHONPATH="${PYTHONPATH}:/work/"

WORKDIR /work
COPY --from=builder /work/dist /work
RUN pip install *.whl
#Uninstall pip to address CVE-2018-20225
RUN python3 -m pip uninstall -y pip
COPY --from=builder /work/riva/proto/ /work/riva/proto/

COPY ./python/clients/nlp/riva_nlp/test_qa.py ./python/
COPY ./python/clients/asr/*.py ./python/
COPY ./python/clients/tts/*.py ./python/
COPY ./python/clients/nlp/*.py ./python/

COPY examples /work/examples

