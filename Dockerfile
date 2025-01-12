FROM nvcr.io/nvidia/pytorch:24.12-py3

RUN apt update -y && apt install -y \
    git tmux gh
RUN apt-get update && apt-get install -y \
    vim \
    libgl1-mesa-dev \
    zsh
ENV FORCE_CUDA="1"

# Zsh install
ENV SHELL /bin/zsh
RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.5/zsh-in-docker.sh)" -- \
    -t robbyrussell

# Install python package.
WORKDIR /modules
COPY ./ /modules
RUN pip install --upgrade pip

# Install modules
RUN pip install . && \
    pip install pre-commit && \
    pip uninstall -y $(pip list --format=freeze | grep opencv) && \
    rm -rf /usr/local/lib/python3.10/dist-packages/cv2/ && \
    pip install opencv-python-headless

# Language settings
ENV LANG C.UTF-8
ENV LANGUAGE en_US

RUN git config --global --add safe.directory /workspace

WORKDIR /workspace
