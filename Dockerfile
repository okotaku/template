FROM nvcr.io/nvidia/pytorch:24.03-py3

RUN apt update -y && apt install -y \
    git tmux gh
RUN apt-get update && apt-get install -y \
    vim \
    libgl1-mesa-dev \
    zsh \
    python3-tk
ENV FORCE_CUDA="1"

# Zsh install
ENV SHELL /bin/zsh
RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.5/zsh-in-docker.sh)" -- \
    -t robbyrussell

# Install python package.
WORKDIR /modules
COPY ./ /modules
RUN pip install --upgrade pip

# Install xformers
RUN pip install ninja
RUN export TORCH_CUDA_ARCH_LIST="8.6 9.0+PTX" MAX_JOBS=2 && \
    pip install -v -U git+https://github.com/facebookresearch/xformers.git@v0.0.24#egg=xformers

# Install Stable Fast
RUN export TORCH_CUDA_ARCH_LIST="8.6 9.0+PTX" MAX_JOBS=2 && \
    pip install -v -U git+https://github.com/chengzeyi/stable-fast.git@main#egg=stable-fast

# Install modules
RUN pip install . && \
    pip install pre-commit && \
    pip uninstall -y $(pip list --format=freeze | grep opencv) && \
    rm -rf /usr/local/lib/python3.10/dist-packages/cv2/ && \
    pip install opencv-python-headless

# patches
RUN cp diffengine/patches/attention.py /usr/local/lib/python3.10/dist-packages/diffusers/models/attention.py && \
    cp diffengine/patches/transformer_2d.py /usr/local/lib/python3.10/dist-packages/diffusers/models/transformers/transformer_2d.py && \
    cp diffengine/patches/resnet.py /usr/local/lib/python3.10/dist-packages/diffusers/models/resnet.py && \
    cp diffengine/patches/unet_2d_condition.py /usr/local/lib/python3.10/dist-packages/diffusers/models/unets/unet_2d_condition.py

# Language settings
ENV LANG C.UTF-8
ENV LANGUAGE en_US

RUN git config --global --add safe.directory /workspace

WORKDIR /workspace
