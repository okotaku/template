FROM ghcr.io/astral-sh/uv:latest

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    tmux \
    gh \
    vim \
    libgl1-mesa-dev \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
ARG USERNAME=developer
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Set environment variables
ENV FORCE_CUDA="1"
ENV LANG=C.UTF-8
ENV LANGUAGE=en_US

# Switch to non-root user
USER $USERNAME
WORKDIR /home/$USERNAME/workspace

# Copy project files with correct ownership
COPY --chown=$USER_UID:$USER_GID pyproject.toml uv.lock ./

# Install dependencies using uv
RUN uv sync --frozen --no-dev

# Copy source code with correct ownership
COPY --chown=$USER_UID:$USER_GID . .

# Install package in development mode
RUN uv pip install -e .

# Set git safe directory for user
RUN git config --global --add safe.directory /home/$USERNAME/workspace
RUN git config --global user.name "Developer"
RUN git config --global user.email "developer@example.com"

# Set default working directory
WORKDIR /home/$USERNAME/workspace
