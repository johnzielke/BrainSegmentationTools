FROM FROM nvidia/cuda:13.2.0-cudnn-devel-ubuntu24.04

# Install system tools needed for Python/CUDA development and repo bootstrap.
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
        build-essential \
        cmake \
        pkg-config \
        git \
        git-annex \
        gdb \
        lldb \
        linux-perf \
        tmux \
        just \
    && apt-get autoremove -y && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*
