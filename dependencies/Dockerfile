FROM nvidia/cuda:11.2.1-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN \
  # Update package list
  apt-get update -y && \
  # Install...
  apt-get install -y \
  git \
  vim \
  wget \
  python3 \
  python3-dev \
  python3-pip \
  tmux \
  # Remove package lists
  && rm -rf /var/lib/apt/lists/*

# # Install conda
# RUN cd /home && \
#     wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh && \
#     bash Miniconda3-py38_4.10.3-Linux-x86_64.sh -b -p $HOME/miniconda
# RUN $HOME/miniconda/bin/conda init bash

# # Install conda dependencies
# COPY bayes_design_env.yml /tmp/bayes_design_env.yml
# RUN $HOME/miniconda/bin/conda env create -f tmp/bayes_design_env.yml

# RUN echo "export PATH=$HOME/miniconda/bin:$PATH" >> $HOME/.bashrc
# RUN echo "conda init bash" >> $HOME/.bashrc
# RUN echo "conda activate bayes_design_env" >> $HOME/.bashrc

RUN pip3 install --upgrade pip

RUN pip3 install torch --extra-index-url https://download.pytorch.org/whl/cu116

COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt

# # Install ProteinMPNN
# RUN cd $HOME && git clone -q https://github.com/dauparas/ProteinMPNN.git
# # Checkout the hash that this codebase is based on
# RUN cd $HOME/ProteinMPNN && git reset --hard e61ecb7e3c32e630ff7a34d16c3a43fcf8f8a8bd
# RUN echo "export PYTHONPATH=$PYTHONPATH:$HOME/ProteinMPNN/" >> $HOME/.bashrc

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-get update && apt-get install --assume-yes --no-install-recommends --quiet cmake

ENTRYPOINT ["/bin/bash"]