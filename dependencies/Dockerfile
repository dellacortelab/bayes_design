FROM nvidia/cuda:12.2.0-runtime-ubuntu20.04

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

RUN ls /tmp/
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt


# Install hhblits
RUN if [ -e /etc/apt/sources.list.d/nvidia-ml.list ]; then rm /etc/apt/sources.list.d/nvidia-ml.list; fi
RUN apt-get update && apt-get install --assume-yes --no-install-recommends --quiet cmake
RUN cd $HOME && git clone https://github.com/soedinglab/hh-suite.git
RUN cd $HOME/hh-suite && ls 
RUN cd $HOME/hh-suite && cmake -DCMAKE_INSTALL_PREFIX=. .
RUN cd $HOME/hh-suite && make -j 4 && make install
RUN export PATH="$PATH:$HOME/hh-suite/bin:$HOME/hh-suite/scripts"

ENTRYPOINT ["/bin/bash"]