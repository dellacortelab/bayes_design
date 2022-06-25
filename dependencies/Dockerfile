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

# Install conda
RUN cd /home && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh && \
    bash Miniconda3-py38_4.10.3-Linux-x86_64.sh -b -p $HOME/miniconda
RUN $HOME/miniconda/bin/conda init bash

# Install conda packages
COPY bayes_design.yml /tmp/bayes_design.yml
RUN $HOME/miniconda/bin/conda env create -f tmp/bayes_design.yml

# Add commands to activate the conda environment on startup
RUN echo "export PATH=$HOME/miniconda/bin:$PATH" >> $HOME/.bashrc
RUN echo "conda init bash" >> $HOME/.bashrc
RUN echo "conda activate bayes_design" >> $HOME/.bashrc

# Install ProteinMPNN
RUN cd $HOME && git clone -q https://github.com/dauparas/ProteinMPNN.git
RUN export PYTHONPATH=$PYTHONPATH:$HOME/ProteinMPNN/vanilla_proteinmpnn

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-get update && apt-get install --assume-yes --no-install-recommends --quiet cmake

# Install hhblits
RUN cd $HOME && git clone https://github.com/soedinglab/hh-suite.git
RUN cd $HOME/hh-suite && ls 
RUN cd $HOME/hh-suite && cmake -DCMAKE_INSTALL_PREFIX=. .
RUN cd $HOME/hh-suite && make -j 4 && make install
RUN export PATH="$PATH:$HOME/hh-suite/bin:$HOME/hh-suite/scripts"
# Download reference HMM database for hhblits
RUN cd $HOME/hh-suite && mkdir databases
RUN cd $HOME/hh-suite/databases && wget http://wwwuser.gwdg.de/~compbiol/uniclust/2022_02/UniRef30_2022_02_hhsuite.tar.gz
RUN cd $HOME/hh-suite/databases && tar -xzvf UniRef30_2022_02_hhsuite.tar.gz

ENTRYPOINT ["/bin/bash"] 
