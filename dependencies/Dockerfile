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

# Install pip packages
RUN pip install --upgrade pip
RUN pip install torch===1.4.0 torchvision===0.5.0 -f https://download.pytorch.org/whl/torch_stable.html
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Install ProteinMPNN
RUN cd $HOME && git clone -q https://github.com/dauparas/ProteinMPNN.git
RUN echo "export PYTHONPATH=$PYTHONPATH:$HOME/ProteinMPNN/vanilla_proteinmpnn" > ~/.bashrc

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
# RUN cd $HOME/hh-suite && mkdir databases
# RUN cd $HOME/hh-suite/databases && wget http://wwwuser.gwdg.de/~compbiol/uniclust/2022_02/UniRef30_2022_02_hhsuite.tar.gz
# RUN cd $HOME/hh-suite/databases && tar -xzvf UniRef30_2022_02_hhsuite.tar.gz

ENTRYPOINT ["/bin/bash"] 
