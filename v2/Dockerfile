FROM nvidia/cuda:7.5-cudnn5-devel

#THEANO CONFIG
ENV THEANO_FLAGS=device=gpu,floatX=float32,lib.cnmem=1,optimizer=fast_compile

#KERAS CONFIG
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
  && rm -rf /var/lib/apt/lists/*

RUN curl -qsSLkO \
    https://repo.continuum.io/miniconda/Miniconda-latest-Linux-`uname -p`.sh \
  && bash Miniconda-latest-Linux-`uname -p`.sh -b \
  && rm Miniconda-latest-Linux-`uname -p`.sh

ENV PATH=/root/miniconda2/bin:$PATH

RUN conda install -y \
    h5py \
    pandas \
    theano \
    scikit-learn
RUN  conda clean --yes --tarballs --packages --source-cache 
RUN  pip install --upgrade -I setuptools
RUN  pip install --upgrade \
         keras \
         https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.9.0rc0-cp27-none-linux_x86_64.whl \
         hyperopt \
         matplotlib
