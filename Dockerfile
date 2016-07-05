#Dockerfile for emote
#
#Ubuntu base container with:
#OpenCV 3.1
#dlib
#pip
#tensorflow


FROM ubuntu:15.04
MAINTAINER Dan Whitcomb <danrwhitcomb@gmail.com>

####INSTALL OPENCV######

ARG OPENCV_VERISON="3.0.0"
ARG CONTRIB_DIR=/tmp/contrib
ARG OPENCV_DIR=/tmp/opencv-$OPENCV_VERISON/build
ARG DLIB_DIR=/tmp/dlib
ARG WITH_GPU
ARG EMOTE_PATH


#OpenCV dependencies
RUN apt-get -y update
RUN apt-get -y install build-essential
RUN apt-get -y install cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
RUN apt-get -y install python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev

#Get OpenCV Source
RUN mkdir -p $OPENCV_DIR
RUN mkdir -p $CONTRIB_DIR
RUN curl -sL https://github.com/Itseez/opencv/archive/$OPENCV_VERISON.tar.gz | tar xvz -C /tmp
RUN curl -sL https://github.com/Itseez/opencv_contrib/archive/master.tar.gz | tar xvz -C $CONTRIB_DIR

WORKDIR $OPENCV_DIR

# install OpenCV
RUN cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D INSTALL_C_EXAMPLES=OFF \
    -D INSTALL_PYTHON_EXAMPLES=ON \
    -D OPENCV_EXTRA_MODULES_PATH=$CONTRIB_DIR \
    -D BUILD_EXAMPLES=ON ..

RUN make
RUN make install

RUN echo "/usr/local/lib" > /etc/ld.so.conf.d/opencv.conf
RUN ldconfig
RUN echo "ln /dev/null /dev/raw1394" >> /etc/bash.bashrc

#### Install dlib ####
RUN apt-get -y install libboost-python-dev python-setuptools
RUN mkdir $DLIB_DIR
RUN curl -sL https://github.com/davisking/dlib/archive/master.tar.gz | tar xvz -C $DLIB_DIR
RUN python $DLIB_DIR/dlib-master/setup.py install


#### Install OpenFace
WORKDIR /tmp
RUN git clone https://github.com/cmusatyalab/openface.git
WORKDIR /tmp/openface
RUN python setup.py install

# Setup python deps
RUN apt-get -y install python-pip 
RUN pip install pillow
ARG TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.9.0-cp27-none-linux_x86_64.whl
RUN pip install --upgrade $TF_BINARY_URL


RUN mkdir /source

VOLUME ["/source"]
WORKDIR /source

CMD CMD bash -C '/source/startup.sh';'bash'
