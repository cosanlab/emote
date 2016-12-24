#!/bin/bash

BACKEND=$1
CODE=$2
DATA=$3
MODEL=$4
nvidia-docker run --rm -it -e KERAS_BACKEND=$BACKEND -v $CODE:/emote -v $DATA:/data -v $MODEL:/models emote/ghosh python /emote/optimize.py /data/training /data/validation /models
