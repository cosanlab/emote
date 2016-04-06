#!/bin/bash

realpath() {
    [[ $1 = /* ]] && echo "$1" || echo "$PWD/${1#./}"
}

#Assumes CK data is in folder  at emote/data/CK+
if [ "$2" == "--help" ] || [ $# -lt "2" ]; then
    echo "DIFSA+ data cleaning script"
    echo "Execution instructions:"
    echo "./DIFSA_clean.sh <processed_image_dir> <emote_dir>"
    exit
fi

if [ ! -d $1 ]; then
	mkdir $1
fi


OUT_DIR=`pwd`
OUT_FILE=`pwd`
OUT_DIR="$OUT_DIR/$1"
OUT_FILE="$OUT_FILE/$2"
EMOTE_DIR=$2


#Move to dir
cd ../DIFSA/Video_LeftCamera

#Process images
for video_file in *; do
	python $EMOTE_DIR/emote.py video -O $OUT_DIR/ $video_file
done

echo "Processing complete"





