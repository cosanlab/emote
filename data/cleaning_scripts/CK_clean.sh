#!/bin/bash

realpath() {
    [[ $1 = /* ]] && echo "$1" || echo "$PWD/${1#./}"
}

#Assumes CK data is in folder  at emote/data/CK+
if [ "$2" == "--help" ] || [ $# -lt "2" ]; then
    echo "CK+ data cleaning script"
    echo "Execution instructions:"
    echo "./CK_clean.sh <processed_image_dir> <fac_file>"
    exit
fi

if [ ! -d $1 ]; then
	mkdir $1
fi

if [ ! -e $2 ]; then
	touch $2
else
	> $2 #clear the old file
fi

PROJECT_HOME=`pwd`/../../

OUT_DIR=`pwd`
OUT_FILE=`pwd`
OUT_DIR="$OUT_DIR/$1"
OUT_FILE="$OUT_FILE/$2"


#Move to dir
cd ../CK+/FACS

#Compile FAC log
for subject_dir in *; do
	subject=`basename $subject_dir`
	for image_dir in $subject_dir/*; do
		image=`basename $image_dir`
		id=${subject}_${image}

		for file in $image_dir/*; do
			new_line=$id
			while read line; do
				new_line="$new_line $line"
			done <$file
			echo $new_line >> $OUT_FILE
			break 1
		done
	done
done

cd ../cohn-kanade-images

#Process images
for subject_dir in *; do
	subject=`basename $subject_dir`
	for image_dir in $subject_dir/*; do
		image=`basename $image_dir`
		id=${subject}_${image}
		image_file=`pwd`
		image_file=$image_file/$image_dir
		image_file=$image_file/`ls -t $image_dir | tail -1`
		echo -e "python $PROJECT_HOME/emote.py image -o $OUT_DIR/$id.jpg $image_file"

		python $PROJECT_HOME/emote.py image -o $OUT_DIR/$id.png $image_file
	done
done

echo "Processing complete"





