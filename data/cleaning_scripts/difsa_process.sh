#!/bin/bash

for FILE in /idata/lchang/Data/FACS_data/DIFSA/Video_LeftCamera/*; do 
    NAME="${FILE##*/}"
    python preprocess.py mirror $FILE /idata/lchang/Data/FACS_data/DIFSA_clean/Images/Mirror${NAME:0:10}/ 96
done
