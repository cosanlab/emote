#!/bin/bash

for FILE in /idata/lchang/Data/FACS_data/DIFSA/Video_RightCamera/*; do 
    NAME="${FILE##*/}"
    python preprocess.py mirror $FILE /idata/lchang/Data/FACS_data/DIFSA_clean/Images/Mirror${NAME:0:11}/ 96
done
