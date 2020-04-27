#!/bin/bash
data_dir='./nycflights/' 
temp_dir='./tmp/' # to avoid temp files in /tmp due to remote machine specification
n_jobs=8
mkdir -p $data_dir
python3 download.py -d $data_dir -n 1000
python3 main.py -d $data_dir -t $temp_dir -j $n_jobs