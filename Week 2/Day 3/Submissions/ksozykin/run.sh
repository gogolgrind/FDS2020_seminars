#!/bin/bash
data_dir='/ksozykinraid/data/nycflights' #'./nycflights/'
temp_dir='/ksozykinraid/tmp/' # to avoid temp files in /tmp
mkdir -p $data_dir
python3 download.py -d $data_dir 
python3 main.py -d $data_dir -t $temp_dir