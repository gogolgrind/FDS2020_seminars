#!/bin/bash
data_dir='./nycflights/' 
temp_dir='./tmp/' # to avoid temp files in /tmp due to remote machine specification
n_jobs=8 # also trained n_jobs 4, found that  at least on this remote machine args.threads_per_worker more important
mkdir -p $data_dir
python3 download.py -d $data_dir -n 1000 # also traied n_rows 10000, in both cases all csv files is used
python3 main.py -d $data_dir -t $temp_dir -j $n_jobs