import os
import numpy as np
import pandas as pd
import tarfile
import urllib.request
import zipfile
from glob import glob
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='nycflights')
    parser.add_argument('-d','--data_dir', type=str, default='/ksozykinraid/data/nycflights',
                    help='where to store nycflights')
    parser.add_argument('-n','--n_rows', type=int, default=10000,
                    help='number of lines to process')
    
    return parser.parse_args()

def get_data(data_dir,n_rows):
    flights_raw = os.path.join(data_dir, 'nycflights.tar.gz')
    flightdir = os.path.join(data_dir, 'nycflights')
    jsondir = os.path.join(data_dir, 'flightjson')

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    if not os.path.exists(flights_raw):
        print("- Downloading NYC Flights dataset... ", end='', flush=True)
        url = "https://storage.googleapis.com/dask-tutorial-data/nycflights.tar.gz"
        urllib.request.urlretrieve(url, flights_raw)
        print("done", flush=True)

    if not os.path.exists(flightdir):
        print("- Extracting flight data... ", end='', flush=True)
        tar_path = os.path.join(data_dir, 'nycflights.tar.gz')
        with tarfile.open(tar_path, mode='r:gz') as flights:
            flights.extractall(data_dir)
        print("done", flush=True)

    if not os.path.exists(jsondir):
        print("- Creating json data... ", end='', flush=True)
        os.mkdir(jsondir)
        for path in glob(os.path.join(data_dir, 'nycflights', '*.csv')):
            prefix = os.path.splitext(os.path.basename(path))[0]
            # Just take the first n_rows rows for the demo
            df = pd.read_csv(path).iloc[:n_rows]
            df.to_json(os.path.join(data_dir, 'flightjson', prefix + '.json'),
                       orient='records', lines=True)
        print("done", flush=True)


def main():
    print("Setting up data directory")
    print("-------------------------")
    args = parse_args()
    data_dir = args.data_dir
    n_rows = args.n_rows
    get_data(data_dir,n_rows)
    print("** Finished! **")
    
if __name__ == '__main__':
    main()