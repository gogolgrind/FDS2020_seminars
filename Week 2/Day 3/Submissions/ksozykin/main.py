import warnings
warnings.filterwarnings("ignore")
import os
from glob import glob
import argparse
from dask.distributed import Client,LocalCluster
import dask
from solvers import XgbSolver,XgbDaskSolver

def parse_args():
    parser = argparse.ArgumentParser(description='nycflights')
    parser.add_argument('-d','--data_dir', type=str, default='/ksozykinraid/data/nycflights',
                    help='where nycflights stored')
    parser.add_argument('-t','--tmp_dir', type=str, default='/ksozykinraid/tmp/',
                    help='temp dir for joblib and dask')
    parser.add_argument('-b','--backend', type=str, default='pandas',
                    help='backend for model and dataframe processor')
    parser.add_argument('-m','--memory_limit', type=str, default='2GB',
                    help='memory_limit')
    parser.add_argument('-j','--n_jobs', type=int, default=4,
                    help='num of jobs')
    parser.add_argument('-jt','--threads_per_worker', type=int, default=2,
                    help='num of jobs')
    return parser.parse_args()

    
if __name__ == '__main__':
    
    args = parse_args()
    data_dir = args.data_dir
    tmp_dir =  args.tmp_dir 
    n_jobs = args.n_jobs
    threads_per_worker = args.threads_per_worker
    memory_limit = args.memory_limit
    dask.config.set({'temporary_directory': tmp_dir})
    os.environ['JOBLIB_TEMP_FOLDER'] = tmp_dir 
    cluster = LocalCluster(threads_per_worker=threads_per_worker, n_workers=n_jobs,memory_limit=memory_limit)
    client = Client(cluster)
    print('pandas + sklearn based solution')
    pd_solver = XgbSolver(n_jobs=n_jobs,data_dir=data_dir)
    pd_solver.solve()
    print(pd_solver.info())
    print('dask based solution')
    dask_solver = XgbDaskSolver(n_jobs=n_jobs,data_dir=data_dir)
    dask_solver.solve()
    print(dask_solver.info())