import xgboost
from dask_ml import xgboost as dxgboost
import pandas as pd
import dask.dataframe as dpd
from .base import BaseSolver

class XgbSolver(BaseSolver):
    
    def __init__(self,max_years=10,
                      seed = 42,
                      test_size = 0.25,
                      data_dir = '/ksozykinraid/data/nycflights',
                      n_jobs = 4,
                      do_cv = True,verbose=True):
        
        self.set_const(data_dir,max_years,seed,test_size,n_jobs,do_cv,verbose)
        self.model = xgboost.XGBRegressor(max_depth=6,n_estimators=500,
                                          reg_lambda=10,n_jobs=self.n_jobs)
        self.backend_name = 'pandas'
        self.backend = pd
        self.param_grid = {
            'n_estimators': [100, 300, 500],
            'reg_lambda': [0.001, 0.01, 0.1, 1, 10],
            'max_depth': [6, 8, 12],
            'n_jobs' : [self.n_jobs]
        }
        
class XgbDaskSolver(BaseSolver):
    
    def __init__(self,max_years=10,
                      seed = 42,
                      test_size = 0.25,
                      data_dir = '/ksozykinraid/data/nycflights',
                      n_jobs = 4,
                      do_cv = False,verbose=True):
        
        self.set_const(data_dir,max_years,seed,test_size,n_jobs,do_cv,verbose)
        self.model = dxgboost.XGBRegressor(max_depth=6,n_estimators=500,
                                          reg_lambda=10,n_jobs=self.n_jobs)
        self.backend = dpd
        self.backend_name = 'dask'
        self.param_grid = {
            'n_estimators': [100, 300, 500],
            'reg_lambda': [0.001, 0.01, 0.1, 1, 10],
            'max_depth': [6, 8, 12],
            'n_jobs' : [self.n_jobs]
        }