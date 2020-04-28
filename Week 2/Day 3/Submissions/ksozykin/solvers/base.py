from .abc import AbcSolver
import time
from dask_ml.model_selection import train_test_split, GridSearchCV
import pandas as pd
import dask.dataframe as dpd
from glob import glob
import dask
import joblib

class BaseSolver(AbcSolver):
    """
        the idea here is create base class that useful with any model with eather pandas or dask as dataframe backend
    """
    def __init__(self,data_dir = '/ksozykinraid/data/nycflights'):
        self.model = None
        self.backend = None
        self.self.param_grid = None
        self.backend_name = ''
        
        
    def set_const(self,
                   data_dir,
                   max_years,
                   seed,test_size,n_jobs,do_cv,verbose):
        self.unique_cities = ['ABE', 'ALB', 'ATL', 'BDL', 'BOS', 'BTV', 'BUF', 'BWI', 'CAE',
           'CHO', 'CHS', 'CLE', 'CLT', 'CMH', 'CRW', 'CVG', 'DAB', 'DAY',
           'DCA', 'DEN', 'DFW', 'DTW', 'EWR', 'FLL', 'GSO', 'GSP', 'HOU',
           'IAD', 'IAH', 'IND', 'JAX', 'JFK', 'LAX', 'LGA', 'MCI', 'MCO',
           'MDW', 'MEM', 'MHT', 'MIA', 'MKE', 'MLB', 'MSP', 'MSY', 'MTJ',
           'ORD', 'ORF', 'PBI', 'PDX', 'PHL', 'PHX', 'PIT', 'PVD', 'PWM',
           'RDU', 'RIC', 'ROA', 'ROC', 'RSW', 'SAN', 'SAT', 'SDF', 'SEA',
           'SFO', 'SJC', 'SJU', 'SLC', 'SNA', 'SRQ', 'STL', 'STT', 'SYR',
           'TPA']
        self.json_paths = glob("{}/{}/*.json".format(data_dir,'flightjson','*.json'))
        self.seed = seed
        self.test_size = test_size
        self.n_jobs = n_jobs
        self.max_years = max_years
        self._info = {}
        self.target_column  = 'DepDelay'
        self.feature_colums = ['Year', 'Month', 'DayofMonth', 'DayOfWeek', 'DepTime', 'CRSDepTime',
                               'ArrTime', 'CRSArrTime', 'FlightNum', 'ActualElapsedTime',
                               'CRSElapsedTime', 'AirTime', 'ArrDelay', 'Distance', 'Cancelled',
                               'Diverted']
        self.do_cv = do_cv
        self.verbose = verbose
        
    
    @dask.delayed
    def load_json(self,json):
        return pd.read_json(json,lines=True)
        
    @dask.delayed
    def transform(self,dfi):
        dfi = dfi[dfi['ArrDelay'].notnull()]
        dfi['Dest'] = dfi['Dest'].apply(lambda x:  self.unique_cities.index(x))
        dfi['Origin'] = dfi['Origin'].apply(lambda x:  self.unique_cities.index(x))
        dfi = dfi.drop(['TailNum','UniqueCarrier',"TaxiIn","TaxiOut"],1)
        return dfi
        
    def parse(self):
        dataset = []
        for idx,json in enumerate(self.json_paths[:self.max_years]):
            dfi = self.transform(self.load_json(json))
            dataset.append(dfi)
        dataset = dpd.from_delayed(dataset)
        if self.backend_name == 'pandas':
            dataset = dataset.compute()
        y = dataset[self.target_column]
        X = dataset[self.feature_colums]
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X,y, 
                                                    test_size=self.test_size, random_state=self.seed)
        
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train,self.y_train,
                                                    test_size=self.test_size, random_state=self.seed)
    
    def train(self):
        start = time.time()
        self.model.fit(self.X_train, self.y_train)
        stop = time.time()
        self._info['FIT_TIME'] = stop - start
        
        
    def cv(self):
        gcv = GridSearchCV(self.model, self.param_grid,
                                   cv=3, n_jobs=1)
        start = time.time()
        with joblib.parallel_backend("dask"):
            gcv.fit(self.X_val, self.y_val)
        stop = time.time()
        self._info['CV_TIME'] = stop - start
        self.model = self.model.__class__(**gcv.best_params_)
            
    def info(self):
        return self._info
    
    def predict(self):
        return self.model.predict(self.X_test)
    
    def test(self):
        y_pred = self.predict()
        return ((self.y_test-y_pred)**2).mean(0)
    
    def solve(self):
        if self.verbose:
            print("{} based solution".format(self.backend_name))
        self.parse()
        if self.do_cv:
            self.cv()
        else:
            self._info['CV_TIME'] = '-1'
        self.train()
        test_mse = self.test()
        if self.backend_name == 'dask':
            test_mse = test_mse.compute()
        self._info['TEST_MSE'] = test_mse
        