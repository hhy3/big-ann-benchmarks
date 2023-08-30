import faiss
import numpy as np
from neurips23.ood.base import BaseOODANN
from benchmark.datasets import DATASETS, download_accelerated
import os
import time
import multiprocessing as mp

class Fastscan(BaseOODANN):
    def __init__(self, metric, index_params):
        self.name = "fastscan"
        if (index_params.get("nlist")==None):
            print("Error: missing parameter nlist")
            return
        self._index_params = index_params
        self._metric = metric

        self.nlist = index_params.get('nlist')

    def index_name(self):
        return f"{self.name}_nlist{self.nlist}"
      
    def load_index(self, dataset):
      pass

    def create_index_dir(self, dataset):
        index_dir = os.path.join(os.getcwd(), "data", "indices", "ood")
        os.makedirs(index_dir, mode=0o777, exist_ok=True)
        index_dir = os.path.join(index_dir, 'fastscan')
        os.makedirs(index_dir, mode=0o777, exist_ok=True)
        index_dir = os.path.join(index_dir, dataset.short_name())
        os.makedirs(index_dir, mode=0o777, exist_ok=True)
        index_dir = os.path.join(index_dir, self.index_name())
        os.makedirs(index_dir, mode=0o777, exist_ok=True)
        return index_dir
      
    def translate_dist_fn(self, metric):
        if metric == 'euclidean':
            return faiss.METRIC_L2
        elif metric == 'ip':
            return faiss.METRIC_INNER_PRODUCT
        else:
            raise Exception('Invalid metric')
          
    def translate_dtype(self, dtype:str):
        if dtype == 'uint8':
            return np.uint8
        elif dtype == 'int8':
            return np.int8
        elif dtype == 'float32':
            return np.float32
        else:
            raise Exception('Invalid data type')
          
    def fit(self, dataset):
        """
        Build the index for the data points given in dataset name.
        """

        ds = DATASETS[dataset]()
        d = ds.d

        index_dir = self.create_index_dir(ds)
        
        if  hasattr(self, 'index'):
            print('Index object exists already')
            return

        print(ds.get_dataset_fn())

        index_name = f"IVF{self.nlist},PQ{d//2}x4fsr"
        index_path = os.path.join(index_dir, index_name)
        self.index = faiss.index_factory(d, index_name, self.translate_dist_fn(self._metric))
        self.index.cp.spherical = False
        if os.path.exists(index_path):
          self.index = faiss.read_index(index_path)
        else:
          faiss.omp_set_num_threads(8)
          print("start train")
          print(ds.get_dataset().shape)
          assert(self.index.cp.spherical == False)
          self.index.train(ds.get_dataset())
          print("start add")
          self.index.add(ds.get_dataset())
          print("start write")
          faiss.write_index(self.index, index_path)
        self.refine = faiss.index_factory(d, "SQfp16", self.translate_dist_fn(self._metric))
        self.refine.add(ds.get_dataset())
        self.index_refine = faiss.IndexRefine(self.index, self.refine)
        self.index_refine.k_factor = 20
        print('Index ready for search')
        
        
    def query(self, X, k):
        """Carry out a batch query for k-NN of query set X."""
        _, self.res = self.index_refine.search(X, k=k)
        
    def set_query_arguments(self, query_args):
        self._query_args = query_args
        self.nprobe = self._query_args['nprobe']
        self.index.nprobe = self.nprobe

    def __str__(self):
        return f'fastscan({self.index_name(), self._query_args})'
      