from __future__ import absolute_import
import psutil
import os
import time
import numpy as np
import glass

from neurips23.ood.base import BaseOODANN
from benchmark.datasets import DATASETS, download_accelerated


class Glass(BaseOODANN):
    def __init__(self, metric, index_params):
        self.name = "glass"
        if (index_params.get("R") == None):
            print("Error: missing parameter R")
            return
        if (index_params.get("L") == None):
            print("Error: missing parameter L")
            return
        self._index_params = index_params
        self._metric = metric

        self.R = index_params.get("R")
        self.L = index_params.get("L")

        self.dir = "indices"
        self.path = self.index_name()

    def index_name(self):
        return f"R{self.R}_L{self.L}"

    def translate_dist_fn(self, metric):
        if metric == 'euclidean':
            return 'L2'
        elif metric == 'ip':
            return 'IP'
        else:
            raise Exception('Invalid metric')

    def translate_dtype(self, dtype: str):
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

        buildthreads = self._index_params.get("buildthreads", -1)
        print(buildthreads)
        if buildthreads == -1:
            buildthreads = 0

        if hasattr(self, 'index'):
            print('Index object exists already')
            return

        print(ds.get_dataset_fn())

        # X = np.fromfile(ds.get_dataset_fn())
        # shape = X.view('int32')[:2]
        # X = X.view('float32')[2:].reshape(shape)

        if not os.path.exists(self.dir):
            os.mkdir(self.dir)
        if self.path not in os.listdir(self.dir):
            p = glass.Index("HNSW", dim=d,
                            metric=self.translate_dist_fn(ds.distance()), quant="SQ8U", R=self.R, L=self.L)
            g = p.build(ds.get_dataset())
            g.save(os.path.join(self.dir, self.path))
            del p
            del g
        g = glass.Graph(os.path.join(self.dir, self.path))
        self.searcher = glass.Searcher(
            g, ds.get_dataset(), self.translate_dist_fn(ds.distance()), "SQ8U")
        self.searcher.optimize()
        print('Index ready for search')

    def load_index(self, dataset):
        """
        Load the index for dataset. Returns False if index
        is not available, True otherwise.

        Checking the index usually involves the dataset name
        and the index build paramters passed during construction.
        """
        return False

    def query(self, X, k):
        """Carry out a batch query for k-NN of query set X."""
        nq, dim = (np.shape(X))
        self.res = self.searcher.batch_search(
            X, k).reshape(nq, -1)

    def set_query_arguments(self, query_args):
        self.searcher.set_ef(query_args.get("ef"))

    def __str__(self):
        return f'glass({self.index_name()})'
