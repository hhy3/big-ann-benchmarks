from __future__ import absolute_import

import numpy as np

from benchmark.algorithms.base import BaseANN
from benchmark.datasets import DATASETS
import linscancpp

class Linscan(BaseANN):
    def __init__(self, metric, index_params):
        assert metric == "ip"
        self.name = "linscancpp"
        self._index = linscancpp.Index()
        self._budget = 0.0
        self.drop_ratio = index_params['drop_ratio']
        print("Linscan index initialized: " + str(self._index))

    def fit(self, dataset): # e.g. dataset = "sparse-small"

        self.ds = DATASETS[dataset]()
        assert self.ds.data_type() == "sparse"

        print("start add")
        self._index.add(self.ds.get_dataset_fn(), self.drop_ratio)
        print("done add")

        print("Index status: " + str(self._index))


    def load_index(self, dataset):
        return None

    def set_query_arguments(self, query_args):
        self._budget = query_args["budget"]

    def query(self, X, k):
        """Carry out a batch query for k-NN of query set X."""
        nq = X.shape[0]
        self.res = self._index.search_batch(nq, X.indptr, X.indices, X.data, k, self._budget).reshape(-1, k)

    def get_results(self):
        return self.res

