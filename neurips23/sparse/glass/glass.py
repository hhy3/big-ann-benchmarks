from __future__ import absolute_import

import numpy as np

from benchmark.algorithms.base import BaseANN
from benchmark.datasets import DATASETS
import glass


class Glass(BaseANN):
    def __init__(self, metric, index_params):
        assert metric == "ip"
        self.name = "linscan"
        self._index = glass.SparseIndex(index_params['L'], index_params['R'])
        print("SparseGlass index initialized: " + str(self._index))

    def fit(self, dataset):  # e.g. dataset = "sparse-small"

        self.ds = DATASETS[dataset]()
        assert self.ds.data_type() == "sparse"

        N_VEC_LIMIT = 100000  # batch size
        it = self.ds.get_dataset_iterator(N_VEC_LIMIT)
        for d in it:
            for i in range(d.shape[0]):
                d1 = d.getrow(i)
                self._index.insert(dict(zip(d1.indices, d1.data)))
        g = self._index.build()
        g.save('fuck.glass')
        self.searcher = glass.SparseSearcher(g, lst)
        print("Index status: " + str(self._index))

    def load_index(self, dataset):
        return None

    def set_query_arguments(self, query_args):
        self.searcher.set_ef(query_args["ef"])

    def query(self, X, k):
        """Carry out a batch query for k-NN of query set X."""
        nq = X.shape[0]

        # prepare the queries as a list of dicts
        self.queries = []
        for i in range(nq):
            qc = X.getrow(i)
            q = dict(zip(qc.indices, qc.data))
            self.queries.append(q)

        res = self.searcher.batch_search(self.queries, k)
        self.I = np.array(res, dtype='int32').reshape(-1, k)

    def get_results(self):
        return self.I
