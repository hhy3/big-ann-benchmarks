import numpy as np
import scann

from neurips23.ood.base import BaseOODANN
from benchmark.datasets import DATASETS, download_accelerated


class Scann(BaseOODANN):
    def __init__(self, metric, index_params):
        nlist = index_params['nlist']
        avq_threshold = 0.2
        self.nlist = nlist
        self.avq_threshold = avq_threshold
        self.dims_per_block = 2
        self.dist = {"ip" : "dot_product", "euclidean" : "squared_l2"}[metric]
        print(self.dist)
        self.name = "scann n_leaves={} avq_threshold={:.02f} dims_per_block={}".format(
            self.nlist, self.avq_threshold, self.dims_per_block
        )

    def fit(self, dataset):
        ds = DATASETS[dataset]()
        X = ds.get_dataset()
        n, d = X.shape
        print("START BUILD")
        self.searcher = (
            scann.scann_ops_pybind.builder(X, 10, self.dist)
            .tree(self.nlist, 1, training_sample_size=n, spherical=False, quantize_centroids=True)
            .score_ah(self.dims_per_block, anisotropic_quantization_threshold=self.avq_threshold)
            .reorder(1)
            .build()
        )
        print("DONE BUILD")

    def load_index(self, x):
        pass
    
    def set_query_arguments(self, query_args):
        self.nprobe = query_args['nprobe']
        self.reorder = 200

    def query(self, v, n):
        self.res = self.searcher.search_batched(v, n, self.reorder, self.nprobe)[0]

