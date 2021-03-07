from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import faiss

from practicalrecs_examples.ann_search import IndexTypes, build_ann_index
from practicalrecs_examples.filtering import build_bloom_filters
from practicalrecs_examples.matrix_factorization import load_model


@dataclass
class ArtifactSet:
    # model_name: str = None
    # index_type: str = None
    # filters_name: str = None

    model: Any = None
    index: Any = None
    filters: Any = None


class ArtifactRegistry:
    def __init__(self, use_cuda=True, gpu_id=0):
        self.models = {}
        self.indices = defaultdict(dict)
        self.filters = {}

        self.use_cuda = use_cuda
        self.gpu_id = gpu_id
        self.gpu_res = None

    def load_model(self, name, path):
        model = load_model(path)

        if self.use_cuda:
            model = model.cuda(self.gpu_id)

        self.models[name] = model

    def _build_ann_index_for_model(self, model_name, index_type, nprobe=1):
        if self.use_cuda and not self.gpu_res:
            self.gpu_res = faiss.StandardGpuResources()

        model = self.models[model_name]
        item_vectors = model.item_embeddings.weight.cpu().data.detach().numpy()

        build_ann_index(
            item_vectors,
            index_type,
            nprobe=nprobe,
            use_cuda=self.use_cuda,
            gpu_res=self.gpu_res,
            gpu_id=self.gpu_id,
        )

        self.indices[model_name][index_type]

    def build_ann_indices_for_model(self, model_name, nprobe=30):
        self._build_ann_index_for_model(model_name, IndexTypes.EXACT, nprobe)
        self._build_ann_index_for_model(model_name, IndexTypes.APPROX, nprobe)

    def build_bloom_filters(self, filters_name, *args, **kwargs):
        filters = build_bloom_filters(*args, **kwargs)

        self.filters[filters_name] = filters

        return filters

    def retrieve_artifacts(self, model_name=None, index_type=None, filters_name=None):
        model = self.models[model_name] if model_name else None
        index = self.indices[model_name][IndexTypes(index_type)] if index_type else None
        filters = self.filters[filters_name] if filters_name else None

        return ArtifactSet(model, index, filters)
