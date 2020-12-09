import faiss
import numpy as np
import torch as th

from .pipeline import RecsPipelineComponent, UserRecs


def build_nn_search_index(item_vectors, embedding_dim, index_type, nprobe=1, gpu=0):
    index = faiss.index_factory(embedding_dim, index_type, faiss.METRIC_INNER_PRODUCT)
    index.nprobe = nprobe

    if th.cuda.is_available():
        res = faiss.StandardGpuResources()
        cloner_options = faiss.GpuClonerOptions()
        index = faiss.index_cpu_to_gpu(res, gpu, index, cloner_options)

    index.train(item_vectors)
    index.add(item_vectors)

    return index


class AllItems(RecsPipelineComponent):
    def __init__(self, total_items):
        self.total_items = total_items

    def run(self, user_recs):
        neighbors = th.arange(0, self.total_items)

        user_recs.candidates = neighbors.flatten()
        return user_recs


class RandomCandidates(RecsPipelineComponent):
    def __init__(self, total_items, num_candidates):
        self.total_items = total_items
        self.num_candidates = num_candidates

    def run(self, user_recs):
        neighbors = th.randint(0, self.total_items, (self.num_candidates,))

        user_recs.candidates = neighbors.flatten()
        return user_recs


class IdealizedANNSearch(RecsPipelineComponent):
    def __init__(self, eval_dataset, index, total_items, num_candidates, overfetch=1.2):
        self.eval_dataset = eval_dataset
        self.index = index
        self.total_items = total_items
        self.num_candidates = num_candidates
        self.overfetch = overfetch

    # TODO: Move num_candidates and num_recs into user_recs
    def run(self, user_recs):
        num_candidates = int(self.overfetch * self.num_candidates)

        eval_interactions = self.eval_dataset[user_recs.user_id][
            "interactions"
        ].coalesce()
        eval_item_ids = eval_interactions.indices()[1]

        if not user_recs.user_embeddings.isnan().any():
            _, neighbor_indices = self.index.search(
                np.array(user_recs.user_embeddings.cpu()), num_candidates
            )
        else:
            neighbor_indices = th.randint(self.total_items, (num_candidates,))

        neighbor_set = set(neighbor_indices.flatten().tolist())
        eval_set = set(eval_item_ids.flatten().tolist())
        padding_set = neighbor_set - eval_set

        ideal_indices = list(eval_set) + list(padding_set)
        candidates = th.tensor(ideal_indices).flatten()

        user_recs.candidates = candidates[: self.num_candidates]
        return user_recs


class ANNSearch(RecsPipelineComponent):
    def __init__(self, index, total_items, num_candidates, overfetch=1.2):
        self.index = index
        self.total_items = total_items
        self.num_candidates = num_candidates
        self.overfetch = overfetch

    def run(self, user_recs):
        num_candidates = int(self.overfetch * self.num_candidates)

        # TODO: Filter out any embeddings containing nan, then check how many are left
        if not user_recs.user_embeddings.isnan().any():
            candidates_per = (
                int(num_candidates / user_recs.user_embeddings.shape[0]) + 1
            )
            _, neighbor_indices = self.index.search(
                np.array(user_recs.user_embeddings.cpu()), candidates_per
            )
            neighbors = th.tensor(neighbor_indices).flatten().unique(sorted=False)
        else:
            # TODO: Extract padding to a separate stage
            neighbors = th.randint(self.total_items, (num_candidates,))

        user_recs.candidates = neighbors[: self.num_candidates]
        return user_recs
