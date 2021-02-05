import faiss
import numpy as np
import torch as th

from .pipeline import RecsPipelineComponent, UserRecs


def build_ann_index(item_vectors, embedding_dim, index_type, nprobe):
    index = faiss.index_factory(embedding_dim, index_type, faiss.METRIC_INNER_PRODUCT)
    index.nprobe = nprobe

    index.train(item_vectors)
    index.add(item_vectors)

    return index


def build_ann_index_for_model(
    item_vectors, index_type, nprobe=1, use_cuda=True, gpu_res=None, gpu_id=0
):
    dim = len(item_vectors[0])
    index = build_ann_index(item_vectors, dim, index_type, nprobe)

    if th.cuda.is_available() and use_cuda and gpu_res:
        cloner_options = faiss.GpuClonerOptions()
        index = faiss.index_cpu_to_gpu(gpu_res, gpu_id, index, cloner_options)

    return index


class AllItems(RecsPipelineComponent):
    def run(self, user_recs, artifacts, config):
        neighbors = th.arange(0, config.num_items)

        user_recs.candidates = neighbors.flatten()
        return user_recs


class RandomCandidates(RecsPipelineComponent):
    def run(self, user_recs, artifacts, config):
        neighbors = th.randint(0, config.num_items, (config.num_candidates,))

        user_recs.candidates = neighbors.flatten()
        return user_recs


class IdealizedANNSearch(RecsPipelineComponent):
    def __init__(self, eval_dataset, index, overfetch=1.2):
        self.eval_dataset = eval_dataset
        self.index = index
        self.overfetch = overfetch

    def run(self, user_recs, artifacts, config):
        num_candidates = int(self.overfetch * config.num_candidates)

        eval_interactions = self.eval_dataset[user_recs.user_id][
            "interactions"
        ].coalesce()
        eval_item_ids = eval_interactions.indices()[1]

        if not user_recs.user_embeddings.isnan().any():
            _, neighbor_indices = self.index.search(
                np.array(user_recs.user_embeddings.cpu()), num_candidates
            )
        else:
            neighbor_indices = th.randint(config.num_items, (num_candidates,))

        neighbor_set = set(neighbor_indices.flatten().tolist())
        eval_set = set(eval_item_ids.flatten().tolist())
        padding_set = neighbor_set - eval_set

        ideal_indices = list(eval_set) + list(padding_set)
        candidates = th.tensor(ideal_indices).flatten()

        user_recs.candidates = candidates[:num_candidates]
        return user_recs


class ANNSearch(RecsPipelineComponent):
    def __init__(self, index, overfetch=1.2):
        self.index = index
        self.overfetch = overfetch

    def run(self, user_recs, artifacts, config):
        adj_num_candidates = int(self.overfetch * config.num_candidates)

        # TODO: Filter out any embeddings containing nan, then check how many are left
        if not user_recs.user_embeddings.isnan().any():
            candidates_per = (
                int(adj_num_candidates / user_recs.user_embeddings.shape[0]) + 1
            )
            _, neighbor_indices = self.index.search(
                user_recs.user_embeddings.cpu().detach().numpy(), candidates_per
            )
            neighbors = th.tensor(neighbor_indices).flatten().unique(sorted=False)
        else:
            # TODO: Extract padding to a separate stage
            neighbors = th.randint(config.num_items, (adj_num_candidates,))

        user_recs.candidates = neighbors[:adj_num_candidates]
        return user_recs
