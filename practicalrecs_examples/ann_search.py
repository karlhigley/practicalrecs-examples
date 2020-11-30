import numpy as np
import torch as th

from .pipeline import RecsPipelineStage, UserRecs


class IdealizedANNSearch(RecsPipelineStage):
    def __init__(self, eval_dataset, index, total_items, num_candidates, overfetch=1.2):
        self.eval_dataset = eval_dataset
        self.index = index
        self.total_items = total_items
        self.num_candidates = num_candidates
        self.overfetch = overfetch

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


class ANNSearch(RecsPipelineStage):
    def __init__(self, index, total_items, num_candidates, overfetch=1.2):
        self.index = index
        self.total_items = total_items
        self.num_candidates = num_candidates
        self.overfetch = overfetch

    def run(self, user_recs):
        num_candidates = int(self.overfetch * self.num_candidates)

        # TODO: Filter out any embeddings containing nan, then check how many are left
        if not user_recs.user_embeddings.isnan().any():
            _, neighbor_indices = self.index.search(
                np.array(user_recs.user_embeddings.cpu()), num_candidates
            )
            neighbors = th.tensor(neighbor_indices)
        else:
            # TODO: Extract padding to a separate stage
            neighbors = th.randint(self.total_items, (num_candidates,))

        user_recs.candidates = neighbors.flatten().unique()[: self.num_candidates]
        return user_recs
