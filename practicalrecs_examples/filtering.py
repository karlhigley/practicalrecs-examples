import pybloomfilter as pbf
import torch as th

from .pipeline import RecsPipelineComponent


def build_bloom_filters(dataset, expected_items, fp_rate):
    filters = {}

    for user in dataset:
        user_id = user["user_ids"].squeeze().cpu().item()
        interactions = user["interactions"].coalesce()
        item_ids = list(interactions.indices()[1].cpu().numpy())

        bloom = pbf.BloomFilter(expected_items, fp_rate)
        bloom.update(item_ids)

        filters[user_id] = bloom

    return filters


class IdealizedFilter(RecsPipelineComponent):
    def __init__(self, train_dataset):
        self.train_dataset = train_dataset

    def run(self, user_recs):
        train_interactions = self.train_dataset[user_recs.user_id][
            "interactions"
        ].coalesce()
        train_item_ids = train_interactions.indices()[1]

        candidate_set = set(user_recs.candidates.flatten().tolist())
        train_set = set(train_item_ids.flatten().tolist())

        filtered_indices = list(candidate_set - train_set)

        user_recs.candidates = th.tensor(filtered_indices).flatten()
        return user_recs


class BloomFilter(RecsPipelineComponent):
    def __init__(self, filters):
        self.filters = filters

    def run(self, user_recs):
        bloom = self.filters[user_recs.user_id]
        filtered = list(filter(lambda c: c not in bloom, user_recs.candidates.numpy()))

        user_recs.candidates = filtered
        return user_recs


class CandidatePadding(RecsPipelineComponent):
    def __init__(self, total_items, num_candidates):
        self.total_items = total_items
        self.num_candidates = num_candidates

    def run(self, user_recs):
        candidates = th.tensor(user_recs.candidates, dtype=th.long)

        # Normalize number of scored candidates to num_candidates
        if candidates.shape[0] > self.num_candidates:
            candidates = candidates[: self.num_candidates]
        elif candidates.shape[0] < self.num_candidates:
            padding_size = self.num_candidates - candidates.shape[0]
            candidates = th.cat(
                [candidates, th.randint(self.total_items, (padding_size,))]
            )

        user_recs.candidates = candidates
        return user_recs
