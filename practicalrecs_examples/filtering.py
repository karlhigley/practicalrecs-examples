import torch as th

from .pipeline import RecsPipelineStage


class BloomFilter(RecsPipelineStage):
    def __init__(self, filters):
        # TODO: Construct the filters here?
        self.filters = filters

    def run(self, user_recs):
        bloom = self.filters[user_recs.user_id]
        filtered = list(filter(lambda c: c not in bloom, user_recs.candidates.numpy()))

        user_recs.candidates = filtered
        return user_recs


class CandidatePadding(RecsPipelineStage):
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
