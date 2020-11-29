import torch as th

from .pipeline import RecsPipelineStage


class DescendingOrdering(RecsPipelineStage):
    def __init__(self, num_recs):
        self.num_recs = num_recs

    def run(self, user_recs):
        topk_scores, topk_indices = th.topk(user_recs.scores, self.num_recs)

        masked_scores = th.empty_like(user_recs.scores).fill_(-float("inf"))
        masked_scores[topk_indices] = topk_scores

        user_recs.scores = masked_scores
        return user_recs


class DitheredOrdering(RecsPipelineStage):
    def __init__(self, num_recs, epsilon=1.25):
        self.num_recs = num_recs
        self.epsilon = epsilon

    def run(self, user_recs):
        _, topk_indices = th.topk(user_recs.scores, self.num_recs)

        log_ranks = th.log(th.arange(self.num_recs) + 1.0)
        std_dev = th.sqrt(th.log(th.tensor(self.epsilon)))

        dithered_scores = -(log_ranks + th.randn_like(log_ranks) * std_dev)
        dithered_scores = dithered_scores.to(device=user_recs.scores.device)

        masked_scores = th.empty_like(user_recs.scores).fill_(-float("inf"))
        masked_scores[topk_indices] = dithered_scores

        return user_recs