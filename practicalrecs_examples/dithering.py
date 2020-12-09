import torch as th

from .pipeline import RecsPipelineComponent


class DescendingOrdering(RecsPipelineComponent):
    def __init__(self, num_recs):
        self.num_recs = num_recs

    def run(self, user_recs):
        topk_scores, topk_indices = th.topk(user_recs.scores, self.num_recs)

        masked_scores = th.empty_like(user_recs.scores).fill_(-float("inf"))
        masked_scores[topk_indices] = topk_scores

        user_recs.scores = masked_scores
        return user_recs


class DitheredOrdering(RecsPipelineComponent):
    def __init__(self, num_recs, epsilon=1.25):
        self.num_recs = num_recs
        self.epsilon = epsilon

        self.log_ranks = th.log(th.arange(self.num_recs) + 1.0)
        self.std_dev = th.sqrt(th.log(th.tensor(self.epsilon)))

    def run(self, user_recs):
        _, topk_indices = th.topk(user_recs.scores, self.num_recs)

        dithered_scores = -(
            self.log_ranks + th.randn_like(self.log_ranks) * self.std_dev
        )
        dithered_scores = dithered_scores.to(device=user_recs.scores.device)

        masked_scores = th.empty_like(user_recs.scores).fill_(-float("inf"))
        masked_scores[topk_indices] = dithered_scores

        user_recs.scores = masked_scores
        return user_recs