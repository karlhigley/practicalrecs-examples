import torch as th

from .pipeline import RecsPipelineComponent


class DescendingOrdering(RecsPipelineComponent):
    def run(self, user_recs, artifacts, config):
        topk_scores, topk_indices = th.topk(user_recs.scores, config.num_recs)

        masked_scores = th.empty_like(user_recs.scores).fill_(-float("inf"))
        masked_scores[topk_indices] = topk_scores

        user_recs.scores = masked_scores
        return user_recs


class DitheredOrdering(RecsPipelineComponent):
    def __init__(self, epsilon=1.5):
        self.epsilon = epsilon
        self.std_dev = th.sqrt(th.log(th.tensor(self.epsilon)))

    def run(self, user_recs, artifacts, config):
        log_ranks = th.log(th.arange(config.num_candidates) + 1.0)

        dithered_scores = -(log_ranks + th.randn_like(log_ranks) * self.std_dev)
        dithered_scores = dithered_scores.to(device=user_recs.scores.device)

        candidate_scores = th.full_like(user_recs.scores, -float("inf"))
        _, candidate_indices = th.topk(user_recs.scores, config.num_candidates)
        candidate_scores[candidate_indices] = dithered_scores

        rec_scores = th.full_like(user_recs.scores, -float("inf"))
        _, rec_indices = th.topk(candidate_scores, config.num_recs)
        rec_scores[rec_indices] = candidate_scores[rec_indices]

        user_recs.scores = rec_scores
        return user_recs