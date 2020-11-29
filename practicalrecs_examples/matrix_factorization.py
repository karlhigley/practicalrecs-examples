import torch as th

from .pipeline import RecsPipelineStage


class UserEmbeddingFetcher(RecsPipelineStage):
    def __init__(self, model):
        self.model = model

    def run(self, user_recs):
        user_emb = self.model.user_embeddings.weight[user_recs.user_id]

        user_recs.user_embeddings = user_emb.unsqueeze(dim=0)
        return user_recs


class ItemEmbeddingsFetcher(RecsPipelineStage):
    def __init__(self, model):
        self.model = model

    def run(self, user_recs):
        if len(user_recs.item_ids) > 0:
            item_embs = self.model.item_embeddings.weight[user_recs.item_ids]
        else:
            total_items = self.model.item_embeddings.weight.shape[0]
            item_ids = th.randint(total_items, (1,))
            item_embs = self.model.item_embeddings.weight[item_ids]

        user_recs.item_embeddings = item_embs
        return user_recs


class UserAvgEmbeddingFetcher(RecsPipelineStage):
    def __init__(self, model):
        self.model = model

    def run(self, user_recs):
        item_embs = self.model.item_embeddings.weight[user_recs.item_ids]

        user_recs.user_embeddings = th.mean(item_embs, dim=0).unsqueeze(dim=0)
        return user_recs


class MatrixFactorizationScoring(RecsPipelineStage):
    def __init__(self, model):
        self.model = model

    def run(self, user_recs):
        item_vectors = self.model.item_embeddings.weight.squeeze()
        item_biases = self.model.item_biases.weight.squeeze()

        # TODO: Optimize to only score candidates
        scores = self.model._similarity_scores(
            user_recs.user_embeddings, th.empty((1, 1)), item_vectors, item_biases
        ).flatten()

        masked_scores = th.empty_like(scores).fill_(-float("inf"))
        masked_scores[user_recs.candidates] = scores[user_recs.candidates]

        user_recs.scores = masked_scores
        return user_recs
