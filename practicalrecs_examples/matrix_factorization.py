from argparse import ArgumentParser

import torch as th
from pytorch_lightning import Trainer
from torch_factorization_models.implicit_mf import ImplicitMatrixFactorization

from .pipeline import RecsPipelineComponent


def load_model(path):
    state_dict = th.load("../models/mf_example.pt")
    del state_dict["preprocessor"]
    state_dict["global_bias_idx"] = th.LongTensor([0])

    parser = ArgumentParser(add_help=False)
    parser = Trainer.add_argparse_args(parser)
    parser = ImplicitMatrixFactorization.add_model_specific_args(parser)

    args = parser.parse_args(args=[])
    args.num_users = state_dict["user_embeddings.weight"].shape[0]
    args.num_items = state_dict["item_embeddings.weight"].shape[0]
    args.embedding_dim = state_dict["user_embeddings.weight"].shape[1]

    # TODO: Move this elsewhere
    args.eval_cutoff = th.tensor([100])

    model = ImplicitMatrixFactorization(args)
    model.load_state_dict(state_dict)

    return model


class UserEmbeddingFetcher(RecsPipelineComponent):
    def __init__(self, model):
        self.model = model

    def run(self, user_recs):
        user_emb = self.model.user_embeddings.weight[user_recs.user_id]

        user_recs.user_embeddings = user_emb.unsqueeze(dim=0)
        return user_recs


class ItemEmbeddingsFetcher(RecsPipelineComponent):
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


class UserAvgEmbeddingFetcher(RecsPipelineComponent):
    def __init__(self, model):
        self.model = model

    def run(self, user_recs):
        item_embs = self.model.item_embeddings.weight[user_recs.item_ids]

        user_recs.user_embeddings = th.mean(item_embs, dim=0).unsqueeze(dim=0)
        return user_recs


class UseItemEmbeddingsAsUserEmbeddings(RecsPipelineComponent):
    def __init__(self, append=False):
        self.append = append

    def run(self, user_recs):
        if self.append:
            user_recs.user_embeddings = th.cat(
                [user_recs.user_embeddings, user_recs.item_embeddings]
            )
        else:
            user_recs.user_embeddings = user_recs.item_embeddings
        return user_recs


class IdealizedMatrixFactorizationScoring(RecsPipelineComponent):
    def __init__(self, model, eval_dataset):
        self.model = model
        self.eval_dataset = eval_dataset

    def run(self, user_recs):
        item_vectors = self.model.item_embeddings.weight.squeeze()
        item_biases = self.model.item_biases.weight.squeeze()

        # TODO: Optimize to only score candidates
        raw_scores = self.model._similarity_scores(
            user_recs.user_embeddings, th.empty((1, 1)), item_vectors, item_biases
        ).flatten()

        val_interactions = self.eval_dataset[user_recs.user_id][
            "interactions"
        ].coalesce()
        val_item_ids = val_interactions.indices()[1]
        boosted_scores = raw_scores.clone().detach()

        # TODO: Keep score distribution but rearrange item ids
        boosted_scores[val_item_ids] += 10.0

        masked_scores = th.empty_like(boosted_scores).fill_(-float("inf"))
        masked_scores[user_recs.candidates] = boosted_scores[user_recs.candidates]

        user_recs.scores = masked_scores
        return user_recs


class MatrixFactorizationScoring(RecsPipelineComponent):
    def __init__(self, model):
        self.model = model

    def run(self, user_recs):
        item_ids = user_recs.candidates

        item_vectors = self.model.item_embeddings.weight[item_ids].squeeze()
        item_biases = self.model.item_biases.weight[item_ids].squeeze()

        scores = self.model._similarity_scores(
            user_recs.user_embeddings, th.empty((1, 1)), item_vectors, item_biases
        ).flatten()

        masked_scores = th.empty(
            (self.model.hparams.num_items,), dtype=scores.dtype, device=scores.device
        ).fill_(-float("inf"))
        masked_scores[user_recs.candidates] = scores

        user_recs.scores = masked_scores
        return user_recs
