from argparse import ArgumentParser

import faiss
import numpy as np
import torch as th
from practicalrecs_examples.ann_search import ANNSearch, IdealizedANNSearch
from practicalrecs_examples.pipeline import UserRecs
from pytorch_lightning import Trainer, seed_everything
from torch_factorization_models.implicit_mf import ImplicitMatrixFactorization
from torch_factorization_models.movielens import MovielensDataModule


def test_ann_search_returns_num_candidates_or_less(trained_model, exact_ann_index):
    stage = ANNSearch(exact_ann_index, trained_model.hparams.num_items, 250)

    for user_id, user_embedding in enumerate(trained_model.user_embeddings.weight.data):
        if th.rand((1,)).squeeze().item() > 0.1:
            continue

        user_recs = UserRecs(
            user_id=user_id, user_embeddings=user_embedding.unsqueeze(0)
        )
        user_recs = stage.run(user_recs)
        assert len(user_recs.candidates) <= stage.num_candidates


def test_idealized_ann_search_returns_num_candidates_or_less(
    trained_model, exact_ann_index, sim_eval_dataset
):
    stage = IdealizedANNSearch(
        sim_eval_dataset, exact_ann_index, trained_model.hparams.num_items, 250
    )

    for user_id, user_embedding in enumerate(trained_model.user_embeddings.weight.data):
        if th.rand((1,)).squeeze().item() > 0.1:
            continue

        user_recs = UserRecs(
            user_id=user_id, user_embeddings=user_embedding.unsqueeze(0)
        )
        user_recs = stage.run(user_recs)
        assert len(user_recs.candidates) <= stage.num_candidates
