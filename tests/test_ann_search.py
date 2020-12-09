from argparse import ArgumentParser

import faiss
import numpy as np
import torch as th
from practicalrecs_examples.ann_search import (
    AllItems,
    ANNSearch,
    IdealizedANNSearch,
    RandomCandidates,
)
from practicalrecs_examples.matrix_factorization import ItemEmbeddingsFetcher
from practicalrecs_examples.pipeline import UserRecs
from pytorch_lightning import Trainer, seed_everything
from torch_factorization_models.implicit_mf import ImplicitMatrixFactorization
from torch_factorization_models.movielens import MovielensDataModule


def test_all_items_returns_all_items(trained_model):
    stage = AllItems(trained_model.hparams.num_items)
    user_recs = stage.run(UserRecs())

    assert len(user_recs.candidates) == trained_model.hparams.num_items
    assert len(set(user_recs.candidates)) == trained_model.hparams.num_items


def test_random_candidates_returns_num_candidates(trained_model):
    stage = RandomCandidates(trained_model.hparams.num_items, 250)
    user_recs = stage.run(UserRecs())

    assert len(user_recs.candidates) == stage.num_candidates
    assert len(set(user_recs.candidates)) == stage.num_candidates


def test_ann_search_returns_num_candidates(trained_model, exact_ann_index):
    user_embeddings = trained_model.user_embeddings.weight.data[:1000]
    stage = ANNSearch(exact_ann_index, trained_model.hparams.num_items, 250)

    for user_id, user_embedding in enumerate(user_embeddings):
        user_recs = UserRecs(
            user_id=user_id, user_embeddings=user_embedding.unsqueeze(0)
        )
        user_recs = stage.run(user_recs)

        assert len(user_recs.candidates) == stage.num_candidates
        assert len(set(user_recs.candidates)) == stage.num_candidates


def test_idealized_ann_search_returns_num_candidates(
    trained_model, exact_ann_index, sim_eval_dataset
):
    user_embeddings = trained_model.user_embeddings.weight.data[:1000]
    stage = IdealizedANNSearch(
        sim_eval_dataset, exact_ann_index, trained_model.hparams.num_items, 250
    )

    for user_id, user_embedding in enumerate(user_embeddings):
        user_recs = UserRecs(
            user_id=user_id, user_embeddings=user_embedding.unsqueeze(0)
        )
        user_recs = stage.run(user_recs)

        assert len(user_recs.candidates) == stage.num_candidates
        assert len(set(user_recs.candidates)) == stage.num_candidates


def test_ann_search_with_multiple_embeddings_returns_num_candidates(
    trained_model, exact_ann_index
):
    item_emb_fetcher = ItemEmbeddingsFetcher(trained_model)
    user_embeddings = trained_model.user_embeddings.weight.data[:1000]
    stage = ANNSearch(exact_ann_index, trained_model.hparams.num_items, 250)

    for user_id, user_embedding in enumerate(user_embeddings):
        user_recs = UserRecs(
            user_id=user_id, user_embeddings=user_embedding.unsqueeze(0)
        )

        user_recs = item_emb_fetcher.run(user_recs)
        user_recs = stage.run(user_recs)

        assert len(user_recs.candidates) == stage.num_candidates
        assert len(set(user_recs.candidates)) == stage.num_candidates
