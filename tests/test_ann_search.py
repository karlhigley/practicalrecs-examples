from argparse import ArgumentParser

import faiss
import numpy as np
import torch as th
from practicalrecs_examples.candidates import (
    AllItemsAsCandidates,
    ANNSearch,
    IdealizedANNSearch,
    RandomCandidates,
)
from practicalrecs_examples.artifacts import ArtifactSet
from practicalrecs_examples.matrix_factorization import ItemEmbeddingsFetcher
from practicalrecs_examples.pipeline import PipelineConfig, UserRecs
from pytorch_lightning import Trainer, seed_everything
from torch_factorization_models.implicit_mf import ImplicitMatrixFactorization
from torch_factorization_models.movielens import MovielensDataModule


def test_all_items_returns_all_items(trained_model):
    artifacts = ArtifactSet(model=trained_model)
    config = PipelineConfig(
        num_items=trained_model.hparams.num_items, num_candidates=250
    )

    stage = AllItemsAsCandidates()
    user_recs = stage.run(UserRecs(), artifacts, config)

    assert len(user_recs.candidates) == trained_model.hparams.num_items
    assert len(set(user_recs.candidates)) == trained_model.hparams.num_items


def test_random_candidates_returns_num_candidates(trained_model):
    artifacts = ArtifactSet(model=trained_model)
    config = PipelineConfig(
        num_items=trained_model.hparams.num_items, num_candidates=250
    )

    stage = RandomCandidates()
    user_recs = stage.run(UserRecs(), artifacts, config)

    assert len(user_recs.candidates) == config.num_candidates
    assert len(set(user_recs.candidates)) == config.num_candidates


def test_ann_search_returns_num_candidates(trained_model, exact_ann_index):
    artifacts = ArtifactSet(model=trained_model, index=exact_ann_index)
    config = PipelineConfig(
        num_items=trained_model.hparams.num_items, num_candidates=250
    )

    user_embeddings = trained_model.user_embeddings.weight.data[:1000]
    stage = ANNSearch()

    for user_id, user_embedding in enumerate(user_embeddings):
        user_recs = UserRecs(
            user_id=user_id, user_embeddings=user_embedding.unsqueeze(0)
        )
        user_recs = stage.run(user_recs, artifacts, config)

        assert len(user_recs.candidates) == config.num_candidates
        assert len(set(user_recs.candidates)) == config.num_candidates


def test_idealized_ann_search_returns_num_candidates(
    trained_model, exact_ann_index, sim_eval_dataset
):
    artifacts = ArtifactSet(model=trained_model, index=exact_ann_index)
    config = PipelineConfig(
        num_items=trained_model.hparams.num_items, num_candidates=250
    )

    user_embeddings = trained_model.user_embeddings.weight.data[:1000]
    stage = IdealizedANNSearch(sim_eval_dataset)

    for user_id, user_embedding in enumerate(user_embeddings):
        user_recs = UserRecs(
            user_id=user_id, user_embeddings=user_embedding.unsqueeze(0)
        )
        user_recs = stage.run(user_recs, artifacts, config)

        assert len(user_recs.candidates) == config.num_candidates
        assert len(set(user_recs.candidates)) == config.num_candidates


def test_ann_search_with_multiple_embeddings_returns_num_candidates(
    trained_model, exact_ann_index
):
    artifacts = ArtifactSet(model=trained_model, index=exact_ann_index)
    config = PipelineConfig(
        num_items=trained_model.hparams.num_items, num_candidates=250
    )

    item_emb_fetcher = ItemEmbeddingsFetcher()
    user_embeddings = trained_model.user_embeddings.weight.data[:1000]
    stage = ANNSearch()

    for user_id, user_embedding in enumerate(user_embeddings):
        user_recs = UserRecs(
            user_id=user_id, user_embeddings=user_embedding.unsqueeze(0)
        )

        user_recs = item_emb_fetcher.run(user_recs, artifacts, config)
        user_recs = stage.run(user_recs, artifacts, config)

        assert len(user_recs.candidates) == config.num_candidates
        assert len(set(user_recs.candidates)) == config.num_candidates
