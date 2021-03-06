import faiss
import numpy as np
import pybloomfilter as pbf
import torch as th
from practicalrecs_examples.ann_search import AllItems
from practicalrecs_examples.filtering import BloomFilter, IdealizedFilter
from practicalrecs_examples.matrix_factorization import ItemEmbeddingsFetcher
from practicalrecs_examples.pipeline import UserRecs
from pytorch_lightning import Trainer, seed_everything
from torch_factorization_models.implicit_mf import ImplicitMatrixFactorization
from torch_factorization_models.movielens import MovielensDataModule


def test_idealized_filter_removes_training_set_items(trained_model, sim_eval_dataset):
    user_id = 0

    all_items = AllItems(trained_model.hparams.num_items)
    user_recs = all_items.run(UserRecs(user_id=user_id))

    length_before = len(user_recs.candidates)

    stage = IdealizedFilter(sim_eval_dataset)
    user_recs = stage.run(user_recs)

    length_after = len(user_recs.candidates)

    assert length_after > length_before


def test_bloom_filter_removes_training_set_items(trained_model, sim_eval_dataset):
    user_id = 0

    interactions = sim_eval_dataset[user_id]["interactions"].coalesce()
    item_ids = list(interactions.indices()[1].numpy())

    # TODO: See if these assertions can be moved to test the simulated dataset
    assert len(item_ids) > 0
    assert item_ids[0] < trained_model.hparams.num_items
    assert type(item_ids) == list

    bloom_filter = pbf.BloomFilter(100, 0.1)
    bloom_filter.update(item_ids)

    all_items = AllItems(trained_model.hparams.num_items)
    user_recs = all_items.run(UserRecs(user_id=user_id))

    length_before = len(user_recs.candidates)

    stage = BloomFilter({0: bloom_filter})
    user_recs = stage.run(user_recs)

    length_after = len(user_recs.candidates)

    assert length_after > length_before