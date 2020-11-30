from argparse import ArgumentParser

import faiss
import numpy as np
import pytest
import torch as th
from practicalrecs_examples.ann_search import ANNSearch, IdealizedANNSearch
from practicalrecs_examples.pipeline import UserRecs
from pytorch_lightning import Trainer, seed_everything
from torch_factorization_models.implicit_mf import ImplicitMatrixFactorization
from torch_factorization_models.movielens import MovielensDataModule


@pytest.fixture(scope="package")
def trained_model():
    parser = ArgumentParser(add_help=False)
    parser = Trainer.add_argparse_args(parser)
    parser = ImplicitMatrixFactorization.add_model_specific_args(parser)

    args = parser.parse_args(args=[])
    args.num_users = 138287
    args.num_items = 20720
    args.embedding_dim = 32
    args.eval_cutoff = th.tensor([100])

    model = ImplicitMatrixFactorization(args)

    state_dict = th.load("./practicalrecs_examples/models/mf_example.pt")
    del state_dict["preprocessor"]
    state_dict["global_bias_idx"] = th.LongTensor([0])

    model.load_state_dict(state_dict)

    return model


@pytest.fixture(scope="package")
def exact_ann_index(trained_model):
    dim = trained_model.hparams.embedding_dim
    item_vectors = np.array(trained_model.item_embeddings.weight.data)

    res = faiss.StandardGpuResources()

    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = 0

    exact_index = faiss.GpuIndexFlatIP(res, dim, flat_config)
    exact_index.add(item_vectors)

    return exact_index


class SimulatedEvalDataset:
    def __init__(self, num_users, num_items, device):
        self.num_users = num_users
        self.num_items = num_items
        self.device = device

    def __len__(self):
        return self.num_users

    def __getitem__(self, index):
        user_id = int(index)

        return {
            "user_ids": th.tensor([user_id], device=self.device),
            "interactions": self.simulate_interactions(user_id),
        }

    def simulate_interactions(self, user_id):
        num_interactions = th.randint(0, 100, (1,)).squeeze().item()
        item_ids = th.randint(0, self.num_items, (num_interactions,))
        targets = th.empty_like(item_ids).fill_(1.0)

        return self._sparse_vector(
            user_id, item_ids, targets, self.num_users, self.num_items, self.device
        )

    def _sparse_vector(self, user_id, item_ids, targets, num_users, num_items, device):
        item_indices = item_ids.to(dtype=th.int64)
        user_indices = th.empty_like(item_indices, dtype=th.int64).fill_(user_id)
        item_labels = targets.to(dtype=th.float64)

        return th.sparse.FloatTensor(
            th.stack([user_indices, item_indices]), item_labels, (num_users, num_items)
        ).to(device=device)


@pytest.fixture(scope="package")
def sim_eval_dataset(trained_model):
    dataset = SimulatedEvalDataset(
        trained_model.hparams.num_users,
        trained_model.hparams.num_items,
        trained_model.device,
    )

    return dataset
