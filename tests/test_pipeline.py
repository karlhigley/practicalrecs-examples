import pytest
import torch as th
from practicalrecs_examples.pipeline import RecsPipeline


def retrieve_candidates(user_id):
    interactions = train_dataset[user_id]["interactions"].coalesce()
    item_ids = interactions.indices()[1]


def batch_recommend(recs_fn):
    def recommend(user_ids, num_items):
        user_scores = [recs_fn(user_id) for user_id in user_ids]

        return th.stack(user_scores)

    return recommend


def test_recs_pipeline():
    pipeline = RecsPipeline(
        retrieve_candidates, filter_candidates, score_candidates, dither_scores
    )

    pipeline_fn = pipeline.fn

    # TODO: Load a model
    # TODO: Load datamodule

    pipeline_metrics = model.compute_validation_metrics(
        val_dataloader, batch_recommend(pipeline.fn)
    )
