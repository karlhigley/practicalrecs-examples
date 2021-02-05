from collections import defaultdict

import faiss
import numpy as np
import torch as th
from ranking_metrics_torch.cumulative_gain import ndcg_at
from ranking_metrics_torch.precision_recall import precision_at, recall_at

from practicalrecs_examples.ann_search import build_ann_index_for_model
from practicalrecs_examples.filtering import build_bloom_filters
from practicalrecs_examples.matrix_factorization import load_model
from practicalrecs_examples.pipeline import (
    PipelineArtifacts,
    PipelineConfig,
    RecsPipelineBuilder,
)


class EvaluationHarness:
    def __init__(self, num_candidates=500, num_recs=100, use_cuda=True, gpu_id=0):
        self.num_candidates = num_candidates
        self.num_recs = num_recs

        self.use_cuda = use_cuda
        self.gpu_id = gpu_id
        self.gpu_res = None

        self.models = {}
        self.indices = defaultdict(dict)
        self.filters = {}

        # TODO: Write code/function to create idealized version of stages with each model
        self.pipelines = {}
        self.builders = {}
        self.stages = {}

        # TODO: Create a metrics registry?
        self.metrics = defaultdict(dict)

        # TODO: Extract dataloader stuff to a separate class?

    def load_model(self, name, path):
        model = load_model(path)

        if self.use_cuda:
            model = model.cuda(self.gpu_id)

        self.models[name] = model

    # TODO: Use an enumeration for exact/approx types
    def build_ann_indices(
        self, model_name, approx_type="IVF1024,PQ32", approx_nprobe=30
    ):
        if self.use_cuda and not self.gpu_res:
            self.gpu_res = faiss.StandardGpuResources()

        model = self.models[model_name]
        item_vectors = model.item_embeddings.weight.cpu().data.detach().numpy()

        exact_index = build_ann_index_for_model(
            item_vectors,
            "Flat",
            use_cuda=self.use_cuda,
            gpu_res=self.gpu_res,
            gpu_id=self.gpu_id,
        )

        approx_index = build_ann_index_for_model(
            item_vectors,
            approx_type,
            nprobe=approx_nprobe,
            use_cuda=self.use_cuda,
            gpu_res=self.gpu_res,
            gpu_id=self.gpu_id,
        )

        self.indices[model_name]["exact"] = exact_index
        self.indices[model_name]["approx"] = approx_index

    def build_bloom_filters(self, filters_name, *args, **kwargs):
        filters = build_bloom_filters(*args, **kwargs)

        self.filters[filters_name] = filters

        return filters

    def create_builder(self, model_name, builder_name, stages):
        builder = RecsPipelineBuilder(defaults=stages)

        self.builders[builder_name] = builder

        return builder

    def create_pipeline(self, pipeline_name, builder_name=None, stages=None):
        if builder_name is None:
            builder_name = pipeline_name

        if stages is not None:
            self.stages[pipeline_name] = stages

        builder = self.builders[builder_name]

        pipeline = builder.build(overrides=stages)

        self.stages[pipeline_name] = stages
        self.pipelines[pipeline_name] = pipeline

        return pipeline

    def evaluate_model(self, model_name, val_dataloader):
        model = self.models[model_name]

        metrics = self.compute_ranking_metrics(
            model.similarity_to_users, val_dataloader, self.num_recs
        )

        self.metrics[model_name]["model"] = metrics

        return metrics

    def evaluate_pipeline(
        self,
        pipeline_name,
        model_name,
        index_type,
        filters_name,
        train_dataloader,
        val_dataloader,
    ):
        pipeline = self.pipelines[pipeline_name]

        model = self.models[model_name]
        # index = self.indices[model_name][index_type]
        # filters = self.filters[filters_name]

        num_items = model.hparams.num_items

        def pipeline_predict(user_ids):
            user_scores = []
            for user_id in user_ids:
                user_id = int(user_id.cpu().item())

                interactions = train_dataloader.dataset[user_id][
                    "interactions"
                ].coalesce()
                item_ids = interactions.indices()[1]

                config = PipelineConfig(num_items, self.num_candidates, self.num_recs)
                artifacts = PipelineArtifacts(model)
                scores = pipeline.recommend(user_id, item_ids, artifacts, config)

                user_scores.append(scores)

            return th.stack(user_scores)

        metrics = self.compute_ranking_metrics(
            pipeline_predict,
            val_dataloader,
            self.num_recs,
        )

        self.metrics[model_name][pipeline_name] = metrics

        return metrics

    def evaluate_batch(self, batch, batch_idx, prediction_fn, list_cutoff):
        user_ids = batch["user_ids"]
        interactions = batch["interactions"].coalesce()

        # Summing non-overlapping sparse vectors along the batch dim
        # just condenses them into a single vector
        # Then we extract the relevant rows and convert them to a dense vector
        condensed = th.sparse.sum(interactions, 0)
        labels = (
            th.stack([condensed[int(user_id)] for user_id in user_ids])
            .to_dense()
            .to(dtype=th.float64)
        )

        # Score all the items for each user in the batch
        predictions = prediction_fn(th.unique(user_ids)).to(dtype=th.float64)

        # Compute per-user metrics
        batch_precisions = precision_at(list_cutoff, predictions, labels)
        batch_recalls = recall_at(list_cutoff, predictions, labels)
        batch_ndcgs = ndcg_at(list_cutoff, predictions, labels)

        metrics = {
            "precision": batch_precisions.squeeze(),
            "recall": batch_recalls.squeeze(),
            "ndcg": batch_ndcgs.squeeze(),
        }
        return metrics

    def compute_ranking_metrics(self, prediction_fn, dataloader, list_cutoff):
        outputs = []

        list_cutoff = th.tensor([list_cutoff]).flatten()

        for batch_idx, batch in enumerate(dataloader):
            output = self.evaluate_batch(batch, batch_idx, prediction_fn, list_cutoff)
            outputs.append(output)

        precisions = th.cat([batch["precision"].flatten() for batch in outputs])
        recalls = th.cat([batch["recall"].flatten() for batch in outputs])
        ndcgs = th.cat([batch["ndcg"].flatten() for batch in outputs])

        # Only include users that have relevant items in the average metrics
        metrics = {
            "precision": precisions[~th.isnan(precisions)].mean(),
            "recall": recalls[~th.isnan(recalls)].mean(),
            "ndcg": ndcgs[~th.isnan(ndcgs)].mean(),
        }

        return metrics

    # TODO: Figure out how to create idealized stages this can depend on
    # def evaluate_idealized_stages(
    #     self, model_name, base_name, train_dataloader, val_dataloader
    # ):
    #     for stage_name in ["retrieval", "filtering", "scoring", "ordering"]:
    #         ideal_name = f"ideal-{stage_name}"
    #         combined_name = f"{base_name}-with-{ideal_name}"

    #         self.pipelines[model_name][combined_name] = self.builders[model_name][
    #             base_name
    #         ].build(overrides=self.stages[model_name][ideal_name])

    #         self.metrics[model_name][combined_name] = self.models[
    #             model_name
    #         ].compute_ranking_metrics(
    #             build_prediction_fn(
    #                 self.pipelines[model_name][combined_name], train_dataloader
    #             ),
    #             val_dataloader,
    #             num_recs,
    #         )

    #         print(f"With idealized {stage_name}:")
    #         print_metrics(self.metrics[model_name][combined_name])
