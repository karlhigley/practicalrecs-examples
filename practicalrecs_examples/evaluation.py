from collections import defaultdict

import faiss
import numpy as np
import torch as th

from practicalrecs_examples.ann_search import build_ann_index
from practicalrecs_examples.artifacts import ArtifactRegistry
from practicalrecs_examples.filtering import build_bloom_filters
from practicalrecs_examples.matrix_factorization import load_model
from practicalrecs_examples.metrics import MetricsCalculator
from practicalrecs_examples.pipeline import (
    PipelineConfig,
    PipelineRegistry,
    RecsPipelineTemplate,
)


class EvaluationHarness:
    def __init__(
        self,
        pipelines=None,
        artifacts=None,
        calculator=None,
        num_candidates=500,
        num_recs=100,
        use_cuda=True,
        gpu_id=0,
    ):
        # TODO: Write code/function to create idealized version of stages with each model
        self.pipelines = pipelines or PipelineRegistry()
        self.artifacts = artifacts or ArtifactRegistry(use_cuda, gpu_id)
        self.calculator = calculator or MetricsCalculator(to_compute=["recall", "ndcg"])

        self.num_candidates = num_candidates
        self.num_recs = num_recs

        # TODO: Extract dataloader stuff to a separate class?

    def evaluate_model(self, model_name, val_dataloader):
        model = self.artifacts.models[model_name]

        metrics = self.calculator.compute_metrics(
            model.similarity_to_users, val_dataloader, self.num_recs
        )

        # TODO: Move metrics caching into calculator
        self.calculator.metrics[model_name]["model"] = metrics

        return metrics

    def evaluate_pipeline(
        self,
        *,
        pipeline,
        model,
        index,
        filters,
        train,
        val,
    ):
        # TODO: Figure out how to resolve name confusions here
        # TODO: Add a retrieve method to the PipelineRegistry?
        pipeline_ = self.pipelines.pipelines[pipeline]
        artifacts = self.artifacts.retrieve(
            model_name=model, index_type=index, filters_name=filters
        )

        num_items = artifacts.model.hparams.num_items

        def pipeline_predict(user_ids):
            user_scores = []
            for user_id in user_ids:
                user_id = int(user_id.cpu().item())

                interactions = train.dataset[user_id]["interactions"].coalesce()
                item_ids = interactions.indices()[1]

                config = PipelineConfig(num_items, self.num_candidates, self.num_recs)
                user_recs = pipeline_.recommend(user_id, item_ids, artifacts, config)

                user_scores.append(user_recs.scores)

            return th.stack(user_scores)

        metrics = self.calculator.compute_metrics(
            pipeline_predict,
            val,
            self.num_recs,
        )

        # TODO: Move metrics caching into calculator
        self.calculator.metrics[model][pipeline] = metrics

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

        # TODO: Extract filter registry too?

        # def sweep_filter_params(model, builder, capacities, error_rates):
        #     filter_metrics = {}

        #     for error_rate in error_rates:
        #         metrics = []

        #         for capacity in capacities:
        #             filters = self.build_bloom_filters(
        #                 f"cap{capacity}-fp{error_rate}",
        #                 tqdm(train_dataloader.dataset),
        #                 expected_items=capacity,
        #                 fp_rate=error_rate
        #             )

        #             stages = RecsPipelineStages(
        #                 filtering = [
        #                     BloomFilter(filters),
        #                     CandidatePadding(),
        #                 ]
        #             )

        #             pipeline = builder.build(overrides=stages)

        #             m = model.compute_validation_metrics(
        #                 build_prediction_fn(pipeline, train_dataloader),
        #                 tqdm(val_dataloader),
        #                 harness.num_recs
        #             )
        #             metrics.append((capacity, m))
        #         filter_metrics[error_rate] = metrics

        #     return filter_metrics

        # import math

        # def compute_bytes(capacity, error_rate):
        #     num_hashes = max(math.floor(math.log2(1 / error_rate)), 1)
        #     bits_per_hash = math.ceil(
        #                 capacity * abs(math.log(error_rate)) /
        #                 (num_hashes * (math.log(2) ** 2)))
        #     num_bits = max(num_hashes * bits_per_hash,128)
        #     return num_bits//8

        # def compute_kbytes(capacity, error_rate):
        #     return compute_bytes(capacity, error_rate)/1024

        # TODO: Extract filtering registry thingy too?

        # import copy

        # ordering_epsilons = list(float_range(1.0,3.75,0.25))

        # dithering_metrics = []

        # for epsilon in ordering_epsilons:
        #     harness.create_pipeline(
        #         "warp", f"improved-ordering-{epsilon}", builder_name="improved-filtering",
        #         stages = RecsPipelineStages(
        #             ordering = [
        #                 DitheredOrdering(epsilon=epsilon),
        #             ]
        #         )
        #     )

        #     dithering_pipeline = harness.pipelines["warp"][f"improved-ordering-{epsilon}"]
        #     dithering_pipeline.caching = True

        #     m = harness.evaluate_pipeline(
        #         "warp", f"improved-ordering-{epsilon}", train_dataloader, tqdm(val_dataloader)
        #     )

        #     initial_results = dithering_pipeline.cache
        #     dithering_pipeline.cache = {}

        #     for user_id in tqdm(initial_results.keys()):
        #         user_recs = copy.deepcopy(initial_results[user_id])
        #         user_recs = dithering_pipeline.components[-1].run(user_recs)
        #         dithering_pipeline.cache[user_id] = user_recs

        #     rerun_results = dithering_pipeline.cache

        #     dithering_pipeline.caching = False
        #     dithering_pipeline.cache = {}

        #     overlaps = []

        #     for user_id in tqdm(initial_results.keys()):
        #         _, initial_indices = th.topk(initial_results[user_id].scores, harness.num_recs)
        #         _, rerun_indices = th.topk(rerun_results[user_id].scores, harness.num_recs)

        #         intersection = len(np.intersect1d(initial_indices, rerun_indices))
        #         overlaps.append(intersection)

        #     initial_results = None
        #     rerun_results = None

        #     m['median_overlap'] = np.median(np.array(overlaps))
        #     m['mean_overlap'] = np.mean(np.array(overlaps))
        #     m['min_overlap'] = np.min(np.array(overlaps))
        #     m['max_overlap'] = np.max(np.array(overlaps))

        # dithering_metrics.append((epsilon, m))
