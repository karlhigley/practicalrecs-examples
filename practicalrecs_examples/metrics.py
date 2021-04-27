from collections import defaultdict
from enum import Enum

import torch as th
from ranking_metrics_torch.cumulative_gain import ndcg_at
from ranking_metrics_torch.precision_recall import precision_at, recall_at


metric_fns = {
    "precision": precision_at,
    "recall": recall_at,
    "ndcg": ndcg_at,
}


class MetricsCalculator:
    def __init__(self, to_compute=None):
        self.metrics = defaultdict(dict)
        self.to_compute = to_compute or []

    def _compute_batch_metrics(self, batch, batch_idx, prediction_fn, list_cutoff):
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
        metrics = {}
        for metric_name in self.to_compute:
            metric_fn = metric_fns[metric_name]
            result = metric_fn(list_cutoff, predictions, labels)
            metrics[metric_name] = result.squeeze()

        return metrics

    def compute_metrics(
        self, model_name, pipeline_name, prediction_fn, dataloader, list_cutoff
    ):
        outputs = []

        list_cutoff = th.tensor([list_cutoff]).flatten()

        for batch_idx, batch in enumerate(dataloader):
            output = self._compute_batch_metrics(
                batch, batch_idx, prediction_fn, list_cutoff
            )
            outputs.append(output)

        metrics = {}

        for metric_name in self.to_compute:
            flattened = th.cat([batch[metric_name].flatten() for batch in outputs])
            metrics[metric_name] = flattened[~th.isnan(flattened)].mean()

        self.metrics[model_name][pipeline_name] = metrics

        return metrics
