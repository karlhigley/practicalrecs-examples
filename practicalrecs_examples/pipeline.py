import time
from dataclasses import dataclass
from typing import Any

import torch as th


class RecsPipelineStage:
    def run(self):
        raise NotImplementedError


@dataclass
class UserRecs:
    user_id: int = None
    user_embeddings: Any = None
    item_ids: list = None
    item_embeddings: Any = None
    candidates: Any = None
    scores: Any = None


class RecsPipeline:
    def __init__(self, *stages):
        self.stages = stages
        self.timers = {}

        for stage in self.stages:
            self.timers[type(stage).__name__] = 0.0

    def recommend(self, user_id, interacted_item_ids):
        user_recs = UserRecs(user_id=user_id, item_ids=interacted_item_ids)

        for stage in self.stages:
            start = time.perf_counter()
            user_recs = stage.run(user_recs)
            stop = time.perf_counter()

            self.timers[type(stage).__name__] += stop - start

        return user_recs.scores

    def reset(self):
        self.timers = {}
