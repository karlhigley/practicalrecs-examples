import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import torch as th


class RecsPipelineComponent:
    def run(self):
        raise NotImplementedError


# TODO: firm up the type annotations here
@dataclass
class UserRecs:
    user_id: int = None
    user_embeddings: Any = None
    item_ids: list = field(default_factory=list)
    item_embeddings: Any = None
    candidates: Any = None
    scores: Any = None


class RecsPipeline:
    def __init__(self, *components):
        self.components = components
        self.timers = defaultdict(float)

    def recommend(self, user_id, interacted_item_ids):
        user_recs = UserRecs(user_id=user_id, item_ids=interacted_item_ids)

        for component in self.components:
            start = time.perf_counter()
            user_recs = component.run(user_recs)
            stop = time.perf_counter()

            self.timers[type(component).__name__] += stop - start

        return user_recs.scores

    def reset_timers(self):
        self.timers = defaultdict(float)


@dataclass
class RecsPipelineStages:
    retrieval: list = None
    filtering: list = None
    scoring: list = None
    ordering: list = None


class RecsPipelineBuilder:
    def __init__(self, defaults=RecsPipelineStages()):
        self.defaults = defaults

    def build(self, overrides=RecsPipelineStages()):
        components = []
        for stage in ["retrieval", "filtering", "scoring", "ordering"]:
            override_components = getattr(overrides, stage)
            default_components = getattr(self.defaults, stage)
            if override_components is not None:
                components.extend(override_components)
            elif default_components and len(default_components) > 0:
                components.extend(default_components)

        return RecsPipeline(*components)
