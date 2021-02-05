import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import torch as th


@dataclass
class PipelineConfig:
    num_items: int = None
    num_candidates: int = None
    num_recs: int = None


@dataclass
class PipelineArtifacts:
    # model_name: str = None
    # index_type: str = None
    # filters_name: str = None

    model: Any = None
    index: Any = None
    filters: Any = None


# TODO: firm up the type annotations here
@dataclass
class UserRecs:
    user_id: int = None
    user_embeddings: Any = None
    item_ids: list = field(default_factory=list)
    item_embeddings: Any = None
    candidates: Any = None
    scores: Any = None

    def finalize(self):
        # Clears out intermediate results and preps final results
        self.user_embeddings = None
        self.item_embeddings = None
        self.candidates = self.candidates.cpu()
        self.scores = self.scores.cpu()


# TODO: Make this abstract
class RecsPipelineComponent:
    def run(self, user_recs, artifacts, config):
        raise NotImplementedError


class RecsPipeline:
    # Should know nothing about constructing stages
    def __init__(self, *components, timing=False, caching=False):
        self.components = components
        self.timing = timing
        self.timers = defaultdict(float)
        self.caching = caching
        self.cache = {}

    def recommend(self, user_id, interacted_item_ids, artifacts, config):
        user_recs = UserRecs(user_id=user_id, item_ids=interacted_item_ids)

        for component in self.components:
            if self.timing:
                start = time.perf_counter()
                user_recs = component.run(user_recs, artifacts, config)
                stop = time.perf_counter()
                self.timers[type(component).__name__] += stop - start
            else:
                user_recs = component.run(user_recs, artifacts, config)

        scores = user_recs.scores

        if self.caching:
            self.cache[user_recs.user_id] = user_recs.finalize()

        # TODO: Change this to return the full recs
        return scores

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

    def build(self, overrides=None):
        if overrides is None:
            overrides = RecsPipelineStages()

        components = []
        # TODO: Create an enumeration for stages
        for stage in ["retrieval", "filtering", "scoring", "ordering"]:
            override_components = getattr(overrides, stage)
            default_components = getattr(self.defaults, stage)
            if override_components is not None:
                components.extend(override_components)
            elif default_components and len(default_components) > 0:
                components.extend(default_components)

        return RecsPipeline(*components)
