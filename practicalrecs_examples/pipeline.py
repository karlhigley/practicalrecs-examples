import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch as th


class Stages(Enum):
    RETRIEVAL = "retrieval"
    FILTERING = "filtering"
    SCORING = "scoring"
    ORDERING = "ordering"


@dataclass
class RecsPipelineStages:
    retrieval: list = None
    filtering: list = None
    scoring: list = None
    ordering: list = None


@dataclass
class PipelineConfig:
    num_items: int = None
    num_candidates: int = None
    num_recs: int = None


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
        # Clears out intermediate results (to save space) and preps final results
        self.user_embeddings = None
        self.item_embeddings = None
        self.candidates = self.candidates.cpu()
        self.scores = self.scores.cpu()


class RecsPipelineComponent(ABC):
    @abstractmethod
    def run(self, user_recs, artifacts, config):
        raise NotImplementedError


class RecsPipeline:
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

        if self.caching:
            self.cache[user_recs.user_id] = user_recs.finalize()

        return user_recs

    def reset_timers(self):
        self.timers = defaultdict(float)


class RecsPipelineTemplate:
    def __init__(self, defaults=RecsPipelineStages()):
        self.defaults = defaults

    def build(self, overrides=None):
        if overrides is None:
            overrides = RecsPipelineStages()

        components = []

        for stage in [
            Stages.RETRIEVAL,
            Stages.FILTERING,
            Stages.SCORING,
            Stages.ORDERING,
        ]:
            override_components = getattr(overrides, stage)
            default_components = getattr(self.defaults, stage)
            if override_components is not None:
                components.extend(override_components)
            elif default_components and len(default_components) > 0:
                components.extend(default_components)

        return RecsPipeline(*components)


class PipelineRegistry:
    def __init__(self):
        self.pipelines = {}
        self.templates = {}
        self.stages = {}

    def create_template(self, model_name, template_name, stages):
        template = RecsPipelineTemplate(defaults=stages)

        self.templates[template_name] = template

        return template

    def create_pipeline(self, pipeline_name, template_name=None, stages=None):
        if template_name is None:
            template_name = pipeline_name

        if stages is not None:
            self.stages[pipeline_name] = stages

        template = self.templates[template_name]

        pipeline = template.build(overrides=stages)

        self.stages[pipeline_name] = stages
        self.pipelines[pipeline_name] = pipeline

        return pipeline
