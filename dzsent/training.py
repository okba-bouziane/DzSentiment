"""Trainer pieces kept apart from the script so they can be tested without a model."""

from __future__ import annotations

from torch.utils.data import SequentialSampler
from transformers import Trainer


class OrderedEvalTrainer(Trainer):
    """Group training batches by length for speed, but keep evaluation and prediction in row order.

    With `group_by_length=True`, recent transformers versions also sort the eval/test sets by length,
    so `predict()` returns rows in a different order than the dataset. Metrics that join predictions
    back to rows by position are then silently random (this happened in the first run).
    """

    def _get_eval_sampler(self, eval_dataset):
        if eval_dataset is None:
            return None
        return SequentialSampler(eval_dataset)
