import torch
from torch.utils.data import SequentialSampler
from transformers import AutoConfig, AutoModelForSequenceClassification, TrainingArguments

from dzsent.training import OrderedEvalTrainer


class Rows(torch.utils.data.Dataset):
    def __init__(self, lengths):
        self.lengths = lengths

    def __len__(self):
        return len(self.lengths)

    def __getitem__(self, i):
        return {"input_ids": [1] * self.lengths[i], "labels": 0}


def tiny_model():
    cfg = AutoConfig.for_model("bert", vocab_size=10, hidden_size=8, num_hidden_layers=1,
                               num_attention_heads=1, intermediate_size=8, num_labels=3)
    return AutoModelForSequenceClassification.from_config(cfg)


def test_eval_order_is_kept_even_with_group_by_length(tmp_path):
    args = TrainingArguments(output_dir=str(tmp_path), group_by_length=True, per_device_eval_batch_size=2,
                             report_to=[], use_cpu=True)
    trainer = OrderedEvalTrainer(model=tiny_model(), args=args)
    ds = Rows([3, 50, 1, 20, 7])  # mixed lengths: a length-grouped sampler would reorder these
    sampler = trainer._get_eval_sampler(ds)
    assert isinstance(sampler, SequentialSampler)
    assert list(sampler) == [0, 1, 2, 3, 4]
