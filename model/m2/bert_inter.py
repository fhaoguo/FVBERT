# coding: utf-8
# @Author     :fenghaoguo
# @Time       :2022/5/21 12:57
# @FileName   :bert_inter.py
# @Description:

from .bert_act import ACT2FN
from .linear import Linear
from .module import Module


class BertIntermediate(Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = Linear(config.hidden_size, config.intermediate_size)
        # if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states
