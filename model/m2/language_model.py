# coding: utf-8
# @Author     :fenghaoguo
# @Time       :2022/5/20 00:55
# @FileName   :bert_lm.py
# @Description:


from .bert import BERT
from .module import Module
from .linear import Linear
from .activation import LogSoftmax


class BERTLM(Module):
    """
    BERT Language Model
    Next Sentence Prediction Model + Masked Language Model
    """

    def __init__(self, bert: BERT, vocab_size):
        """
        :param bert: BERT m2 which should be trained
        :param vocab_size: total vocab size for masked_lm
        """

        super().__init__()
        self.bert = bert
        self.next_sentence = NextSentencePrediction(self.bert.hidden)
        self.mask_lm = MaskedLanguageModel(self.bert.hidden, vocab_size)

    def forward(self, x, segment_label):
        x = self.bert(x, segment_label)
        return self.next_sentence(x), self.mask_lm(x)


class NextSentencePrediction(Module):
    """
    2-class classification m2 : is_next, is_not_next
    """

    def __init__(self, hidden):
        """
        :param hidden: BERT m2 output size
        """
        super().__init__()
        self.linear = Linear(hidden, 2)
        self.softmax = LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x[:, 0]))


class MaskedLanguageModel(Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, hidden, vocab_size):
        """
        :param hidden: output size of BERT m2
        :param vocab_size: total vocab size
        """
        super().__init__()
        self.linear = Linear(hidden, vocab_size)
        self.softmax = LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))


if __name__ == '__main__':
    run_code = 0
