import numpy as np
import random
import collections
from pytorch_transformers import BertTokenizer

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

def main():
    a = np.load('/home/ypd-19-2/abu/beyond_baseline/train_examples.npy', allow_pickle=True)
    b = np.load('/home/ypd-19-2/abu/beyond_baseline/val_examples.npy', allow_pickle=True)
    c = np.load('/home/ypd-19-2/abu/beyond_baseline/test_examples.npy', allow_pickle=True)
    train_examples = []
    val_examples = []
    for example in a:
        train_examples.append(example)
    for example in b:
        train_examples.append(example)
    for example in c:
        val_examples.append(example)
        train_examples.append(example)


    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=False)
    tokens_list = []
    for (ex_index, example) in enumerate(train_examples):
        a_tokens = []
        b_tokens = []
        for t in example.text_a:
            a_tokens = a_tokens + tokenizer.tokenize(t)
        for t in example.text_b:
            b_tokens = b_tokens + tokenizer.tokenize(t)
        tokens = ['[CLS]']+a_tokens+['[SEP]']+b_tokens+['[SEP]']
        tokens_list.append(tokens)
    max_len = 0
    for tokens in tokens_list:
        if max_len <= len(tokens):
            max_len = len(tokens)

    print('bupt')
if __name__ == '__main__':
    main()