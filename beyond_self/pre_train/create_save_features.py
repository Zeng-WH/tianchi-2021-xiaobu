import numpy as np
import argparse
from pytorch_transformers import BertTokenizer
from data_processor import convert_examples_to_features
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

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label = label
def iter_create(train_examples, val_examples, iters, max_seq_length, tokenizer, max_ngram, masked_lm_prob, max_predictions_per_seq):
    for iter in range(iters):
        train_features = convert_examples_to_features(train_examples, max_seq_length, tokenizer, max_ngram,
                                                      masked_lm_prob, max_predictions_per_seq)
        eval_features = convert_examples_to_features(val_examples, max_seq_length, tokenizer, max_ngram,
                                                  masked_lm_prob, max_predictions_per_seq)
        np.save('/home/ypd-19-2/abu/beyond_self/n_gram_mask/train_features_'+str(iter)+'.npy', train_features)
        np.save('/home/ypd-19-2/abu/beyond_self/n_gram_mask/eval_features_'+str(iter)+'.npy', eval_features)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert_model", default="bert-base-chinese",
                        type=str, required=False,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    # MLM Pre Train
    parser.add_argument("--max_ngram", default=2, type=int,
                        help="")
    parser.add_argument("--masked_lm_prob", default=0.15, type=int,
                        help="")
    parser.add_argument("--max_predictions_per_seq", default=20, type=int,
                        help="")
    args = parser.parse_args()
    train_examples = []
    val_examples = []
    a = np.load('/home/ypd-19-2/abu/beyond_baseline/train_examples.npy', allow_pickle=True)
    b = np.load('/home/ypd-19-2/abu/beyond_baseline/val_examples.npy', allow_pickle=True)
    c = np.load('/home/ypd-19-2/abu/beyond_baseline/test_examples.npy', allow_pickle=True)
    for example in a:
        train_examples.append(example)
    for example in b:
        train_examples.append(example)
    for example in c:
        val_examples.append(example)
        train_examples.append(example)
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    iter_create(train_examples, val_examples, 100, args.max_seq_length, tokenizer, args.max_ngram, args.masked_lm_prob, args.max_predictions_per_seq)
if __name__ == '__main__':
    main()