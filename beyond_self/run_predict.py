from __future__ import absolute_import, division, print_function

import argparse
import csv
import json
import logging
import os
import random
import sys
import time
import modeling_bert
from sklearn.metrics import roc_auc_score
from transformers import WEIGHTS_NAME, CONFIG_NAME

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_transformers import (WEIGHTS_NAME, AdamW, BertConfig, BertForSequenceClassification,
                                  BertForTokenClassification, BertTokenizer,
                                  WarmupLinearSchedule)
from torch import nn
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from seqeval.metrics import classification_report
'''模型融合的预测代码'''

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label = label

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

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)
def convert_examples_to_features(examples, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    features = []
    for (ex_index, example) in enumerate(examples):
        a_tokens = []
        b_tokens = []
        for t in example.text_a:
            a_tokens = a_tokens + tokenizer.tokenize(t)
        for t in example.text_b:
            b_tokens = b_tokens + tokenizer.tokenize(t)

        '''
        tokens = ['[CLS]']+a_tokens+['[SEP]']+b_tokens+['[SEP]']
        segment_ids = [0] * (len(a_tokens)+2) + [1] * (len(b_tokens)+1)

        '''
        tokens = ['[CLS]'] + b_tokens + ['[SEP]'] + a_tokens + ['[SEP]']
        segment_ids = [0] * (len(b_tokens) + 2) + [1] * (len(a_tokens) + 1)





        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label=example.label
            )
        )
    return features

def all_predict(args, model_dirs, test_examples):
    all_probs = []
    for model_path in model_dirs:
        logger.info("***** Model Path *****")
        logger.info(model_path)
        logger.info(" Num examples = %d", len(test_examples))
        logger.info(" Batch size = %d", args.eval_batch_size)
        tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=args.do_lower_case)
        test_features = convert_examples_to_features(test_examples, args.max_seq_length, tokenizer)
        all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
        all_labels = torch.tensor([f.label for f in test_features], dtype=torch.long)
        test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_labels)
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)
        config = BertConfig.from_pretrained(model_path, num_labels=2)
        model = modeling_bert.BERT_Match_FocalLoss.from_pretrained(
            model_path,
            from_tf=False,
            config=config,
            args=args,
        )
        model.cuda()
        model.eval()
        probs = []
        for input_ids, input_mask, segment_ids, label in tqdm(test_dataloader, desc="Predicting"):
            input_ids = input_ids.cuda()
            input_mask = input_mask.cuda()
            segment_ids = segment_ids.cuda()
            label = label.cuda()

            with torch.no_grad():
                _,_, probabilities = model(input_ids, input_mask, segment_ids, label)
                probs.extend(probabilities[:,1].cpu().numpy())
        all_probs.append(probs)
    np.save('test_B_b_4_8_roberta.npy', all_probs)
    print('bupt')

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--model_paths", default="/home/ypd-19-2/abu/beyond_output/roberta_100_3_epoch",
                        type=str, required=False,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    args = parser.parse_args()
    test_examples = np.load('/home/ypd-19-2/abu/beyond_baseline/test_examples_B.npy', allow_pickle=True)
    list_dir = os.listdir(args.model_paths)
    model_dirs = [os.path.join(args.model_paths, dir_) for dir_ in list_dir]
    #model_dirs = ['/home/ypd-19-2/abu/beyond_output/nezha_wwm_n_gram_mask_100_epoch_10_fold/0']
    all_predict(args, model_dirs, test_examples)


if __name__ == '__main__':
    main()