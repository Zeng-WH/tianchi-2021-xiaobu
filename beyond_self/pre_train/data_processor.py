import numpy as np
import random
import collections
import torch
import logging

MaskedLmInstance = collections.namedtuple("MaskedLmInstance", ["index", "label"])

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

def create_masked_lm_predictions(tokens, max_ngram, masked_lm_prob, max_predictions_per_seq, vocab_list):
    """Creates the predictions for the masked LM objective. This is mostly copied from the Google BERT repo, but
    with several refactors to clean it up and remove a lot of unnecessary variables."""

    # n-gram masking Albert
    ngrams = np.arange(1, max_ngram + 1, dtype=np.int64)
    pvals = 1. / np.arange(1, max_ngram + 1)
    pvals /= pvals.sum(keepdims=True)  # p(n) = 1/n / sigma(1/k)
    cand_indices = []
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            continue
        # Whole Word Masking means that if we mask all of the wordpieces
        # corresponding to an original word. When a word has been split into
        # WordPieces, the first token does not have any marker and any subsequence
        # tokens are prefixed with ##. So whenever we see the ## token, we
        # append it to the previous set of word indexes.
        #
        # Note that Whole Word Masking does *not* change the training code
        # at all -- we still predict each WordPiece independently, softmaxed
        # over the entire vocabulary.
        cand_indices.append(i)
    num_to_mask = min(max_predictions_per_seq, max(1, int(round(len(tokens) * masked_lm_prob))))
    random.shuffle(cand_indices)
    masked_token_labels = []
    covered_indices = set()
    for index in cand_indices:
        n = np.random.choice(ngrams, p=pvals)
        if len(masked_token_labels) >= num_to_mask:
            break
        if index in covered_indices:
            continue
        if index < len(cand_indices) - (n - 1):
            for i in range(n):
                ind = index + i
                if ind in covered_indices:
                    continue
                covered_indices.add(ind)
                # 80% of the time, replace with [MASK]
                if random.random() < 0.8:
                    masked_token = "[MASK]"
                else:
                    # 10% of the time, keep original
                    if random.random() < 0.5:
                        masked_token = tokens[ind]
                    # 10% of the time, replace with random word
                    else:
                        masked_token = random.choice(vocab_list)
                masked_token_labels.append(MaskedLmInstance(index=ind, label=tokens[ind]))
                tokens[ind] = masked_token

    #assert len(masked_token_labels) <= num_to_mask
    masked_token_labels = sorted(masked_token_labels, key=lambda x: x.index)
    mask_indices = [p.index for p in masked_token_labels]
    masked_labels = [p.label for p in masked_token_labels]
    return tokens, mask_indices, masked_labels


def convert_examples_to_features(examples, max_seq_length, tokenizer, max_ngram, masked_lm_prob, max_predictions_per_seq):
    features = []
    for (ex_index, example) in enumerate(examples):
        a_tokens = []
        b_tokens = []
        for t in example.text_a:
            a_tokens = a_tokens + tokenizer.tokenize(t)
        for t in example.text_b:
            b_tokens = b_tokens + tokenizer.tokenize(t)
        tokens = ['[CLS]']+a_tokens+['[SEP]']+b_tokens+['[SEP]']
        segment_ids = [0] * (len(a_tokens)+2) + [1] * (len(b_tokens)+1)
        labels = tokens.copy()
        labels = [-1 for i in tokens]
        tokens, mask_indices, masked_labels = create_masked_lm_predictions(tokens, max_ngram, masked_lm_prob, max_predictions_per_seq, list(tokenizer.vocab.keys()))
        for ctk, indice in enumerate(mask_indices):
            labels[indice] = tokenizer.convert_tokens_to_ids(masked_labels[ctk])
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            labels.append(-1)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(labels) == max_seq_length
        '''
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info(
                "labels: %s" % " ".join([str(x) for x in labels])
            )
        '''
        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label=labels
            )
        )
        tokens = ['[CLS]'] + b_tokens + ['[SEP]'] + a_tokens + ['[SEP]']
        segment_ids = [0] * (len(b_tokens) + 2) + [1] * (len(a_tokens) + 1)
        labels = tokens.copy()
        labels = [-1 for i in tokens]
        tokens, mask_indices, masked_labels = create_masked_lm_predictions(tokens, max_ngram, masked_lm_prob,
                                                                           max_predictions_per_seq,
                                                                           list(tokenizer.vocab.keys()))
        for ctk, indice in enumerate(mask_indices):
            labels[indice] = tokenizer.convert_tokens_to_ids(masked_labels[ctk])
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            labels.append(-1)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(labels) == max_seq_length
        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label=labels
            )
        )
    return features


def convert_examples_to_features_electra(examples, max_seq_length, tokenizer, max_ngram, masked_lm_prob, max_predictions_per_seq):
    features = []
    for (ex_index, example) in enumerate(examples):
        a_tokens = []
        b_tokens = []
        for t in example.text_a:
            a_tokens = a_tokens + tokenizer.tokenize(t)
        for t in example.text_b:
            b_tokens = b_tokens + tokenizer.tokenize(t)
        tokens = ['[CLS]']+a_tokens+['[SEP]']+b_tokens+['[SEP]']
        segment_ids = [0] * (len(a_tokens)+2) + [1] * (len(b_tokens)+1)
        labels = tokens.copy()
        labels = [-100 for i in tokens]
        tokens, mask_indices, masked_labels = create_masked_lm_predictions(tokens, max_ngram, masked_lm_prob, max_predictions_per_seq, list(tokenizer.vocab.keys()))
        for ctk, indice in enumerate(mask_indices):
            labels[indice] = tokenizer.convert_tokens_to_ids(masked_labels[ctk])
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            labels.append(-100)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(labels) == max_seq_length
        '''
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info(
                "labels: %s" % " ".join([str(x) for x in labels])
            )
        '''
        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label=labels
            )
        )
        tokens = ['[CLS]'] + b_tokens + ['[SEP]'] + a_tokens + ['[SEP]']
        segment_ids = [0] * (len(b_tokens) + 2) + [1] * (len(a_tokens) + 1)
        labels = tokens.copy()
        labels = [-100 for i in tokens]
        tokens, mask_indices, masked_labels = create_masked_lm_predictions(tokens, max_ngram, masked_lm_prob,
                                                                           max_predictions_per_seq,
                                                                           list(tokenizer.vocab.keys()))
        for ctk, indice in enumerate(mask_indices):
            labels[indice] = tokenizer.convert_tokens_to_ids(masked_labels[ctk])
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            labels.append(-100)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(labels) == max_seq_length
        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label=labels
            )
        )
    return features
print('bupt')