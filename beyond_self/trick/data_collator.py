from transformers import DataCollatorForLanguageModeling, BertTokenizer
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
import torch
from transformers.tokenization_utils_base import BatchEncoding
from transformers.data.data_collator import  _collate_batch
import numpy as np
import random
import collections
MaskedLmInstance = collections.namedtuple("MaskedLmInstance", ["index", "label"])
class DataCollatorForNGramMask(DataCollatorForLanguageModeling):
    """
    Data collator used for language modeling
    """

    def __call__(
            self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], (dict, BatchEncoding)):
            batch = self.tokenizer.pad(examples, return_tensors="pt")
        else:
            batch = {"input_ids": _collate_batch(examples, self.tokenizer)}

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.n_gram_mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return batch

    def n_gram_mask_tokens(self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None
                           )-> Tuple[torch.Tensor, torch.Tensor]:
        """
                Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
                """
        labels = inputs.clone()
        # n-gram masking Albert
        max_ngram = 3
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        masked_lm_prob = 0.15
        max_predictions_per_seq = 20
        vocab_list = list(tokenizer.vocab.keys())
        ngrams = torch.arange(1, max_ngram + 1, dtype=np.int64)
        pvals = 1. / torch.arange(1, max_ngram + 1)
        pvals /= pvals.sum(keepdims=True)  # p(n) = 1/n / sigma(1/k)
        cand_indices = []
        for (i, token) in enumerate(inputs):
            if torch.equal(token, torch.tensor(tokenizer.convert_tokens_to_ids("[CLS]"))) or torch.equal(token,torch.tensor(tokenizer.convert_tokens_to_ids("[SEP]"))):
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
        num_to_mask = min(max_predictions_per_seq, max(1, int(round(len(inputs) * masked_lm_prob))))
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
                        masked_token = torch.tensor(tokenizer.convert_tokens_to_ids("[MASK]"))
                    else:
                        # 10% of the time, keep original
                        if random.random() < 0.5:
                            masked_token = inputs[ind]
                        # 10% of the time, replace with random word
                        else:
                            masked_token = tokenizer.convert_tokens_to_ids(random.choice(vocab_list))
                    masked_token_labels.append(MaskedLmInstance(index=ind, label=inputs[ind]))
                    inputs[ind] = masked_token

        masked_token_labels = sorted(masked_token_labels, key=lambda x: x.index)

        return inputs, labels


class DataCollatorForNGramMaskFixed(DataCollatorForLanguageModeling):
    def __call__(
            self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], (dict, BatchEncoding)):
            batch = self.tokenizer.pad(examples, return_tensors="pt")
        else:
            batch = {"input_ids": _collate_batch(examples, self.tokenizer)}

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.n_gram_mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return batch

    def n_gram_mask_tokens(self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None
                           )-> Tuple[torch.Tensor, torch.Tensor]:
        """
                Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
                """
        labels = inputs.clone()
        max_ngram = 2
        masked_lm_prob = 0.15
        max_predictions_per_seq = 20
        vocab_list = list(self.tokenizer.vocab.keys())
        ngrams = np.arange(1, max_ngram + 1, dtype=np.int64)
        pvals = 1. / np.arange(1, max_ngram + 1)
        pvals /= pvals.sum(keepdims=True)  # p(n) = 1/n / sigma(1/k)
        for ink, input in enumerate(inputs):
            cand_indices = []
            for (i, token) in enumerate(input):
                if torch.equal(token, torch.tensor(self.tokenizer.convert_tokens_to_ids("[CLS]"))) or torch.equal(token,
                                                                                                                  torch.tensor(
                                                                                                                      self.tokenizer.convert_tokens_to_ids(
                                                                                                                          "[SEP]"))):
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
            num_to_mask = min(max_predictions_per_seq, max(1, int(round(len(input) * masked_lm_prob))))
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
                            masked_token = torch.tensor(self.tokenizer.convert_tokens_to_ids("[MASK]"))
                        else:
                            # 10% of the time, keep original
                            if random.random() < 0.5:
                                masked_token = input[ind]
                            # 10% of the time, replace with random word
                            else:
                                masked_token = self.tokenizer.convert_tokens_to_ids(random.choice(vocab_list))
                        masked_token_labels.append(MaskedLmInstance(index=ind, label=input[ind]))
                        input[ind] = masked_token
            masked_token_labels = sorted(masked_token_labels, key=lambda x: x.index)
            mask_indices = [p.index for p in masked_token_labels]
            masked_labels = [p.label for p in masked_token_labels]
            labels_return = torch.full(input.shape, -100)
            for ctk, indice in enumerate(mask_indices):
                labels_return[indice] = masked_labels[ctk]
            labels[ink] = labels_return
            inputs[ink] = input
        return inputs, labels






