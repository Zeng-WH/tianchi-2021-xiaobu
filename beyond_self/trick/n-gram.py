from sklearn.feature_extraction.text import CountVectorizer
from pytorch_transformers import BertTokenizer
import numpy as np
import random
import collections
MaskedLmInstance = collections.namedtuple("MaskedLmInstance", ["index", "label"])
def create_masked_lm_predictions(tokens, max_ngram, masked_lm_prob, max_predictions_per_seq, vocab_list, tokenizer):
    """Creates the predictions for the masked LM objective. This is mostly copied from the Google BERT repo, but
    with several refactors to clean it up and remove a lot of unnecessary variables."""
    labels = tokens.copy()
    # n-gram masking Albert
    ngrams = np.arange(1, max_ngram + 1, dtype=np.int64)
    pvals = 1. / np.arange(1, max_ngram + 1)
    pvals /= pvals.sum(keepdims=True)  # p(n) = 1/n / sigma(1/k)
    cand_indices = []
    for (i, token) in enumerate(tokens):
        if token == tokenizer.convert_tokens_to_ids("[CLS]") or token == tokenizer.convert_tokens_to_ids("[SEP]"):
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
                    masked_token = tokenizer.convert_tokens_to_ids("[MASK]")
                else:
                    # 10% of the time, keep original
                    if random.random() < 0.5:
                        masked_token = tokens[ind]
                    # 10% of the time, replace with random word
                    else:
                        masked_token = tokenizer.convert_tokens_to_ids(random.choice(vocab_list))
                masked_token_labels.append(MaskedLmInstance(index=ind, label=tokens[ind]))
                tokens[ind] = masked_token

    #assert len(masked_token_labels) <= num_to_mask
    masked_token_labels = sorted(masked_token_labels, key=lambda x: x.index)
    mask_indices = [p.index for p in masked_token_labels]
    masked_labels = [p.label for p in masked_token_labels]
    return labels, tokens, mask_indices, masked_labels
def main():
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=False)
    vocab_words = list(tokenizer.vocab.keys())
    text = ['我', '是', '你', '爹', '你', '是', '我']
    tokens = []
    input_ids = []
    for token in text:
        tokens.extend(tokenizer.tokenize(token))
    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    for token in tokens:
        input_ids.append(tokenizer.convert_tokens_to_ids(token))
    a,b,c, d = create_masked_lm_predictions(input_ids, 3, 0.15, 20, vocab_words, tokenizer)



    print('bupt')

if __name__ == '__main__':
    main()
