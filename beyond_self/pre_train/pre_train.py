from torch.utils.data.dataset import Dataset
import os
import numpy as np
import torch
from transformers import BertTokenizer
from transformers import BertForMaskedLM
from transformers import Trainer, TrainingArguments

class Input_Set(Dataset):

    def __init__(self, input_ids, input_mask, segment_ids, labels):
        self.input_ids = input_ids
        self.attention_mask = input_mask
        self.token_type_ids = segment_ids
        self.masked_lm_labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, item):
        return self.input_ids[item], self.attention_mask[item], self.token_type_ids[item], self.masked_lm_labels[item]

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    train_features = np.load('/home/ypd-19-2/abu/beyond_self/n_gram_mask/train_features_' + '0' + '.npy',
                             allow_pickle=True)
    eval_features = np.load('/home/ypd-19-2/abu/beyond_self/n_gram_mask/eval_features_' + '0' + '.npy',
                            allow_pickle=True)
    model = BertForMaskedLM.from_pretrained('bert-base-chinese')
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_labels = torch.tensor([f.label for f in train_features], dtype=torch.long)
    train_set = Input_Set(all_input_ids, all_input_mask, all_segment_ids, all_labels)
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_labels = torch.tensor([f.label for f in eval_features], dtype=torch.long)
    eval_set = Input_Set(all_input_ids, all_input_mask, all_segment_ids, all_labels)
    model = BertForMaskedLM.from_pretrained('bert-base-chinese')
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=True)
    training_args = TrainingArguments(
        "bert_base_n_gram",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        num_train_epochs=300,
        weight_decay=0.01,
        save_steps=30000,
        overwrite_output_dir=True,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=eval_set,
    )
    print('--------------Run Training----------------')
    trainer.train()
    print('bupt')

if __name__ == '__main__':
    main()
