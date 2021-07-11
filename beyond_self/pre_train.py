from datasets import load_dataset
from transformers import BertTokenizer
from transformers import BertForMaskedLM
from transformers import Trainer, TrainingArguments
from trick.data_collator import DataCollatorForNGramMask, DataCollatorForNGramMaskFixed
import os

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    dataset = load_dataset('text', data_files={'train':['/home/ypd-19-2/abu/beyond_baseline/train.txt','/home/ypd-19-2/abu/beyond_baseline/val.txt'], 'test':['/home/ypd-19-2/abu/beyond_baseline/val.txt']})
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=True)
    def tokenize_function(examples):
        return tokenizer(examples["text"])
    tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])
    block_size = 128

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        batch_size=32,
        num_proc=4,
    )
    model = BertForMaskedLM.from_pretrained('bert-base-chinese')

    #data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
    data_collator = DataCollatorForNGramMaskFixed(tokenizer=tokenizer, mlm_probability=0.15)
    training_args = TrainingArguments(
        "bert_base_n_gram_mask_300",
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
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["test"],
        data_collator=data_collator,

    )
    print('--------------Run Training----------------')
    trainer.train()
    print('bupt')

if __name__ == '__main__':
    main()
