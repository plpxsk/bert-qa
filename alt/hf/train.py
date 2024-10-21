from datasets import load_dataset

from transformers import DefaultDataCollator, TrainingArguments, Trainer
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

from utils import preprocess_tokenize_function


# ============================================================================
# load data
# ============================================================================
filter_size = 500

split_str = "train[:" + str(filter_size) + "]"

squad = load_dataset("squad", split=split_str)
squad = squad.train_test_split(test_size=0.2)

# ============================================================================
# load pretrained model
# ============================================================================
bert_model = "bert-base-uncased"
bert_model = "bert-large-uncased"

model = AutoModelForQuestionAnswering.from_pretrained(bert_model)
tokenizer = AutoTokenizer.from_pretrained(bert_model)

# ============================================================================
# preprocess data
# ============================================================================
args_dict = dict(tokenizer=tokenizer)

tokenized_squad = squad.map(preprocess_tokenize_function, batched=True,
                            remove_columns=squad["train"].column_names,
                            fn_kwargs=args_dict)

data_collator = DefaultDataCollator()

# ============================================================================
# Fine Tune aka Post Train
# ============================================================================
# training on 50 records dumps 4GB of data into this dir
output_dir = "weights/post_train_qa_long"

training_args = TrainingArguments(output_dir=output_dir)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=tokenized_squad["train"],
    eval_dataset=tokenized_squad["test"],
    data_collator=data_collator
)

# can fill 22GB into computer RAM
print("Training...")
trainer.train()
