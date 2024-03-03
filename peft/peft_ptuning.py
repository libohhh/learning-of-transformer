import os
import datasets
# from peft import get_peft_model, PromptEncoderConfig, PromptEncoderReparameterizationType, TaskType
from peft import get_peft_model, PrefixTuningConfig, TaskType
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq

# os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2'
dataset = datasets.Dataset.load_from_disk('./data/alpaca_data_zh')

tokenizer = AutoTokenizer.from_pretrained('./models/bloom_1b4_zh')
def process_func(example):
    MAX_LENGTH = 256
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer("\n".join(["Human: " + example["instruction"], example["input"]]).strip() + "\n\nAssistant: ")
    response = tokenizer(example["output"] + tokenizer.eos_token)
    input_ids = instruction["input_ids"] + response["input_ids"]
    attention_mask = instruction["attention_mask"] + response["attention_mask"]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"]
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }
    
tokenized_dataset = dataset.map(process_func, remove_columns=dataset.column_names)

model = AutoModelForCausalLM.from_pretrained('./models/bloom_1b4_zh',
                                             load_in_8bit=True,)
                                            #  device_map='auto')

# ptuning
# peft_config = PromptEncoderConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=10,
#                                   encoder_reparameterization_type=PromptEncoderReparameterizationType.MLP, 
#                                   encoder_hidden_size=1024)

# prefix_tuning
peft_config = PrefixTuningConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=10, prefix_projection=True, encoder_hidden_size=1024)
peft_model = get_peft_model(model, peft_config)

train_args = TrainingArguments(output_dir='./for_prefix_model',
                               save_strategy='epoch',
                               logging_steps=30,
                               logging_strategy='steps',
                               per_device_train_batch_size=4, 
                               num_train_epochs=1)

trainer = Trainer(model=peft_model, args=train_args,
                  train_dataset=tokenized_dataset,
                  data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True))
# trainer.train()

from peft import PeftModel

model = PeftModel.from_pretrained(model=model, model_id='./for_ptuning_model/checkpoint-6715')
model = model.to('cuda')
input = 'Human:\n {}\n{}'.format('考试技巧有哪些？', '') + 'Assistant:\n'
input_ids = tokenizer(input, return_tensors='pt').to('cuda')
res = tokenizer.decode(model.generate(**input_ids, do_sample=True, max_length=256)[0], skip_special_tokens=True)
print(res)