import torch
import datasets
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq

dataset = datasets.Dataset.load_from_disk('./data/alpaca_data_zh')

tokenizer = AutoTokenizer.from_pretrained('/mnt/HDD/lb/model/modelscope/Llama-2-7b-ms')
tokenizer.pad_token_id = 2
tokenizer.padding_side = 'right'
def process_func(data):
    MAX_LENGTH = 384
    input_ids, attention_mask, labels = [], [], []
    input = tokenizer('\n'.join(['Human:' + data['instruction'], data['input']]).strip() + "Assitant:", add_special_tokens=False)
    response = tokenizer(data['output'], add_special_tokens=False)
    input_ids = input['input_ids'] + response['input_ids'] + [tokenizer.eos_token_id]
    attention_mask = input['attention_mask'] + response['attention_mask'] + [1]
    labels = [-100] * len(input['input_ids']) + response['input_ids'] + [tokenizer.eos_token_id]
    
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }
tokenized_dataset = dataset.map(process_func, remove_columns=dataset.column_names)

model = AutoModelForCausalLM.from_pretrained('/mnt/HDD/lb/model/modelscope/Llama-2-7b-ms', device_map='auto', torch_dtype=torch.half)
peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM)
peft_model = get_peft_model(model, peft_config)
peft_model.enable_input_require_grads()
peft_model = peft_model.half()

train_args = TrainingArguments(output_dir='./llama_16bits',
                               per_device_train_batch_size=2,
                               gradient_accumulation_steps=4,
                               gradient_checkpointing=True,
                               save_strategy='epoch',
                               logging_strategy='steps',
                               logging_steps=30,
                               adam_epsilon=1e-4, 
                               num_train_epochs=1)
trainer = Trainer(peft_model, args=train_args, train_dataset=tokenized_dataset,
                  data_collator=DataCollatorForSeq2Seq(tokenizer))

trainer.train()

input = 'Human:{}\n{}'.format('你好', '').strip() + 'Assitant:\n'
input = tokenizer(input, add_special_tokens=False, return_tensors='pt').to('cuda')
res = tokenizer.decode(peft_model.generate(**input, max_length=512, do_sample=True)[0], skip_special_tokens=True)
print(res)