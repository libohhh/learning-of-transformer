import os
import torch
import datasets
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq

dataset = datasets.Dataset.load_from_disk('./data/alpaca_data_zh')

tokenizer = AutoTokenizer.from_pretrained('/mnt/HDD/lb/model/ZhipuAI/chatglm3-6b-base', trust_remote_code=True)

def process_func(data):
    MAX_LENGTH = 384
    input_ids, attention_mask, labels = [], [], []
    query = '\n'.join([ data['instruction'], data['input']]).strip()
    input = tokenizer.build_chat_input(query, history=[], role='user')
    response = tokenizer('\n' + data['output'], add_special_tokens=False)
    
    input_ids = input['input_ids'][0].numpy().tolist() + response['input_ids'] + [tokenizer.eos_token_id]
    attention_mask = input['attention_mask'][0].numpy().tolist() + response['attention_mask'] + [1]
    labels = [-100] * len(input['input_ids'][0].numpy().tolist()) + response['input_ids'] + [tokenizer.eos_token_id]
    
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

model = AutoModelForCausalLM.from_pretrained('/mnt/HDD/lb/model/ZhipuAI/chatglm3-6b-base', trust_remote_code=True, device_map='auto', torch_dtype=torch.half)
peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, target_modules=['query_key_value'])
peft_model = get_peft_model(model, peft_config)
# peft_model.enable_input_require_grads()
# peft_model = peft_model.half()

train_args = TrainingArguments(output_dir='./glm3_16bits',
                               per_device_train_batch_size=2,
                               gradient_accumulation_steps=4,
                               save_strategy='epoch',
                               logging_strategy='steps',
                               logging_steps=30,
                               adam_epsilon=1e-4, 
                               num_train_epochs=1, 
                               remove_unused_columns=False)
trainer = Trainer(peft_model, args=train_args, train_dataset=tokenized_dataset,
                  data_collator=DataCollatorForSeq2Seq(tokenizer))

trainer.train()

# input = 'Human:{}\n{}'.format('你好', '').strip() + 'Assitant:\n'
# input = tokenizer(input, add_special_tokens=False, return_tensors='pt').to('cuda')
# res = tokenizer.decode(peft_model.generate(**input, max_length=512, do_sample=True)[0], skip_special_tokens=True)
query = '考试注意事项有哪些？'
res = peft_model.chat(tokenizer, query, history=[])
print(res)