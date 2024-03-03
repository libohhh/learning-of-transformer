import datasets
from peft import LoraConfig, get_peft_model, TaskType
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq

dataset = datasets.Dataset.load_from_disk('./data/alpaca_data_zh')

tokenizer = AutoTokenizer.from_pretrained('./models/bloom_1b4_zh')
def process_func(data):
    input_ids, attention_mask, labels = [], [], []
    input = tokenizer('\n'.join(["Human:\n" + data['instruction'], data['input']]).strip() + "Assistant:\n")
    response = tokenizer(data['output'] + tokenizer.eos_token)
    input_ids = input['input_ids'] + response['input_ids']
    attention_mask = input['attention_mask'] + response['attention_mask']
    labels = [-100] * len(input['input_ids']) + response['input_ids']
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }

tokenized_dataset = dataset.map(process_func, remove_columns=dataset.column_names)

model = AutoModelForCausalLM.from_pretrained('./models/bloom_1b4_zh', load_in_8bit=True)

peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, target_modules=['query_key_value'])

peft_model = get_peft_model(model, peft_config)

train_args = TrainingArguments(output_dir='for_lora_model',
                               num_train_epochs=1,
                               save_strategy='epoch',
                               logging_steps=30,
                               logging_strategy='epoch',
                               per_device_train_batch_size=4)
trainer = Trainer(model=peft_model, args=train_args,
                  train_dataset=tokenized_dataset,
                  data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True))

# trainer.train()

# load_and_save

model = PeftModel.from_pretrained(model=model, model_id='/mnt/HDD/lb/transformers/for_lora_model/checkpoint-6715')
model = model.to('cuda')
input = 'Human: {}\n{}'.format('维持感情的方法？','').strip() + 'Assitant:\n'
input = tokenizer(input, return_tensors='pt').to('cuda')
res = tokenizer.decode(model.generate(**input, max_length=256, do_sample=True)[0], skip_special_tokens=True)
print(res)