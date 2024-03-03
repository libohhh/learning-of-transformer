import datasets
from peft import PromptTuningConfig, TaskType, get_peft_model, PromptTuningInit
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq

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

model = AutoModelForCausalLM.from_pretrained('./models/bloom_1b4_zh', low_cpu_mem_usage=True, load_in_8bit=True, device_map='auto')

# soft prompt
# config = PromptTuningConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=10)

# hard prompt
config = PromptTuningConfig(task_type=TaskType.CAUSAL_LM, 
                            prompt_tuning_init=PromptTuningInit.TEXT,
                            prompt_tuning_init_text='下面是一段人与助手的对话',
                            num_virtual_tokens=len(tokenizer('下面是一段人与助手的对话')['input_ids']),
                            tokenizer_name_or_path='./models/bloom_1b4_zh')

peft_model = get_peft_model(model, config)

train_args = TrainingArguments(
    output_dir='./for_prompt_tuning/soft_prompt/',
    per_gpu_train_batch_size=4,
    per_gpu_eval_batch_size=16,
    logging_strategy='steps',
    save_strategy='epoch',
    num_train_epochs=1,
    gradient_accumulation_steps=1, 
    optim="adafactor",
)
trainer = Trainer(model=peft_model, args=train_args, 
                  train_dataset=tokenized_dataset,
                  data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True))

# trainer.train()

input_ids = tokenizer('Human:{}\n{}'.format('维持感情的方法？', '').strip() + 'Assitant:\n', return_tensors='pt').to('cuda')
res = tokenizer.decode(peft_model.generate(**input_ids, max_length=256, do_sample=True)[0], skip_special_tokens=True)
print(res)