import torch
print(torch.cuda.is_available())

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,HfArgumentParser,TrainingArguments,pipeline, logging
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
import os, torch, wandb, json
from datasets import load_dataset, Dataset
from trl import SFTTrainer

#transformers==4.42.4

# from huggingface_hub import notebook_login
# notebook_login()
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

# Monitering the LLM
wandb.login(key = '7b2d15046503e9d6859850a5cc804c0f9efedddd')
run = wandb.init(
    project='Fine-tuning-Mistral', 
    job_type="training", 
    anonymous="allow"
)

base_model = "mistralai/Mistral-7B-Instruct-v0.2" 
dataset_name = "mwitiderrick/lamini_mistral"
new_model = "Mistral-7b-v2-finetune"


with open ('/mnt/data2/prince/stance_finetuning/geval/data/train_data.json', 'r') as f:
    data_full = json.load(f)

aspects = ['coherence', 'consistency', 'fluency', 'relevance']

texts = []
for aspect in aspects:
    filename = aspect[0:3]+'_detailed.txt'
    prompt = open('/mnt/data2/prince/stance_finetuning/geval/prompts/summeval/' + filename).read()

    for item in data_full:
        full_input = prompt + '\n Source Text: '+ item['source'] + '\n Summary: ' + item["reference"] + "\n Score is: "
        output = item['scores'][aspect]
        text = "<s>[INST] "+full_input + "[/INST]" + f"Score is {output}" + " <s>"
        texts.append(text)

# with open ('/mnt/data2/prince/stance_finetuning/geval/results/gpt4_coh_detailed.json', 'r') as f:
#     data_full = json.load(f)

# texts = []
# for item in data_full:
#     full_input = item['prompt']
#     output = item['scores']["coherence"]
#     texts.append(f"<s>[INST] {full_input} [/INST] {output} </s>")

data_dict = {"text": texts}

# Create the dataset
dataset = Dataset.from_dict(data_dict)

#Importing the dataset
# dataset = load_dataset(dataset_name, split="train")

bnb_config = BitsAndBytesConfig(  
    load_in_4bit= True,
    bnb_4bit_quant_type= "nf4",
    bnb_4bit_compute_dtype= torch.bfloat16,
    bnb_4bit_use_double_quant= False,
)
model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
)
model.config.use_cache = False # silence the warnings. Please re-enable for inference!
model.config.pretraining_tp = 1
model.gradient_checkpointing_enable()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.padding_side = 'right'
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = True
tokenizer.add_bos_token, tokenizer.add_eos_token

model = prepare_model_for_kbit_training(model)
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj"]
)
model = get_peft_model(model, peft_config)

prompt = "How to make banana bread?"
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
result = pipe(f"<s>[INST] {prompt} [/INST]")
print(result[0]['generated_text'])

training_arguments = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=25,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="wandb"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    max_seq_length= None,
    dataset_text_field="text",
    tokenizer=tokenizer,
    args=training_arguments,
    packing= False,
)
trainer.train()

# Save the fine-tuned model
trainer.model.save_pretrained(new_model)
wandb.finish()
model.config.use_cache = True
model.eval()

prompt = "Can I find information about the code's approach to handling long-running tasks and background jobs?"
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
result = pipe(f"<s>[INST] {prompt} [/INST]")
print(result[0]['generated_text'])




