
import torch
print(torch.cuda.is_available())

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,HfArgumentParser,TrainingArguments,pipeline, logging
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
import os, torch, json
from datasets import load_dataset, Dataset
from trl import SFTTrainer
from tqdm import tqdm

with open ('/mnt/data2/prince/stance_finetuning/geval/data/train_data.json', 'r') as f:
    data_full = json.load(f)

aspects = ['coherence', 'consistency', 'fluency', 'relevance']
        

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

data_dict = {"text": texts}

# Create the dataset
dataset = Dataset.from_dict(data_dict)


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

model = PeftModel.from_pretrained(model, "/mnt/data2/prince/stance_finetuning/results/checkpoint-1275")
model.eval()


def generate_output(prompt):
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100)
    result = pipe(f"<s>[INST] {prompt} [/INST]")
    print(result[0]['generated_text'])

for aspect in tqdm(aspects):
    filename = aspect[0:3]+'_detailed.txt'
    prompt = open('/mnt/data2/prince/stance_finetuning/geval/prompts/summeval/' + filename).read()

    for item in data_full:
        full_input = prompt + '\n Source Text: '+ item['source'] + '\n Summary: ' + item["reference"] + "\n Score is: "
        output = item['scores'][aspect]
        text = "<s>[INST] "+full_input + "[/INST]"
        generate_output(text)
