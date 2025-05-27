import json
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
# from peft import get_peft_model, LoraConfig, TaskType
import torch
from transformers import BitsAndBytesConfig
from datasets import Dataset, DatasetDict
import random
import argparse, os

parser = argparse.ArgumentParser()

parser.add_argument('--train_dataset', default="/data/yeeunkim/self_consistency/bitnet/change_datasets/all_pubmed_trun_train.json")
parser.add_argument('--valid_dataset', default="/data/yeeunkim/self_consistency/bitnet/change_datasets/all_pubmed_trun_val.json")
parser.add_argument('--model_name', default="meta-llama/Llama-3.2-1B-Instruct")
parser.add_argument('--out_dir', default="/data/yeeunkim/self_consistency/")
parser.add_argument('--max_len', type = int, default=10000)

args = parser.parse_args()

os.environ["HUGGINGFACE_TOKEN"] = "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"

if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir, exist_ok=True)

# 1. 모델과 토크나이저 로드 + special token 추가
print("Load Model and Tokenizer...")


tokenizer = AutoTokenizer.from_pretrained(args.model_name,
                                          cache_dir="/data/huggingface_models/",
                                          use_fast=False)
nf4_config = BitsAndBytesConfig(
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    device_map="auto",
    #quantization_config=nf4_config,
    cache_dir="/data/huggingface_models/",
)

# def special_tokens_add(model, tokenizer, max_sent): 
#         new_tokens = ['[sent'+ str(i) + ']' for i in range(0, max_sent+1)]

#         new_tokens_dict = {'additional_special_tokens' : new_tokens} 

#         tokenizer.add_special_tokens(new_tokens_dict)
#         model.resize_token_embeddings(len(tokenizer))

#         return model, tokenizer

# model, tokenizer = special_tokens_add(model, tokenizer, 206)

print("Complete")
print()
# 2. LoRA 설정 및 적용
# print("Load LoRA config and update model...")
# lora_config = LoraConfig(
#     task_type=TaskType.CAUSAL_LM,
#     r=8,
#     lora_alpha=16,
#     lora_dropout=0.1,
#     target_modules=["q_proj", "k_proj", "o_proj"]
# )
# model = get_peft_model(model, lora_config)
model.config.pad_token_id = model.config.eos_token_id
tokenizer.pad_token = tokenizer.eos_token

print("Complete")
print("Device :",model.device)

# 3. 데이터셋 로드 (json 파일)
print("Load data...")
train_data_list = []
valid_data_list = []

def get_dataset(path:str, _list):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        data = list(data)
        data = data[:]
        #print(data)
        long = []
        for row in data:
            input_ = (
                "You are a summarization system. Please summarize the input by selecting the indices of the relevant sentences.\n"
                #"input:\n"
                f'{row["input"]}'
            )
            # print(len(input_))
            long.append(len(input_))
            output_ = (
                #"output:\n"
                f'{row["output"]}'
            )
    # max = 0
    # for i in long:
    #     i = int(i)
    #     if max < i:
    #         max = i
    # print(max)
    # sum = 0
    # for i in long:
    #     sum += i
    # sum = sum / len(long)
    # print(sum) #12511.3442
    # exit()
            _list.append({"input": input_, "output": output_})
    return _list


train_data_list = get_dataset(args.train_dataset, train_data_list)
# print(train_data_list[0])
# exit()
valid_data_list = get_dataset(args.valid_dataset, valid_data_list)
random.seed(123)
random.shuffle(train_data_list)
# train_list = data_list
# valid_list = data_list
train_dataset = Dataset.from_list(train_data_list)
valid_dataset = Dataset.from_list(valid_data_list)
dataset = DatasetDict({
    "train": train_dataset,
    "validation": valid_dataset
})
print("Complete")

def preprocess_function(example):
#    print(example["prompt"])
#    print()
#    print(example["response"])
#    exit()
    # max_length = 12286
    input_tokens = tokenizer(example["input"], truncation=True, max_length=args.max_len, add_special_tokens=False)["input_ids"]
    output_tokens = tokenizer(example["output"], truncation=True, max_length=args.max_len, add_special_tokens=False)["input_ids"]
    
    # 두 부분을 합치고, 마지막에 EOS 토큰 추가
    input_ids = input_tokens + output_tokens + [tokenizer.eos_token_id]
    # labels는 프롬프트 부분은 -100(손실 계산 제외), 응답 부분과 EOS 토큰은 그대로 사용
    labels = [-100] * len(input_tokens) + output_tokens + [tokenizer.eos_token_id]

    return {"input_ids": input_ids, "labels": labels}

tokenized_dataset = dataset.map(preprocess_function, batched=False)

# 5. Data Collator
def custom_collator(features):
    # 각 예제에서 input_ids와 labels를 분리
    input_ids_list = [f["input_ids"] for f in features]
    labels_list = [f["labels"] for f in features]

    # tokenizer.pad를 각각 적용 (여기서는 key를 "input_ids"로 통일해서 사용)
    padded_inputs = tokenizer.pad({"input_ids": input_ids_list}, padding=True, return_tensors="pt")
    padded_labels = tokenizer.pad({"input_ids": labels_list}, padding=True, return_tensors="pt")["input_ids"]

    # labels의 패딩 토큰은 원래 tokenizer.pad_token_id일 텐데, 이를 -100으로 변경
    padded_labels[padded_inputs['attention_mask'] == 0] = -100

    padded_inputs["labels"] = padded_labels
    # print(padded_inputs)
    # exit()
    return padded_inputs


# 6. TrainingArguments 설정
training_args = TrainingArguments(
    output_dir=args.out_dir,
    num_train_epochs=10,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    eval_strategy="epoch",
    save_strategy="best",        
    save_total_limit=2,
    learning_rate=5e-5,
    fp16=True,
    gradient_accumulation_steps=2,
    optim="adamw_torch",
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    weight_decay=0.01,  
)

# 7. Trainer 생성 및 학습 시작
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=custom_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)
tokenizer.save_pretrained(args.out_dir) 

trainer.train()
