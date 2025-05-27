import torch.nn.functional as F
import torch
import json
import fire
from transformers import BitsAndBytesConfig
from transformers import(
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
)
from peft import get_peft_model, LoraConfig, TaskType
from peft import PeftModel
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Diverse beam-search')
parser.add_argument('--epoch', type=int,   default=50)
parser.add_argument('--batch_size', type=int,   default=1)
parser.add_argument('--model_name', type=str,   default="meta-llama/Llama-3.2-1B-Instruct")
parser.add_argument('--finetuning_model', type=str,   default="/data/yeeunkim/self_consistency/PubMed_trunc/full_pubmed_trun_out/checkpoint-5204")
parser.add_argument('--lora_model', type=str,   default="/data/yeeunkim/self_consistency/PubMed_trunc/pubmed_lora_output/checkpoint-41617")
parser.add_argument('--data_path', type=str,   default="/data/yeeunkim/self_consistency/bitnet/oracle_dataset/pubmed_trun/all_pubmed_trun_test.json")
parser.add_argument('--save_name', type=str,   default="full_pubmed_trunc_beam_4")
parser.add_argument('--devices', type=int,   default=0)
parser.add_argument('--num_beams', type=int,   default=4)
parser.add_argument('--num_beam_groups', type=int,   default=2)
parser.add_argument('--lora', action="store_true")
args = parser.parse_args()

model_name = args.model_name
# nf4_config = BitsAndBytesConfig(
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_compute_dtype=torch.bfloat16
# ) 
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# base_model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        padding_side="left",
        cache_dir="/data/huggingface_models/"
    )
tokenizer.pad_token = tokenizer.eos_token

# 설정 불러오기 및 수정
config = AutoConfig.from_pretrained(args.finetuning_model)
config.eos_token_id = config.eos_token_id[0] if isinstance(config.eos_token_id, list) else config.eos_token_id
config.pad_token_id = config.pad_token_id[0] if isinstance(config.pad_token_id, list) else config.pad_token_id

# 수정된 설정으로 모델 로드
base_model = AutoModelForCausalLM.from_pretrained(args.finetuning_model, config=config)

base_model.config.pad_token_id = base_model.config.eos_token_id
base_model.generation_config.pad_token_id = tokenizer.pad_token_id
# if args.lora:
#     lora_config = LoraConfig(
#         task_type=TaskType.CAUSAL_LM,
#         r=8,
#         lora_alpha=16,
#         lora_dropout=0.1,
#         target_modules=["q_proj", "k_proj", "o_proj"]
#     )

#     model = get_peft_model(base_model, lora_config)
#     model.config.pad_token_id = model.config.eos_token_id
#     lora_weights_path = args.lora_weight_path
#     lora_weights = torch.load(lora_weights_path, map_location="cpu")
#     model.load_state_dict(lora_weights, strict=False)
# else:
#     model = base_model

if args.lora:
    print("fine tuning mode")
    lora_path = args.lora_model
    model = PeftModel.from_pretrained(base_model, lora_path)
model = base_model
print(model)
print("-"*50)
    

model.to("cuda")
model.eval()

def ch_datasets(path:str):
    with open(path) as f:
        lines = f.readlines()
        test_sets = []
        for data in tqdm(lines, total = len(lines)):
            data = json.loads(data.strip())
            input_buffer = []
            inputs = data["text"][:717]
            output_buffer = []
            output_idx = data["indices"][:717][0]

            for idx, in_sent in enumerate(inputs):
                sent_idx = idx
                in_token= "[sent" + str(sent_idx)+"] "
                in_sent = in_token + in_sent 
                input_buffer.append(in_sent)
                input_new_sent = " ".join(input_buffer)

            for idx in range(len(output_idx)):
                    output = "[sent" + str(output_idx[idx]) +"]"
                    output_buffer.append(output)
                    output_new_sent = " ".join(output_buffer)
                    idx += 1

            data_ = {
                "input" : input_new_sent,
                "output" : output_new_sent
            }
            test_sets.append(data_)

            # print(test_sets)
            # exit()
    return test_sets

def generate_text_batch_with_diverse_beam_search(
    input_text, model, tokenizer, max_length=10000, num_beams=args.num_beams, num_beam_groups=args.num_beam_groups, diversity_penalty=1.0,
):
    inputs = tokenizer(input_text, padding=True, max_length=10000, truncation=True, return_tensors="pt")
    input_ids = inputs["input_ids"].to("cuda")
    attention_mask = inputs["attention_mask"].to("cuda")
    #print(input_ids.dtype) 
    model_dtype = model.base_model.dtype
    #print(model_dtype)
    #exit()
    output_texts = []
    token_probabilities = []

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=200,
            num_beams=args.num_beams,
            num_beam_groups=num_beam_groups,
            diversity_penalty=diversity_penalty,
            #temperature=temperature,  # Add temperature for randomness
            #top_k=top_k,              # Restrict sampling to top-k tokens
            #top_p=top_p,
            repetition_penalty = 0.9,
            num_return_sequences=args.num_beams,
            early_stopping=False,
            output_scores=True,
            return_dict_in_generate=True,
            do_sample = False,
        )
    # print(outputs)
    # exit()
    generated_ids = outputs.sequences
    sequence_scores = outputs.scores  # List of logits for each generation step
    input_length = input_ids.shape[1]
    for seq_idx, seq in enumerate(generated_ids):
        # text = tokenizer.decode(seq, skip_special_tokens=True)
        # output_texts.append(text)
        text = tokenizer.decode(seq[input_length:], skip_special_tokens=True)
        output_texts.append(text)

        # Calculate probabilities step-by-step for this sequence
        token_probs = []
        for step_logits in sequence_scores:
            # Get probabilities for the tokens in the current step
            step_probs = F.softmax(step_logits[seq_idx], dim=-1)
            top_prob, top_token_id = torch.max(step_probs, dim=-1)
            top_prob_float = top_prob.item()
            top_token_text = tokenizer.decode([top_token_id.item()]).strip()
            token_probs.append((top_token_text, top_prob_float))

        token_probabilities.append(token_probs)
#    output_texts = [tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids]
    return output_texts, token_probabilities

args.data_path = ch_datasets(args.data_path)
def load_data(data):
    buffer = []
    for line in data:
        # print(line)
        # exit()
        buffer.append(line["input"])
        # print(buffer)
        # exit()
    return buffer

batch_size = 1
batched_outputs = []
batched_probs = []
input_texts = load_data(args.data_path)
for i in tqdm(range(0, len(input_texts), batch_size)):
    batch = input_texts[i:i + batch_size]
    output_texts, probs = generate_text_batch_with_diverse_beam_search(batch, model, tokenizer)
    batched_outputs.extend(output_texts)
    # print(batched_outputs)
    # exit()
    batched_probs.extend(probs)

paired_output_data = []
beam = args.num_beams
for idx, input_text in enumerate(input_texts):
    outputs_for_input = batched_outputs[idx * beam : (idx + 1) * beam]
    for beam_idx, output_text in enumerate(outputs_for_input):
        paired_output_data.append({
            "input": input_text,
            "output": output_text,
        })



def write_elems(elems, path):
    with open(path, 'w+') as f:
        for elem in elems:
            try:
                json.dumps(elem)
            except:
                print(elem)
                import sys;sys.exit(1)
            f.write(f'{json.dumps(elem)}\n')
write_elems(paired_output_data, args.save_name + "_output.json")
write_elems(batched_probs, args.save_name + "_prob.json")


