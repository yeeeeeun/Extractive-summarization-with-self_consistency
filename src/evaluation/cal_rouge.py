from rouge_score import rouge_scorer
from collections import Counter

import json
import re
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--gen_data', required=False, default= "/data/yeeunkim/self_consistency/PubMed_trunc/pubmed_trunc_beam_8_output.json")
parser.add_argument('--gold_data', required=False, default= "/data/yeeunkim/self_consistency/bitnet/oracle_dataset/pubmed_trun/all_pubmed_trun_test.json")
parser.add_argument('--num_beam', required=False, type=int, default=8)
parser.add_argument('--out',default="./result_rouge/pubmed_trun_beam_8_top1_result___.json", required=False)
args = parser.parse_args()


def load_gen_data(path):
    gen_sum = []
    with open(path) as f:
        for line in f:
            _gen_sum = []
            line = json.loads(line.strip())
            idx = line["output"]
            texts = line["input"]
            parts = texts.split("[sent")
            pattern = r'(\[sent\d+\])\s*(.*?)(?=\[sent\d+\]|$)'
            matches = re.findall(pattern, texts, re.DOTALL)
            for sent in matches:
                sent_id, sent_text = sent[0], sent[1]
                if sent_id in idx:
                    _gen_sum.append("".join(sent_text))
            gen_sum.append(" ".join(_gen_sum))
    return gen_sum

def load_gold_data(path):
    gold_sum = []
    with open(path) as f:
        for line in f:
            line = json.loads(line.strip())
            gold_sum.append(" ".join(line["summary"]))
    return gold_sum

def get_score(scorer,gen_sum,gold_sum):
    score = scorer.score(gen_sum, gold_sum)
    r1 = score['rouge1'][2]
    r2 = score['rouge2'][2]
    rl = score['rougeL'][2]
    return r1, r2, rl

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
gen_sum = load_gen_data(args.gen_data)


gold_sum = load_gold_data(args.gold_data)
print(len(gen_sum))
print(len(gold_sum))

avg_r1, avg_r2, avg_rl = 0, 0, 0
for i in range(len(gen_sum)):
    _gen_sum = gen_sum[i]
    _gold_sum = gold_sum[i]
    r1, r2, rl = get_score(scorer,_gen_sum,_gold_sum)
    avg_r1 += r1
    avg_r2 += r2
    avg_rl += rl

print("R-1:", avg_r1/len(gen_sum))
print("R-2:", avg_r2/len(gen_sum))
print("R-L:", avg_rl/len(gen_sum))
print("-"*60)
#with open(args.out, 'w') as o_f:
#    json_line = {
#        "R-1:": avg_r1/len(gen_sum),
#        "R-2:": avg_r2/len(gen_sum),
#        "R-L:": avg_rl/len(gen_sum)
#    }
#    for key, value in json_line.items():
#        line = json.dumps({key: value})
#        o_f.write(line + '\n')
