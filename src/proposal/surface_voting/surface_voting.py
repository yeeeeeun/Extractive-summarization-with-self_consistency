import json
import re, os
import numpy as np
from collections import Counter
import random
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--input')
parser.add_argument('--beam_size', type=int, default=4)
parser.add_argument('--gold', default="./gold_data/all_pubmed_trun_test.json")
parser.add_argument('--len_type',choices=["min","max","avg"])
parser.add_argument('--out')

args = parser.parse_args()

out_dir=os.path.dirname(args.out)
if not os.path.isdir(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    

def surface_voting(voting_list, vot_len): 
    res = []
    for i in range(vot_len):
        buffer = dict(Counter([voting_list[j][i] for j in range(len(voting_list)) if len(voting_list[j])>i]))
        buffer = {k: v for k, v in buffer.items() if k not in res}
        max_key = [k for k, v in buffer.items() if max(buffer.values())==v] 
        
        if len(max_key)==1:            
            res.append(max_key[0])
        else:
            try:
                # np.random.seed(1004)
                # res.append(np.random.choice(max_key))
                if len(max_key)>0:
                    res.append(max_key[0])
                else:
                    res.append("100000")            
            except ValueError:
                return res #이미 앞에 뽑힌 index 제거해서 max_key에 없는 경우 vot_len보다 짧은 길이 있을 수 있음
    return res


def main():
    size = args.beam_size
    len_type = args.len_type
    with open(args.gold) as f:
        gold = []
        for line in f:
            data = json.loads(line)
            gold.append(" ".join(data["summary"]))
        
    with open(args.input) as f:
        doc, buffer, res = [], [], []
        for i, line in enumerate(f):
            voting_list = []
            max_vot_len = 0
            vot_len = 100
            buffer.append(json.loads(line))            

            if (i+1)%size==0:
                doc.append(json.loads(line)["input"])
                for entry in buffer:
                    _output = list(map(int, re.findall(r'\d+',entry["output"])))
                    
                    if len_type=="min":
                        vot_len = len(_output) if len(_output)<vot_len else vot_len
                    elif len_type=="max":
                        max_vot_len = len(_output) if len(_output)>max_vot_len else max_vot_len

                    voting_list.append(_output)
                
                if len_type=="avg":
                    vot_len=0
                    for vot in voting_list:
                        vot_len += len(vot)
                    vot_len = round(vot_len/len(voting_list))
                    
                if len_type=="max":
                    res.append(int(x) for x in surface_voting(voting_list, max_vot_len))
                elif len_type=="min" or len_type=="avg":
                    res.append(int(x) for x in surface_voting(voting_list, vot_len))
                buffer=[]
            
            
    with open(args.out, 'w') as f:
        for doc, res, _sum in zip(doc, res, gold):
            sentences = dict(re.findall(r'\[sent(\d+)\]\s*([^[]+)', doc))
            res = sorted(res, reverse=False)
            _res = " ".join(["[sent"+str(x)+"]" for x in res])
            try:
                selected = ' '.join([sentences[str(i)].strip() for i in res])
            except:
                print(sentences)
                print(res)
                selected = ' '.join([sentences[str(i)].strip() for i in res if str(i) in sentences.keys()])
                print(f"selected {selected}")
                
            data = {
                "input": doc,
                "summary": _sum,
                "output": _res,
                "onebyone":selected
            }
            json.dump(data, f)
            f.write('\n')

if __name__ == "__main__":
    main()








