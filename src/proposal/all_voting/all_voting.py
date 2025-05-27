import json
import re, os
from collections import Counter
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--input')
parser.add_argument('--beam_size', type=int, default=4)
parser.add_argument('--gold', default="./gold_data/all_pubmed_trun_test.json")
parser.add_argument('--out')

args = parser.parse_args()

out_dir=os.path.dirname(args.out)
if not os.path.isdir(out_dir):
    os.makedirs(out_dir, exist_ok=True)

def all_voting(voting_list):
    vot_count = dict(Counter(voting_list))
    top_keys = [k for k, v in sorted(vot_count.items(), key=lambda x:x[1], reverse=True)]
    return top_keys


def main():
    size = args.beam_size
    
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
                    
                    voting_list.extend(_output)
                                        
                
                res.append(int(x) for x in all_voting(voting_list))
                buffer=[]
                        
                                   

    with open(args.out, 'w') as f:
        for doc, res, _sum in zip(doc, res, gold):
            sentences = dict(re.findall(r'\[sent(\d+)\]\s*([^[]+)', doc))
            res = sorted(res, reverse=False)
            _res = " ".join(["[sent"+str(x)+"]" for x in res])
            try:
                selected = ' '.join([sentences[str(i)].strip() for i in res])
            except:
                # print(sentences)
                # print(res)
                selected = ' '.join([sentences[str(i)].strip() for i in res if str(i) in sentences.keys()])
                # print(f"selected {selected}")
                
            data = {
                "input": doc,
                "summary": _sum,
                "output": _res,
                "all":selected
            }
            json.dump(data, f)
            f.write('\n')

if __name__ == "__main__":
    main()

