import argparse
import re, json, os
from collections import Counter

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
    all_keys = [k for k, v in sorted(vot_count.items(), key=lambda x:x[1], reverse=True)]
    return all_keys

def _get_ngrams(n, text):
            ngram_set = set()
            text_length = len(text)
            max_index_ngram_start = text_length - n
            for i in range(max_index_ngram_start + 1):
                ngram_set.add(tuple(text[i:i + n]))
            return ngram_set

def _block_tri(c, p):
    tri_c = _get_ngrams(3, c.split())
    for s in p:
        tri_s = _get_ngrams(3, s.split())
        if len(tri_c.intersection(tri_s))>0:
            # print(f"tri_s {tri_s}")
            return True 
    return False

def main():
    size = args.beam_size
    doc, buffer, all_keys = [], [], []
    with open(args.gold) as f:
            gold = []
            for line in f:
                data = json.loads(line)
                gold.append(" ".join(data["summary"]))
    with open(args.input) as f:
        for i, line in enumerate(f):
            voting_list, selected_ids = [], []
            buffer.append(json.loads(line))            
            
            if (i+1)%size==0:
                doc = json.loads(line)["input"]
                gold_summary = gold[i // size]
                sentences = dict(re.findall(r'\[sent(\d+)\]\s*([^[]+)', doc))
                _pred = []
                for entry in buffer:
                    _output = list(map(int, re.findall(r'\d+',entry["output"])))
                    voting_list.extend(_output)
                
                all_keys = sorted([int(x) for x in all_voting(voting_list)])
                
                blocked_ids = []
                selected_sentences = []
                
                for idx in range(len(all_keys)):
                    if str(all_keys[idx]) not in sentences:
                        continue
                    candidate = sentences[str(all_keys[idx])] 
                    if not _block_tri(candidate, selected_sentences): 
                        selected_sentences.append(candidate)
                        blocked_ids.append(all_keys[idx])
                    # else:
                        # blocked_ids.append(all_keys[idx])
                
                
                _res = " ".join(["[sent"+str(x)+"]" for x in blocked_ids])
                # selected_ids = [sel_idx for sel_idx in all_keys if sel_idx not in blocked_ids]
                # _res = " ".join(["[sent"+str(x)+"]" for x in selected_ids])
                tri_sents = ' '.join([sentences[str(i)].strip() for i in blocked_ids if str(i) in sentences.keys()])
                with open(args.out, 'a') as f:
                    f.write(json.dumps({
                        "input": doc,
                        "gold_summary": gold_summary,
                        "output": _res,
                        "selected_ids": blocked_ids,
                        "tri_sents": tri_sents
                    }) + '\n')
                
                buffer = []    

if __name__ == "__main__":
    main()