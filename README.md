# Extractive-summarization-with-self_consistency
## 📂 Structure

folder has the following structure:
```
self_consistency_
├── dataset_sample
│   ├── pubmed_trunc
│       ├──all_pubmed_trun_test.json
│       ├──all_pubmed_trun_train.json
│       └──all_pubmed_trun_val.json
|
├── src
│   ├── data_construction
│   │   └──get_high_rouge_episodes_sp.py
|   |   └──utils.py
|   |
│   ├── evalutation
│   │   └──cal_rouge.py
│   │   └──get_bertscore.py
|   |
│   ├── finetuning
│   │   └──_finetuning.py
│   │
│   ├── inference
│   │   └──inference.py
|   |
│   └── proposal
|      └──all_voting
│       ├──all_votiong.py
│       ├──all_votiong.sh
|      └──surface_votiong
│       ├──surface_voting_new.py
│       ├──surface_voting_new.sh
│       ├──surface_voting.py
│       ├──surface_voting.sh
|      └──top_voting
│       ├──top_voting.py
│       ├──top_voting.sh
|      └──trigram_blocking
│       ├──tri.py
└──     ├──tri.sh

```
