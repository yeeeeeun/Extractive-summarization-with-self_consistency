import argparse
import os
import json
import torch

import bert_score
from tqdm import tqdm

from transformers import AutoModel, AutoTokenizer

import os
os.environ["HF_HOME"] = "/data/huggingface_models/"


def main():
    torch.multiprocessing.set_sharing_strategy("file_system")

    parser = argparse.ArgumentParser("Calculate BERTScore")
    parser.add_argument(
        "--lang",
        type=str,
        default="en",
        help='two-letter abbreviation of the language (e.g., en) or "en-sci" for scientific text',
    )
    parser.add_argument(
        "-m",
        "--model",
        default="bert-base-uncased",
        help="BERT model name (default: bert-base-uncased) or path to a pretrain model",
    )
    parser.add_argument(
        "-l",
        "--num_layers",
        type=int,
        default=None,
        help="use first N layer in BERT (default: 8)",
    )
    parser.add_argument(
        "-b", "--batch_size", type=int, default=256, help="batch size (default: 64)"
    )
    parser.add_argument(
        "--nthreads", type=int, default=4, help="number of cpu workers (default: 4)"
    )
    parser.add_argument(
        "--idf", action="store_true", help="BERT Score with IDF scaling"
    )
    parser.add_argument(
        "--rescale_with_baseline",
        action="store_true",
        help="Rescaling the numerical score with precomputed baselines",
    )
    parser.add_argument(
        "--baseline_path",
        default=None,
        type=str,
        help="path of custom baseline csv file",
    )
    parser.add_argument(
        "--use_fast_tokenizer",
        action="store_false",
        help="whether to use HF fast tokenizer",
    )
    parser.add_argument(
        "-s",
        "--seg_level",
        action="store_true",
        help="show individual score of each pair",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="increase output verbosity"
    )
    parser.add_argument(
        "-r",
        "--ref",
        type=str,
        # nargs= "+", #하나의 요약문이므로 주석처리
    #    required=True,
        help="reference file path(s) or a string",
    )
    parser.add_argument(
        "--summary",
        type=str,
        required=True,
        help="candidate (system outputs) file path or a string",
    )
    #output args
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="bert_scores.txt",
        help="output file path to store BERT scores",
    )

    args = parser.parse_args()

    golds, gens = [], []
    with open(args.summary) as f:
        for line in f:
            line = json.loads(line.strip())
            gold = line["gold_summary"]
            #gen = line["onebyone"]
            #gen=line["all"]
            gen=line["tri_sents"]
            golds.append(gold)
            gens.append(gen)

    assert len(golds) == len(gens)

    all_preds, hash_code = bert_score.score(
        golds,
        gens,
        model_type=args.model,
        num_layers=args.num_layers,
        verbose=args.verbose,
        idf=args.idf,
        batch_size=args.batch_size,
        rescale_with_baseline=args.rescale_with_baseline,
        lang=args.lang,
        return_hash=True,
        baseline_path=args.baseline_path,
        use_fast_tokenizer=args.use_fast_tokenizer,
    )
    final_score = 0.0
    for i, f1_score in tqdm(enumerate(all_preds[2].tolist())):
            final_score += f1_score

    final_bert_score = final_score/len(all_preds[2])
    print("bert_scre :", final_bert_score)
#    with open(args.output,'a+') as f_out:
#        line = "\n" + "joint_triplet2 : " + str(final_bert_score)
#        f_out.write(line)

if __name__ == "__main__":
    main()
