data=PubMed_trunc
beam_size=8
len_type=max


python tri.py --input ../beam_output/lora/${data}/${data}_beam_${beam_size}_output.json \
    --beam_size=$beam_size \
    --gold ../gold_data/all_${data}_test.json \
    --out ./res/lora/random/$data/$len_type/full_${data}_beam_${beam_size}_${len_type}_hy.json