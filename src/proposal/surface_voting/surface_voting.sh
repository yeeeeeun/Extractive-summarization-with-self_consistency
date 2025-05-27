data=PubMed_trunc
beam_size=8
len_type=max


# python surface_voting.py --input ../beam_output/add_length/$data/${data}_beam_${beam_size}_output.json \
#     --beam_size=$beam_size \
#     --gold ../gold_data/all_${data}_test.json \
#     --len_type=$len_type \
#     --out ./res/add_length/random/$data/$len_type/len_${data}_beam_${beam_size}_${len_type}.json

python surface_voting.py --input ../beam_output/lora/${data}/${data}_beam_${beam_size}_output.json \
    --beam_size=$beam_size \
    --gold ../gold_data/all_${data}_test.json \
    --len_type=$len_type \
    --out ./res/first_out_weight/lora/$data/$len_type/lora_${data}_beam_${beam_size}_${len_type}.json
