data=PubMed

beam_size=(4 6 8)
types=(min avg max)

for beam in ${beam_size[@]}; do
    for _type in ${types[@]}; do
        echo "beam_size=$beam, type=$_type"

        python top_voting.py --input="../beam_output/${data}/${data}_beam_${beam}_output.json" \
            --beam_size $beam \
            --gold ../gold_data/all_${data}_test.json \
            --len_type $_type \
            --out ./res/$data/$_type/${data}_beam_${beam}_${_type}_top.json

    done
done