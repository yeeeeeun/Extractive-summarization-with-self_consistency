data=PubMed

beam_size=(4 6 8)

for beam in ${beam_size[@]}; do
        echo "beam_size=$beam"

        python all_voting.py --input="../beam_output/${data}/${data}_beam_${beam}_output.json" \
            --beam_size $beam \
            --gold ../gold_data/all_${data}_test.json \
            --out ./res/$data/$_type/${data}_beam_${beam}_all.json
done
