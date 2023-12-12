#!/bin/bash

gpt_sizes=(124M 355M 774M 1558M)
skip_rates=(0 0.2 0.4)

for sz in "${gpt_sizes[@]}"; do 
    for skip in "${skip_rates[@]}"; do
        dir_name="gpt2-${sz}-${skip}"
        # rm -rf "models/$dir_name"
        mkdir -p "models/$dir_name"

        build/examples/transformer-lm -c "models/gpt2-${sz}/hparams.ini" --model-path "models/$dir_name" --attention-dropout-p "$skip" --ff-dropout-p "$skip" --reset-if-stuck --use-smaller-minibatch --dao-profile 1 | tee "models/$dir_name/train.log"
    done
done
