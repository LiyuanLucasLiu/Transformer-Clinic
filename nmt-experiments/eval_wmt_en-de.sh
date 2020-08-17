#!/bin/bash
echo "Model path" $SAVEDIR
GPUDEV=${2:-0}
SAVEDIR=${1}
UPPER_BOUND=${3:-100}
CP_POINT_NUM=${4:-10}
MODELDIR=$SAVEDIR/model_${UPPER_BOUND}_${CP_POINT_NUM}.pt
if [ -f $MODELDIR  ]; then
    echo $MODELDIR "already exists"
else
    echo "Start averaging model"
    python average_checkpoints.py --inputs $SAVEDIR --num-epoch-checkpoints ${CP_POINT_NUM}  --output $MODELDIR --checkpoint-upper-bound $UPPER_BOUND | grep 'Finish'
    echo "End averaging model"
fi

CUDA_VISIBLE_DEVICES=$GPUDEV fairseq-generate ../data-bin/wmt14_en_de_joined_dict \
                    --path $MODELDIR \
                    --batch-size 128 --beam 4 --lenpen 0.6 --remove-bpe \
                    --user-dir ../radam_fairseq --quiet --fp16
