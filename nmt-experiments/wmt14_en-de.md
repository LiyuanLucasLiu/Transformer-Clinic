# WMT14 En-De

Here we provide the recipes for Post-LN, Pre-LN, and Admin.

## Preprocessing

```
cd ../pre-process
bash wmt14en-de.sh

cd ../nmt-experiments
```

## Post-LN
```
GPUS=0,1,2,3
GPUID=1
TOKEN_NUMBER=8192
UPDATE_FREQUENCE=1

for lnum in 6 12 18
do
  CUDA_VISIBLE_DEVICES=$GPUS fairseq-train \
    ../data-bin/wmt14_en_de_joined_dict/ -s en -t de \
    --arch transformer_wmt_en_de --share-all-embeddings \
    --optimizer radam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr-scheduler inverse_sqrt --max-update 500000 \
    --warmup-init-lr 1e-07 --warmup-updates 8000 --lr 0.001 --min-lr 1e-09  \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --weight-decay 0.0 --attention-dropout 0.1 --relu-dropout 0.1 \
    --max-tokens $TOKEN_NUMBER --update-freq $UPDATE_FREQUENCE \
    --save-dir wmt14ende/wmt-default-${lnum}l --restore-file x.pt --seed 1111 \
    --user-dir ../radam_fairseq --log-format simple --log-interval 500 \
    --init-type default --fp16 --fp16-scale-window 256 \
    --encoder-layers $lnum --decoder-layers $lnum \
    --threshold-loss-scale 0.03125 | tee ./wmt14ende/log/loss_default-${lnum}l.log

  bash eval_wmt_en-de.sh wmt14ende/wmt-default-${lnum}l $GPUID 
done 
```

## Pre-LN
```
GPUS=0,1,2,3
GPUID=1
TOKEN_NUMBER=8192
UPDATE_FREQUENCE=1

for lnum in 6 12 18
do
  CUDA_VISIBLE_DEVICES=$GPUS fairseq-train \
    ../data-bin/wmt14_en_de_joined_dict/ -s en -t de \
    --arch transformer_wmt_en_de --share-all-embeddings \
    --optimizer radam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr-scheduler inverse_sqrt --max-update 500000 \
    --warmup-init-lr 1e-07 --warmup-updates 8000 --lr 0.001 --min-lr 1e-09  \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --weight-decay 0.0 --attention-dropout 0.1 --relu-dropout 0.1 \
    --max-tokens $TOKEN_NUMBER --update-freq $UPDATE_FREQUENCE \
    --save-dir wmt14ende/wmt-preln-${lnum}l --restore-file x.pt --seed 1111 \
    --user-dir ../radam_fairseq --log-format simple --log-interval 500 \
    --init-type default --fp16 --fp16-scale-window 256 \
    --encoder-layers $lnum --decoder-layers $lnum \
    --encoder-normalize-before --decoder-normalize-before \
    --threshold-loss-scale 0.03125 | tee ./wmt14ende/log/loss_preln-${lnum}l.log

  bash eval_wmt_en-de.sh wmt14ende/wmt-preln-${lnum}l $GPUID 
done 
```

## Admin

```
GPUS=0,1,2,3
GPUID=1
TOKEN_NUMBER=8192
UPDATE_FREQUENCE=1
for lnum in 6 12 18
do
  CUDA_VISIBLE_DEVICES=$GPUID fairseq-train \
    ../data-bin/wmt14_en_de_joined_dict/ -s en -t de \
    --arch transformer_wmt_en_de --share-all-embeddings \
    --optimizer radam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr-scheduler inverse_sqrt --max-update 500000 \
    --warmup-init-lr 1e-07 --warmup-updates 8000 --lr 0.001 --min-lr 1e-09  \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --weight-decay 0.0 --attention-dropout 0.1 --relu-dropout 0.1 \
    --max-tokens $TOKEN_NUMBER --update-freq $UPDATE_FREQUENCE \
    --save-dir wmt14ende/wmt-admin-${lnum}l --restore-file x.pt --seed 1111 \
    --user-dir ../radam_fairseq --log-format simple --log-interval 500 \
    --init-type adaptive-profiling --fp16 --fp16-scale-window 256 \
    --encoder-layers $lnum --decoder-layers $lnum \
    --threshold-loss-scale 0.03125 

  CUDA_VISIBLE_DEVICES=$GPUS fairseq-train \
    ../data-bin/wmt14_en_de_joined_dict/ -s en -t de \
    --arch transformer_wmt_en_de --share-all-embeddings \
    --optimizer radam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr-scheduler inverse_sqrt --max-update 500000 \
    --warmup-init-lr 1e-07 --warmup-updates 8000 --lr 0.001 --min-lr 1e-09  \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --weight-decay 0.0 --attention-dropout 0.1 --relu-dropout 0.1 \
    --max-tokens $TOKEN_NUMBER --update-freq $UPDATE_FREQUENCE \
    --save-dir wmt14ende/wmt-admin-${lnum}l --restore-file x.pt --seed 1111 \
    --user-dir ../radam_fairseq --log-format simple --log-interval 500 \
    --init-type adaptive --fp16 --fp16-scale-window 256 \
    --encoder-layers $lnum --decoder-layers $lnum \
    --threshold-loss-scale 0.03125 | tee ./wmt14ende/log/loss_admin-${lnum}l.log

  bash eval_wmt_en-de.sh wmt14ende/wmt-admin-${lnum}l $GPUID 
done 
```

## Other Initializations
```
GPUS=0,1,2,3
GPUID=1
TOKEN_NUMBER=8192
UPDATE_FREQUENCE=1

for init in rezero looklinear
do
  CUDA_VISIBLE_DEVICES=$GPUS fairseq-train \
    ../data-bin/wmt14_en_de_joined_dict/ -s en -t de \
    --arch transformer_wmt_en_de --share-all-embeddings \
    --optimizer radam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr-scheduler inverse_sqrt --max-update 500000 \
    --warmup-init-lr 1e-07 --warmup-updates 8000 --lr 0.001 --min-lr 1e-09  \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --weight-decay 0.0 --attention-dropout 0.1 --relu-dropout 0.1 \
    --max-tokens $TOKEN_NUMBER --update-freq $UPDATE_FREQUENCE \
    --save-dir wmt14ende/wmt-${init} --restore-file x.pt --seed 1111 \
    --user-dir ../radam_fairseq --log-format simple --log-interval 500 \
    --init-type ${init} --fp16 --fp16-scale-window 256 \
    --encoder-layers 18 --decoder-layers 18 \
    --threshold-loss-scale 0.03125 | tee ./wmt14ende/log/loss_${init}.log

    bash eval_wmt_en-de.sh wmt14ende/wmt-${init} $GPUID 
done

```