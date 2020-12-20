# IWSLT De-En

Here we provide the recipes for Post-LN, Pre-LN, Admin, ReZero and ReZero+Post-LN. Each model would be trained for 90 epochs. 

## Preprocessing

```
cd ../pre-process
bash iwslt14de-en.sh

cd ../nmt-experiments
```

## Post-LN
```
GPUID=1
TOKEN_NUMBER=4096
UPDATE_FREQUENCE=1

for SEED in 1111 2222 3333 4444 5555
do
  CUDA_VISIBLE_DEVICES=$GPUID fairseq-train \
    ../data-bin/iwslt14.tokenized.de-en.joined -s de -t en \
    --arch transformer_iwslt_de_en --share-all-embeddings \
    --user-dir ../radam_fairseq --optimizer radam --adam-betas "(0.9,0.98)" \
    --clip-norm 0.0 --lr 7e-4 --lr-scheduler inverse_sqrt \
    --warmup-init-lr 1e-7 --warmup-updates 6000 --max-update 100000 \
    --dropout 0.3 --attention-dropout 0.1 --relu-dropout 0.1 \
    --weight-decay 0.0001 --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 --save-dir iwslt14deen/iwslt-default-$SEED \
    --init-type default --max-tokens $TOKEN_NUMBER \
    --update-freq $UPDATE_FREQUENCE --seed $SEED \
    --log-format simple --fp16 --restore-file x.pt \
    --fp16-scale-window 256 --threshold-loss-scale 0.03125 \
    --log-interval 100 | tee ./iwslt14deen/log/loss_default_$SEED.log

  bash eval_iwslt_de-en.sh iwslt14deen/iwslt-default-$SEED $GPUID 
done
```

## Pre-LN
```
GPUID=1
TOKEN_NUMBER=4096
UPDATE_FREQUENCE=1

for SEED in 1111 2222 3333 4444 5555
do
  CUDA_VISIBLE_DEVICES=$GPUID fairseq-train \
    ../data-bin/iwslt14.tokenized.de-en.joined -s de -t en \
    --arch transformer_iwslt_de_en --share-all-embeddings \
    --user-dir ../radam_fairseq --optimizer radam --adam-betas "(0.9,0.98)" \
    --clip-norm 0.0 --lr 7e-4 --lr-scheduler inverse_sqrt \
    --warmup-init-lr 1e-7 --warmup-updates 6000 --max-update 100000 \
    --dropout 0.3 --attention-dropout 0.1 --relu-dropout 0.1 \
    --weight-decay 0.0001 --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 --save-dir iwslt14deen/iwslt-preln-$SEED \
    --init-type default --max-tokens $TOKEN_NUMBER \
    --update-freq $UPDATE_FREQUENCE --seed $SEED \
    --log-format simple --fp16 --restore-file x.pt \
    --fp16-scale-window 256 --threshold-loss-scale 0.03125 \
    --encoder-normalize-before --decoder-normalize-before \
    --log-interval 100 | tee ./iwslt14deen/log/loss_preln_$SEED.log

  bash eval_iwslt_de-en.sh iwslt14deen/iwslt-preln-$SEED $GPUID 
done
```

## Admin
```
GPUID=1
TOKEN_NUMBER=4096
UPDATE_FREQUENCE=1

CUDA_VISIBLE_DEVICES=$GPUID fairseq-train \
  ../data-bin/iwslt14.tokenized.de-en.joined -s de -t en \
  --arch transformer_iwslt_de_en --share-all-embeddings \
  --user-dir ../radam_fairseq --optimizer radam --adam-betas "(0.9,0.98)" \
  --clip-norm 0.0 --lr 7e-4 --lr-scheduler inverse_sqrt \
  --warmup-init-lr 1e-7 --warmup-updates 6000 --max-update 100000 \
  --dropout 0.3 --attention-dropout 0.1 --relu-dropout 0.1 \
  --weight-decay 0.0001 --criterion label_smoothed_cross_entropy \
  --label-smoothing 0.1 --save-dir iwslt14deen/iwslt-preln-1111 \
  --init-type adaptive-profiling --max-tokens $TOKEN_NUMBER \
  --update-freq $UPDATE_FREQUENCE --seed 1111 \
  --log-format simple --fp16 --restore-file x.pt \
  --fp16-scale-window 256 --threshold-loss-scale 0.03125 \
  --log-interval 100

for SEED in 1111 2222 3333 4444 5555
do
  CUDA_VISIBLE_DEVICES=$GPUID fairseq-train \
    ../data-bin/iwslt14.tokenized.de-en.joined -s de -t en \
    --arch transformer_iwslt_de_en --share-all-embeddings \
    --user-dir ../radam_fairseq --optimizer radam --adam-betas "(0.9,0.98)" \
    --clip-norm 0.0 --lr 7e-4 --lr-scheduler inverse_sqrt \
    --warmup-init-lr 1e-7 --warmup-updates 6000 --max-update 100000 \
    --dropout 0.3 --attention-dropout 0.1 --relu-dropout 0.1 \
    --weight-decay 0.0001 --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 --save-dir iwslt14deen/iwslt-preln-$SEED \
    --init-type adaptive --max-tokens $TOKEN_NUMBER \
    --update-freq $UPDATE_FREQUENCE --seed $SEED \
    --log-format simple --fp16 --restore-file x.pt \
    --fp16-scale-window 256 --threshold-loss-scale 0.03125 \
    --log-interval 100 | tee ./iwslt14deen/log/loss_preln_$SEED.log

  bash eval_iwslt_de-en.sh iwslt14deen/iwslt-preln-$SEED $GPUID 
done
```

## ReZero
```
# 34.66 (2 BLEU points lower than other models)
GPUID=1
TOKEN_NUMBER=4096
UPDATE_FREQUENCE=1

for SEED in 1111 2222 3333 4444 5555
do
  CUDA_VISIBLE_DEVICES=$GPUID fairseq-train \
    ../data-bin/iwslt14.tokenized.de-en.joined -s de -t en \
    --arch transformer_iwslt_de_en --share-all-embeddings \
    --user-dir ../radam_fairseq --optimizer radam --adam-betas "(0.9,0.98)" \
    --clip-norm 0.0 --lr 7e-4 --lr-scheduler inverse_sqrt \
    --warmup-init-lr 1e-7 --warmup-updates 6000 --max-update 100000 \
    --dropout 0.3 --attention-dropout 0.1 --relu-dropout 0.1 \
    --weight-decay 0.0001 --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 --save-dir iwslt14deen/iwslt-rezero-$SEED \
    --init-type rezero --max-tokens $TOKEN_NUMBER \
    --update-freq $UPDATE_FREQUENCE --seed $SEED \
    --log-format simple --fp16 --restore-file x.pt \
    --fp16-scale-window 256 --threshold-loss-scale 0.03125 \
    --log-interval 100 | tee ./iwslt14deen/log/loss_rezero_$SEED.log

  bash eval_iwslt_de-en.sh iwslt14deen/iwslt-rezero-$SEED $GPUID 
done
```


## ReZero + PostLN
```
# 34.66 (1 BLEU point lower than other models)
GPUID=1
TOKEN_NUMBER=4096
UPDATE_FREQUENCE=1

for SEED in 1111 2222 3333 4444 5555
do
  CUDA_VISIBLE_DEVICES=$GPUID fairseq-train \
    ../data-bin/iwslt14.tokenized.de-en.joined -s de -t en \
    --arch transformer_iwslt_de_en --share-all-embeddings \
    --user-dir ../radam_fairseq --optimizer radam --adam-betas "(0.9,0.98)" \
    --clip-norm 0.0 --lr 7e-4 --lr-scheduler inverse_sqrt \
    --warmup-init-lr 1e-7 --warmup-updates 6000 --max-update 100000 \
    --dropout 0.3 --attention-dropout 0.1 --relu-dropout 0.1 \
    --weight-decay 0.0001 --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 --save-dir iwslt14deen/iwslt-rezero_postln-$SEED \
    --init-type rezero_postln --max-tokens $TOKEN_NUMBER \
    --update-freq $UPDATE_FREQUENCE --seed $SEED \
    --log-format simple --fp16 --restore-file x.pt \
    --fp16-scale-window 256 --threshold-loss-scale 0.03125 \
    --log-interval 100 | tee ./iwslt14deen/log/loss_rezero_postln_$SEED.log

  bash eval_iwslt_de-en.sh iwslt14deen/iwslt-rezero_postln-$SEED $GPUID 
done
```
