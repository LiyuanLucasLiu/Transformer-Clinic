# WMT14 En-Fr

Here we provide the recipes for Post-LN, Pre-LN, and Admin. Each model would be trained for 50 epochs. 

## Preprocess
```
cd ../pre-process
bash wmt14en-fr.sh

cd ../nmt-experiments
```

## Post-LN

### DEFAULT 6l

```
GPUID=0
GPUS=0,1,2,3,4,5,6,7
TOKEN_NUMBER=8000 
UPDATE_FREQUENCE=10

CUDA_VISIBLE_DEVICES=$GPUS fairseq-train ../data-bin/wmt14_en_fr_joined_dict \
	--arch transformer_wmt_en_de --share-all-embeddings --optimizer radam \
	--adam-betas '(0.9, 0.98)' --clip-norm 0.0 --lr-scheduler inverse_sqrt \
	--warmup-init-lr 1e-07 --warmup-updates 8000 --lr 0.0007 --min-lr 1e-09 \
	--dropout 0.1 --attention-dropout 0.1 --relu-dropout 0.1 --weight-decay 0.0 \
	--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
	--max-tokens $TOKEN_NUMBER --update-freq $UPDATE_FREQUENCE \
	--fp16 --fp16-scale-window 256 --threshold-loss-scale 0.03125 \
	--seed 1111 --restore-file x.pt --max-epoch 50 --save-dir cps/wmt-default-6l \
	--user-dir ../radam_fairseq --init-type default \
	--log-format simple --log-interval 100 | tee ./log/loss-default-6l.log

bash eval_wmt_en-fr.sh cps/wmt-default-6l/ $GPUID
```

### DEFAULT 60-12l

```
GPUID=0
GPUS=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
TOKEN_NUMBER=5000 
UPDATE_FREQUENCE=16

CUDA_VISIBLE_DEVICES=$GPUS fairseq-train ../data-bin/wmt14_en_fr_joined_dict \
	--arch transformer_wmt_en_de --share-all-embeddings --optimizer radam \
	--adam-betas '(0.9, 0.98)' --clip-norm 0.0 --lr-scheduler inverse_sqrt \
	--warmup-init-lr 1e-07 --warmup-updates 8000 --lr 0.0007 --min-lr 1e-09 \
	--dropout 0.1 --attention-dropout 0.1 --relu-dropout 0.1 --weight-decay 0.0 \
	--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
	--max-tokens $TOKEN_NUMBER --update-freq $UPDATE_FREQUENCE \
	--fp16 --fp16-scale-window 256 --threshold-loss-scale 0.03125 \
	--seed 1111 --restore-file x.pt --max-epoch 50 --save-dir cps/wmt-default-60-12l \
	--encoder-layers 60 --decoder-layers 12 \
	--user-dir ../radam_fairseq --init-type default \
	--log-format simple --log-interval 100 | tee ./log/loss-default-60-12l.log

bash eval_wmt_en-fr.sh cps/wmt-default-60-12l/ $GPUID
```

## PreLN

### PreLN 6l

```
GPUID=0
GPUS=0,1,2,3,4,5,6,7
TOKEN_NUMBER=8000 
UPDATE_FREQUENCE=10

CUDA_VISIBLE_DEVICES=$GPUS fairseq-train ../data-bin/wmt14_en_fr_joined_dict \
	--arch transformer_wmt_en_de --share-all-embeddings --optimizer radam \
	--adam-betas '(0.9, 0.98)' --clip-norm 0.0 --lr-scheduler inverse_sqrt \
	--warmup-init-lr 1e-07 --warmup-updates 8000 --lr 0.0007 --min-lr 1e-09 \
	--dropout 0.1 --attention-dropout 0.1 --relu-dropout 0.1 --weight-decay 0.0 \
	--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
	--max-tokens $TOKEN_NUMBER --update-freq $UPDATE_FREQUENCE \
	--fp16 --fp16-scale-window 256 --threshold-loss-scale 0.03125 \
	--seed 1111 --restore-file x.pt --max-epoch 50 --save-dir cps/wmt-preln-6l \
	--encoder-normalize-before --decoder-normalize-before \
	--user-dir ../radam_fairseq --init-type default \
	--log-format simple --log-interval 100 | tee ./log/loss-preln-6l.log

bash eval_wmt_en-fr.sh cps/wmt-preln-6l/ $GPUID
```

### PreLN 60-12l

```
GPUID=0
GPUS=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
TOKEN_NUMBER=5000 
UPDATE_FREQUENCE=16

CUDA_VISIBLE_DEVICES=$GPUS fairseq-train ../data-bin/wmt14_en_fr_joined_dict \
	--arch transformer_wmt_en_de --share-all-embeddings --optimizer radam \
	--adam-betas '(0.9, 0.98)' --clip-norm 0.0 --lr-scheduler inverse_sqrt \
	--warmup-init-lr 1e-07 --warmup-updates 8000 --lr 0.0007 --min-lr 1e-09 \
	--dropout 0.1 --attention-dropout 0.1 --relu-dropout 0.1 --weight-decay 0.0 \
	--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
	--max-tokens $TOKEN_NUMBER --update-freq $UPDATE_FREQUENCE \
	--fp16 --fp16-scale-window 256 --threshold-loss-scale 0.03125 \
	--seed 1111 --restore-file x.pt --max-epoch 50 --save-dir cps/wmt-preln-60-12l \
	--encoder-layers 60 --decoder-layers 12 --encoder-normalize-before --decoder-normalize-before \
	--user-dir ../radam_fairseq --init-type default \
	--log-format simple --log-interval 100 | tee ./log/loss-preln-60-12l.log

bash eval_wmt_en-fr.sh cps/wmt-preln-60-12l/ $GPUID
```

## admin

### admin 6l

```
GPUID=0
GPUS=0,1,2,3,4,5,6,7
TOKEN_NUMBER=8000 
UPDATE_FREQUENCE=10

CUDA_VISIBLE_DEVICES=$GPUID fairseq-train ../data-bin/wmt14_en_fr_joined_dict \
	--arch transformer_wmt_en_de --share-all-embeddings --optimizer radam \
	--adam-betas '(0.9, 0.98)' --clip-norm 0.0 --lr-scheduler inverse_sqrt \
	--warmup-init-lr 1e-07 --warmup-updates 8000 --lr 0.0007 --min-lr 1e-09 \
	--dropout 0.1 --attention-dropout 0.1 --relu-dropout 0.1 --weight-decay 0.0 \
	--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
	--max-tokens $TOKEN_NUMBER --update-freq $UPDATE_FREQUENCE \
	--fp16 --fp16-scale-window 256 --threshold-loss-scale 0.03125 \
	--seed 1111 --restore-file x.pt --max-epoch 50 --save-dir cps/wmt-admin-6l \
	--user-dir ../radam_fairseq --init-type adaptive-profiling \
	--log-format simple --log-interval 100 | tee ./log/loss-admin-6l.log

CUDA_VISIBLE_DEVICES=$GPUS fairseq-train ../data-bin/wmt14_en_fr_joined_dict \
	--arch transformer_wmt_en_de --share-all-embeddings --optimizer radam \
	--adam-betas '(0.9, 0.98)' --clip-norm 0.0 --lr-scheduler inverse_sqrt \
	--warmup-init-lr 1e-07 --warmup-updates 8000 --lr 0.0007 --min-lr 1e-09 \
	--dropout 0.1 --attention-dropout 0.1 --relu-dropout 0.1 --weight-decay 0.0 \
	--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
	--max-tokens $TOKEN_NUMBER --update-freq $UPDATE_FREQUENCE \
	--fp16 --fp16-scale-window 256 --threshold-loss-scale 0.03125 \
	--seed 1111 --restore-file x.pt --max-epoch 50 --save-dir cps/wmt-admin-6l \
	--user-dir ../radam_fairseq --init-type adaptive \
	--log-format simple --log-interval 100 | tee ./log/loss-admin-6l.log

bash eval_wmt_en-fr.sh cps/wmt-admin-6l/ $GPUID
```

### admin 60-12l

```
GPUID=0
GPUS=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
TOKEN_NUMBER=5000 
UPDATE_FREQUENCE=16
CUDA_VISIBLE_DEVICES=$GPUID fairseq-train ../data-bin/wmt14_en_fr_joined_dict \
	--arch transformer_wmt_en_de --share-all-embeddings --optimizer radam \
	--adam-betas '(0.9, 0.98)' --clip-norm 0.0 --lr-scheduler inverse_sqrt \
	--warmup-init-lr 1e-07 --warmup-updates 8000 --lr 0.0007 --min-lr 1e-09 \
	--dropout 0.1 --attention-dropout 0.1 --relu-dropout 0.1 --weight-decay 0.0 \
	--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
	--max-tokens $TOKEN_NUMBER --update-freq $UPDATE_FREQUENCE \
	--fp16 --fp16-scale-window 256 --threshold-loss-scale 0.03125 \
	--seed 1111 --restore-file x.pt --max-epoch 50 --save-dir cps/wmt-admin-60-12l \
	--encoder-layers 60 --decoder-layers 12 \
	--user-dir ../radam_fairseq --init-type adaptive-profiling \
	--log-format simple --log-interval 100 | tee ./log/loss-admin-60-12l.log

CUDA_VISIBLE_DEVICES=$GPUS fairseq-train ../data-bin/wmt14_en_fr_joined_dict \
	--arch transformer_wmt_en_de --share-all-embeddings --optimizer radam \
	--adam-betas '(0.9, 0.98)' --clip-norm 0.0 --lr-scheduler inverse_sqrt \
	--warmup-init-lr 1e-07 --warmup-updates 8000 --lr 0.0007 --min-lr 1e-09 \
	--dropout 0.1 --attention-dropout 0.1 --relu-dropout 0.1 --weight-decay 0.0 \
	--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
	--max-tokens $TOKEN_NUMBER --update-freq $UPDATE_FREQUENCE \
	--fp16 --fp16-scale-window 256 --threshold-loss-scale 0.03125 \
	--seed 1111 --restore-file x.pt --max-epoch 50 --save-dir cps/wmt-admin-60-12l \
	--encoder-layers 60 --decoder-layers 12 \
	--user-dir ../radam_fairseq --init-type adaptive \
	--log-format simple --log-interval 100 | tee ./log/loss-admin-60-12l.log

bash eval_wmt_en-fr.sh cps/wmt-admin-60-12l/ $GPUID
```
