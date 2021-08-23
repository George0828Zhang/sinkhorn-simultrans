#!/usr/bin/env bash
. ./data_path.sh
WAITK=$1
TASK=wait_${WAITK}_${SRC}${TGT}_mon

python -m fairseq_cli.train ${DATA} --user-dir ${USERDIR} \
    -s ${SRC} -t ${TGT} \
    --train-subset train_monotonic_${TGT} --upsample-primary 1 \
    --max-tokens 8000 \
    --update-freq 4 \
    --task translation_infer \
    --inference-config-yaml infer_mt.yaml \
    --arch waitk_transformer \
    --waitk ${WAITK} \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --clip-norm 10.0 \
    --weight-decay 0.0001 \
    --optimizer adam --lr 5e-4 --lr-scheduler inverse_sqrt \
    --warmup-updates 4000 \
    --max-update 300000 \
    --save-dir checkpoints/${TASK} \
    --no-epoch-checkpoints \
    --save-interval-updates 500 \
    --keep-interval-updates 1 \
    --keep-best-checkpoints 1 \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --wandb-project sinkhorn-cwmt \
    --patience 50 \
    --log-format simple --log-interval 50 \
    --num-workers 4 \
    --fp16 \
    --seed 2