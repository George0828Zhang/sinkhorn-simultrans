#!/usr/bin/env bash
WAITK=3
TASK=wait_${WAITK}_iwslt_distill
. ./data_path.sh

export CUDA_VISIBLE_DEVICES=0

python -m fairseq_cli.train ${DATA} --user-dir ${USERDIR} \
    -s ${SRC} -t ${TGT} \
    --train-subset train_distill \
    --max-tokens 8000 \
    --update-freq 2 \
    --task translation_infer \
    --inference-config-yaml infer_mt.yaml \
    --arch waitk_transformer_small \
    --waitk ${WAITK} \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --clip-norm 10.0 \
    --weight-decay 0.0001 \
    --optimizer adam --lr 5e-4 --lr-scheduler inverse_sqrt \
    --warmup-updates 4000 \
    --max-update 50000 \
    --save-dir checkpoints/${TASK} \
    --no-epoch-checkpoints \
    --save-interval-updates 500 \
    --keep-interval-updates 5 \
    --keep-best-checkpoints 5 \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --wandb-project simulmt \
    --patience 50 \
    --log-format simple --log-interval 50 \
    --num-workers 8 \
    --fp16 \
    --seed 2