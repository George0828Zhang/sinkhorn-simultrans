#!/usr/bin/env bash
. ./data_path.sh
WAITK=$1
TASK=sinkhorn_wait${WAITK}_distill

python -m fairseq_cli.train ${DATA} --user-dir ${USERDIR} \
    -s ${SRC} -t ${TGT} \
    --train-subset train_distill_zh \
    --max-tokens 8000 \
    --update-freq 4 \
    --task translation_infer \
    --inference-config-yaml infer_mt.yaml \
    --arch sinkhorn_waitk --waitk ${WAITK} --mask-ratio 0.5 \
    --sinkhorn-iters 16 --sinkhorn-tau 0.25 --sinkhorn-noise-factor 0.3 --sinkhorn-bucket-size 1 --sinkhorn-energy dot \
    --criterion label_smoothed_ctc --label-smoothing 0.1 --report-sinkhorn-dist \
    --clip-norm 10.0 \
    --weight-decay 0.0001 \
    --optimizer adam --lr 5e-4 --lr-scheduler inverse_sqrt \
    --warmup-updates 4000 \
    --max-update 300000 \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --wandb-project sinkhorn-cwmt \
    --save-dir checkpoints/${TASK} \
    --no-epoch-checkpoints \
    --save-interval-updates 500 \
    --keep-interval-updates 1 \
    --keep-best-checkpoints 1 \
    --patience 50 \
    --log-format simple --log-interval 50 \
    --num-workers 4 \
    --seed 3 \
    --fp16
