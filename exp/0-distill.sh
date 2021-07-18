#!/usr/bin/env bash
TASK=teacher_wmt15_deen
. ./data_path.sh

python -m fairseq_cli.train ${DATA} --user-dir ${USERDIR} \
    -s ${SRC} -t ${TGT} \
    --max-tokens 16000 \
    --update-freq 8 \
    --task translation \
    --arch transformer \
    --encoder-normalize-before --decoder-normalize-before \
    --share-decoder-input-output-embed \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --clip-norm 10.0 \
    --weight-decay 0.0001 \
    --optimizer adam --lr 0.0005 --lr-scheduler inverse_sqrt \
    --warmup-updates 4000 \
    --max-update 300000 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-remove-bpe sentencepiece \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --wandb-project sinkhorn-wmt15 \
    --save-dir checkpoints/${TASK} \
    --no-epoch-checkpoints \
    --save-interval-updates 500 \
    --keep-interval-updates 5 \
    --keep-best-checkpoints 5 \
    --patience 50 \
    --log-format simple --log-interval 50 \
    --num-workers 8 \
    --seed 1 \
    --fp16
