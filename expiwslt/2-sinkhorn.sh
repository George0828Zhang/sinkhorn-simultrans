#!/usr/bin/env bash
TASK=sinkhorn_delay1_eos
. ./data_path.sh

python -m fairseq_cli.train ${DATA} --user-dir ${USERDIR} \
    -s ${SRC} -t ${TGT} \
    --train-subset train_distill \
    --max-tokens 8000 \
    --update-freq 2 \
    --task translation_infer \
    --inference-config-yaml infer_mt.yaml \
    --arch sinkhorn_encoder_iwslt --delay 1 --upsample-ratio 2 \
    --sinkhorn-iters 16 --sinkhorn-tau 0.1 --sinkhorn-noise-factor 0.25 --sinkhorn-bucket-size 1 --sinkhorn-energy dot \
    --criterion label_smoothed_ctc --eos-loss --label-smoothing 0.1 --report-sinkhorn-dist --report-accuracy --decoder-use-ctc \
    --clip-norm 10.0 \
    --weight-decay 0.0001 \
    --optimizer adam --lr 0.0005 --lr-scheduler inverse_sqrt \
    --warmup-updates 4000 \
    --max-update 50000 \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --wandb-project sinkhorn-iwslt14 \
    --save-dir checkpoints/${TASK} \
    --no-epoch-checkpoints \
    --save-interval-updates 500 \
    --keep-interval-updates 1 \
    --keep-best-checkpoints 5 \
    --patience 50 \
    --log-format simple --log-interval 50 \
    --num-workers 8 \
    --seed 1 \
    --fp16
