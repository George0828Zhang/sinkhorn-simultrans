#!/usr/bin/env bash
source ./data_path.sh
DELAY=$1
TASK=sinkhorn_delay${DELAY}_ft

python -m fairseq_cli.train ${DATA} --user-dir ${USERDIR} \
    -s ${SRC} -t ${TGT} \
    --load-pretrained-encoder-from checkpoints/ctc_delay${DELAY}/checkpoint_best.pt \
    --train-subset train_distill_${TGT} \
    --max-tokens 8000 \
    --update-freq 4 \
    --task translation_infer \
    --inference-config-yaml infer_mt.yaml \
    --arch sinkhorn_encoder --delay ${DELAY} --mask-ratio 0.5 \
    --sinkhorn-iters 16 --sinkhorn-tau 0.25 --sinkhorn-noise-factor 0.3 --sinkhorn-bucket-size 1 --sinkhorn-energy dot \
    --criterion label_smoothed_ctc --eos-loss --label-smoothing 0.1 --report-sinkhorn-dist --report-accuracy --decoder-use-ctc \
    --clip-norm 10.0 \
    --weight-decay 0.0001 \
    --optimizer adam --lr 0.0005 --lr-scheduler inverse_sqrt \
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
    --seed 1 \
    --fp16
