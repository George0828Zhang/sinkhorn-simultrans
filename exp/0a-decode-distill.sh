#!/usr/bin/env bash
TASK=teacher_wmt15_deen
SPLIT=train
. ./data_path.sh
CHECKDIR=./checkpoints/${TASK}
AVG=true
RESULT=./mt.results

EXTRAARGS="--scoring sacrebleu --sacrebleu-tokenizer 13a --sacrebleu-lowercase"
GENARGS="--beam 5 --max-len-a 1.2 --max-len-b 10 --remove-bpe sentencepiece"

export CUDA_VISIBLE_DEVICES=0

if [[ $AVG == "true" ]]; then
  CHECKPOINT_FILENAME=avg_best_5_checkpoint.pt
  python ../scripts/average_checkpoints.py \
    --inputs ${CHECKDIR} --num-best-checkpoints 5 \
    --output "${CHECKDIR}/${CHECKPOINT_FILENAME}"
else
  CHECKPOINT_FILENAME=checkpoint_best.pt
fi

python -m fairseq_cli.generate ${DATA} \
  --user-dir ${USERDIR} \
  --gen-subset ${SPLIT} \
  --skip-invalid-size-inputs-valid-test \
  --task translation \
  --path ${CHECKDIR}/${CHECKPOINT_FILENAME} \
  --max-tokens 16000 --fp16 \
  --results-path ${RESULT} \
  ${GENARGS} ${EXTRAARGS}
