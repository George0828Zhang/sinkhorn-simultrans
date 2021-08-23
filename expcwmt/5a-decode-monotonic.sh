#!/usr/bin/env bash
. ./data_path.sh
TASK=wait_9_${SRC}${TGT}_distill
SPLIT=train
CHECKDIR=./checkpoints/${TASK}
AVG=false
RESULT=./monotonic.results

EXTRAARGS="--scoring sacrebleu --sacrebleu-tokenizer ${TGT} --sacrebleu-lowercase"
GENARGS="--beam 5 --max-len-a 1.2 --max-len-b 10 --remove-bpe sentencepiece"


if [[ $AVG == "true" ]]; then
  CHECKPOINT_FILENAME=avg_best_5_checkpoint.pt
  python ../scripts/average_checkpoints.py \
    --inputs ${CHECKDIR} --num-best-checkpoints 5 \
    --output "${CHECKDIR}/${CHECKPOINT_FILENAME}"
else
  CHECKPOINT_FILENAME=checkpoint_best.pt
fi

python -m fairseq_cli.generate ${DATA} \
  -s ${SRC} -t ${TGT} \
  --user-dir ${USERDIR} \
  --gen-subset ${SPLIT} \
  --skip-invalid-size-inputs-valid-test \
  --task translation_infer \
  --inference-config-yaml infer_monotonic.yaml \
  --path ${CHECKDIR}/${CHECKPOINT_FILENAME} \
  --max-tokens 8000 --fp16 \
  --results-path ${RESULT} \
  ${GENARGS} ${EXTRAARGS}