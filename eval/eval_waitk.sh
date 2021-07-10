#!/usr/bin/env bash
TASK=wait_3_iwslt_distill
SPLIT=valid
EXP=../exp
. ${EXP}/data_path.sh
CHECKDIR=${EXP}/checkpoints/${TASK}
AVG=true
# RESULT=./mt.results

EXTRAARGS="--scoring sacrebleu --sacrebleu-tokenizer 13a --sacrebleu-lowercase"
GENARGS="--beam 1 --max-len-a 1.2 --max-len-b 10 \
--remove-bpe sentencepiece --tokenizer moses -s ${SRC} -t ${TGT} --moses-no-escape"

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
  --task translation_infer \
  --path ${CHECKDIR}/${CHECKPOINT_FILENAME} --max-tokens 8000 --fp16 \
  --model-overrides '{"load_pretrained_encoder_from": None}' \
  ${GENARGS} ${EXTRAARGS}
  # --results-path ${RESULT} \