#!/usr/bin/env bash
TASK=wait_3_iwslt
SPLIT=test
AGENT=./agents/simul_t2t_waitk.py
EXP=../exp
. ${EXP}/data_path.sh
CHECKDIR=${EXP}/checkpoints/${TASK}
AVG=true
SPM_MODEL=${DATA}/spm_unigram8000.model
SRC_FILE=$(realpath ${DATA}/../prep/${SPLIT}.${SRC})
TGT_FILE=$(realpath ${DATA}/../prep/${SPLIT}.${TGT})
# SRC_FILE=debug/test.de
# TGT_FILE=debug/test.en
OUTPUT=${TASK}.en-${TGT}.results

export CUDA_VISIBLE_DEVICES=0

if [[ $AVG == "true" ]]; then
  CHECKPOINT_FILENAME=avg_best_5_checkpoint.pt
  # python ../scripts/average_checkpoints.py \
  #   --inputs ${CHECKDIR} --num-best-checkpoints 5 \
  #   --output "${CHECKDIR}/${CHECKPOINT_FILENAME}"
else
  CHECKPOINT_FILENAME=checkpoint_best.pt
fi

simuleval \
  --agent ${AGENT} \
  --user-dir ${USERDIR} \
  --source ${SRC_FILE} \
  --target ${TGT_FILE} \
  --data-bin ${DATA} \
  --model-path ${CHECKDIR}/${CHECKPOINT_FILENAME} \
  --src-splitter-path ${SPM_MODEL} \
  --tgt-splitter-path ${SPM_MODEL} \
  --output ${OUTPUT} \
  --scores \
  --gpu \
  --test-waitk 3

# --sacrebleu-tokenizer ja-mecab \
#     --eval-latency-unit char \