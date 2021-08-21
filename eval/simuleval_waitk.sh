#!/usr/bin/env bash
MODEL=$1
WAITK=$2
TRIAL=$3
PORT=12345
WORKERS=2
AGENT=./agents/simul_t2t_waitk.py
EXP=../expcwmt
source ${EXP}/data_path.sh

if [[ ${MODEL} == "waitk" ]]; then
  MODEL=wait_${WAITK}_${SRC}${TGT}_distill
elif [[ ${MODEL} == "sinkhorn" ]]; then
  MODEL=sinkhorn_wait${WAITK}_distill
else
  echo "no such file ${MODEL}_delay${DELAY}"
  exit
fi

TASK=${MODEL}
CHECKDIR=${EXP}/checkpoints/${TASK}
CHECKPOINT_FILENAME=checkpoint_best.pt
SPM_PREFIX=${DATA}/spm_unigram32000
SRC_FILE=/media/george/Data/cwmt/zh-en/prep/test.${SRC}-${TGT}.${SRC}
TGT_FILE=/media/george/Data/cwmt/zh-en/prep/test.${SRC}-${TGT}.${TGT}.1
# SRC_FILE=/media/george/Data/wmt15/de-en/prep/test.${SRC}
# TGT_FILE=/media/george/Data/wmt15/de-en/prep/test.${TGT}
BLEU_TOK=13a
UNIT=word
BASENAME=$(basename $(dirname $(dirname ${DATA})))
OUTPUT=${BASENAME}_${TGT}-results/${TASK}.${BASENAME}
mkdir -p ${OUTPUT}

if [[ ${TGT} == "zh" ]]; then
  BLEU_TOK=zh
  UNIT=char
  NO_SPACE="--no-space"
fi

simuleval \
  --agent ${AGENT} \
  --user-dir ${USERDIR} \
  --source ${SRC_FILE} \
  --target ${TGT_FILE} \
  --data-bin ${DATA} \
  --model-path ${CHECKDIR}/${CHECKPOINT_FILENAME} \
  --src-splitter-path ${SPM_PREFIX}_${SRC}.model \
  --tgt-splitter-path ${SPM_PREFIX}_${TGT}.model \
  --output ${OUTPUT} \
  --incremental-encoder \
  --sacrebleu-tokenizer ${BLEU_TOK} \
  --eval-latency-unit ${UNIT} \
  --segment-type ${UNIT} \
  ${NO_SPACE} \
  --scores \
  --test-waitk ${WAITK} \
  --port ${PORT} \
  --workers ${WORKERS}

mv ${OUTPUT}/scores ${OUTPUT}/scores.$TRIAL
