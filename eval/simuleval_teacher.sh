#!/usr/bin/env bash
TRIAL=$1
PORT=23451
WORKERS=2
AGENT=./agents/simul_t2t_waitk.py
EXP=../expcwmt
. ${EXP}/data_path.sh

TASK=teacher_wmt15_${SRC}${TGT}
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
  --sacrebleu-tokenizer ${BLEU_TOK} \
  --eval-latency-unit ${UNIT} \
  --segment-type ${UNIT} \
  ${NO_SPACE} \
  --scores \
  --full-sentence \
  --port ${PORT} \
  --workers ${WORKERS}

mv ${OUTPUT}/scores ${OUTPUT}/scores.${TRIAL}
