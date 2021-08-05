#!/usr/bin/env bash
WAITK=1
TASK=wait_${WAITK}_enzh_distill
AGENT=./agents/simul_t2t_waitk.py
EXP=../expcwmt
. ${EXP}/data_path.sh
CHECKDIR=${EXP}/checkpoints/${TASK}
CHECKPOINT_FILENAME=checkpoint_best.pt
SPM_PREFIX=${DATA}/spm_unigram32000
SRC_FILE=/media/george/Data/cwmt/zh-en/prep/test.en-zh.${SRC}
TGT_FILE=/media/george/Data/cwmt/zh-en/prep/test.en-zh.${TGT}.1
# SRC_FILE=debug/tiny.en
# TGT_FILE=debug/tiny.zh
BLEU_TOK=13a
UNIT=word
OUTPUT=${TASK}.$(basename $(dirname $(dirname ${DATA})))

if [[ ${TGT} == "zh" ]]; then
  BLEU_TOK=zh
  UNIT=char
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
  --no-space \
  --scores \
  --test-waitk ${WAITK}

# mv ${OUTPUT}/scores ${OUTPUT}/scores.$1
