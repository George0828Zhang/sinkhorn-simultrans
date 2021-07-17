#!/usr/bin/env bash
TASK=wait_1_deen_distill
SPLIT=valid
AGENT=./agents/simul_t2t_waitk.py
EXP=../exp
. ${EXP}/data_path.sh
CHECKDIR=${EXP}/checkpoints/${TASK}
CHECKPOINT_FILENAME=checkpoint_best.pt
SPM_MODEL=${DATA}/spm_unigram32000.model
LC=~/utility/mosesdecoder/scripts/tokenizer/lowercase.perl
SRC_FILE=$(realpath ${DATA}/../prep/${SPLIT}.${SRC})
TGT_FILE=$(realpath ${DATA}/../prep/${SPLIT}.${TGT})
# SRC_FILE=debug/test.de
# TGT_FILE=debug/test.en.lc
OUTPUT=${TASK}.${SRC}-${TGT}.results

if [ -f ${TGT_FILE}.lc ]; then
  echo "${TGT_FILE}.lc found, skipping lowercase."
else
  echo "lowercase to ${TGT_FILE}.lc ..."
  cat ${TGT_FILE} | perl $LC > ${TGT_FILE}.lc
  echo "done."
fi

simuleval \
  --agent ${AGENT} \
  --user-dir ${USERDIR} \
  --source ${SRC_FILE} \
  --target ${TGT_FILE}.lc \
  --data-bin ${DATA} \
  --model-path ${CHECKDIR}/${CHECKPOINT_FILENAME} \
  --src-splitter-path ${SPM_MODEL} \
  --tgt-splitter-path ${SPM_MODEL} \
  --output ${OUTPUT} \
  --sacrebleu-tokenizer 13a \
  --incremental-encoder \
  --force-finish \
  --scores \
  --test-waitk 1
