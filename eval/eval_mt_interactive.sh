#!/usr/bin/env bash
TASK=teacher_cwmt_enzh
SPLIT=valid
EXP=../expcwmt
. ${EXP}/data_path.sh
CHECKDIR=${EXP}/checkpoints/${TASK}
AVG=true
SPM_PREFIX=${DATA}/spm_unigram32000
SRC_FILE=/livingrooms/george/cwmt/zh-en/prep/test.en-zh.${SRC}
# TGT_FILE=/livingrooms/george/cwmt/zh-en/prep/test.en-zh.${TGT}.1
BLEU_TOK=13a
RESULT=${TASK}.$(basename $(dirname $(dirname ${DATA})))
spm_encode=$FAIRSEQ/scripts/spm_encode.py

if [[ ${TGT} == "zh" ]]; then
  BLEU_TOK=zh
fi

if [[ $AVG == "true" ]]; then
  CHECKPOINT_FILENAME=avg_best_5_checkpoint.pt
  # python ../scripts/average_checkpoints.py \
  #   --inputs ${CHECKDIR} --num-best-checkpoints 5 \
  #   --output "${CHECKDIR}/${CHECKPOINT_FILENAME}"
else
  CHECKPOINT_FILENAME=checkpoint_best.pt
fi

GENARGS="--beam 5 --lenpen 1.5 --max-len-a 1.2 --max-len-b 10 --remove-bpe sentencepiece"

mkdir -p ${RESULT}

python $spm_encode \
  --model=${SPM_PREFIX}_${SRC}.model \
  --output_format=piece < ${SRC_FILE} | \
python -m fairseq_cli.interactive ${DATA} \
  -s ${SRC} -t ${TGT} \
  --user-dir ${USERDIR} \
  --gen-subset ${SPLIT} \
  --task translation \
  --path ${CHECKDIR}/${CHECKPOINT_FILENAME} \
  --buffer-size 2000 --batch-size 128 \
  ${GENARGS} > ${RESULT}/interactive-out.txt